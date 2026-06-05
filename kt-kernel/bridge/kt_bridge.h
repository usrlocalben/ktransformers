#ifndef KT_BRIDGE_H
#define KT_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 *  ktb_engine  —  NUMA-aware CPU inference pool
 * -------------------------------------------------------------------------- */

typedef struct ktb_engine* ktb_engine_t;

/**
 * Create a CPU inference engine.
 *
 * @param n_threads      Total number of CPU worker threads.
 * @param n_numa_nodes   Number of NUMA sub-pools.  Threads are distributed
 *                       round-robin (e.g. 96 threads / 8 NUMA = 12 per node).
 * @return Engine handle, or NULL on failure.
 */
ktb_engine_t ktb_engine_create(int n_threads, int n_numa_nodes);
void         ktb_engine_destroy(ktb_engine_t e);

/* --------------------------------------------------------------------------
 *  ktb_safetensor  —  read-only mmap of one or more .safetensors files
 * -------------------------------------------------------------------------- */

typedef struct ktb_safetensor* ktb_safetensor_t;

/**
 * Open a directory that contains .safetensors shard(s).
 *
 * All .safetensors files in the directory are mmap'd.  Tensors are looked up
 * by the original key name (e.g. "model.layers.0.mlp.experts.0.w1.weight").
 */
ktb_safetensor_t ktb_safetensor_open(const char* dir_path);
void             ktb_safetensor_close(ktb_safetensor_t st);

/**
 * Look up a tensor by exact key name.
 *
 * @return true if found, and fills *out_ptr / *out_bytes.
 *         The pointer is valid for the lifetime of @p st.
 */
bool ktb_safetensor_get(ktb_safetensor_t st, const char* name,
                        void** out_ptr, size_t* out_bytes);

/* --------------------------------------------------------------------------
 *  ktb_moe  —  single MoE layer backed by kt-kernel
 * -------------------------------------------------------------------------- */

typedef struct ktb_moe* ktb_moe_t;

typedef enum {
    KTB_MXFP4 = 0,
    KTB_FP8,
    KTB_BF16,
    KTB_FP8_PERCHANNEL,
    KTB_RAWINT4,
    KTB_GPTQ_INT4,
    KTB_AMXINT4,
    KTB_AMXINT8,
} ktb_method_t;

typedef enum {
    KTB_IO_BF16 = 0,
    KTB_IO_F32,
} ktb_io_type_t;

/**
 * Create a MoE layer (weights are NOT loaded yet).
 *
 * @param e                    Engine from ktb_engine_create().
 * @param layer_idx            Layer index (used for path / logging only).
 * @param num_experts          Total number of experts in this layer.
 * @param num_experts_per_tok  Top-k (e.g. 6 for DeepSeek V4).
 * @param hidden_size          Model hidden dimension (e.g. 7168).
 * @param intermediate_size    MoE intermediate size (e.g. 2048).
 * @param io_type              I/O datatype for activations (KTB_IO_BF16 or KTB_IO_F32).
 *                             Most methods only support KTB_IO_BF16; KTB_IO_F32 is
 *                             currently only valid with KTB_MXFP4.
 * @param method               FFN weight quantization / kernel method.
 * @param group_size           Quantization group size (e.g. 32 for MXFP4).
 *                             Pass 0 for methods that don't use grouped quantization.
 * @param swiglu_limit         Clamp value for SiLU(gate) * up activations.
 *                             Pass 0.0f to disable clamping.
 * @return MoE handle, or NULL on failure.
 */
ktb_moe_t ktb_moe_create(ktb_engine_t e,
                         int layer_idx,
                         int num_experts,
                         int num_experts_per_tok,
                         int hidden_size,
                         int intermediate_size,
                         ktb_io_type_t io_type,
                         ktb_method_t method,
                         int group_size,
                         float swiglu_limit);

void ktb_moe_destroy(ktb_moe_t m);

/**
 * Load weights from per-expert pointer arrays.
 *
 * All arrays have num_experts entries.  For MXFP4:
 *   gate_weight[i] → uint8_t[intermediate_size * hidden_size / 2] (nibbles)
 *   gate_scale[i]  → ggml_bf16_t[intermediate_size * hidden_size / group_size]
 *   group_size is passed explicitly.
 *
 * The arrays are not copied; they must remain valid until the next
 * ktb_moe_load_weights_* call.
 *
 * @param physical_to_logical_map  Optional mapping (num_experts), or NULL.
 * @return 0 on success, non-zero on error.
 */
int ktb_moe_load_weights_ptrs(ktb_moe_t m,
                              int group_size,
                              const void* const* gate_weight,
                              const void* const* gate_scale,
                              const void* const* up_weight,
                              const void* const* up_scale,
                              const void* const* down_weight,
                              const void* const* down_scale,
                              const int64_t* physical_to_logical_map);

/**
 * Convenience: load from a directory of .safetensors.
 *
 * The directory is scanned and the V4-inference key layout is probed:
 *   model.layers.{L}.mlp.experts.{E}.w1.weight
 *   model.layers.{L}.mlp.experts.{E}.w1.scale
 *   ... w3 (up), w2 (down)
 *
 * If no experts are found under the "model." prefix the "layers." prefix
 * is tried as a fallback.
 */
int ktb_moe_load_weights_safetensors(ktb_moe_t m,
                                     ktb_safetensor_t st,
                                     int layer_idx,
                                     const int64_t* physical_to_logical_map);

/**
 * Run a forward step.
 *
 * @param n_tokens     Number of tokens (qlen).
 * @param input        BF16    [n_tokens, hidden_size]
 * @param output       BF16    [n_tokens, hidden_size]  (out-of-place)
 * @param expert_ids   int64   [n_tokens, num_experts_per_tok]
 * @param weights      float   [n_tokens, num_experts_per_tok]
 * @param incremental  If true the output is accumulated instead of zeroed.
 */
void ktb_moe_forward(ktb_moe_t m,
                     int n_tokens,
                     const void*  input,
                     void*        output,
                     const int64_t* expert_ids,
                     const float*   weights,
                     bool incremental,
                     float swiglu_limit);

/**
 * Same as ktb_moe_forward but accepts float32 I/O and int32 expert IDs.
 * When the MoE was created with KTB_IO_F32 (f32-native kernel), data is passed
 * directly without any conversion. For KTB_IO_BF16 MoE layers,
 * the bridge allocates internal scratch, converts f32->bf16 and int32->int64,
 * runs the kernel, then converts the output back to f32.
 * This is the function ds4 will call to keep its existing float32 tensor layout.
 */
void ktb_moe_forward_f32(ktb_moe_t m,
                         int n_tokens,
                         const void*  input_f32,        /* float [n_tokens, hidden_size] */
                         void*        output_f32,       /* float [n_tokens, hidden_size] */
                         const int32_t* expert_ids_i32, /* int32 [n_tokens, num_experts_per_tok] */
                         const float*   weights,
                         bool incremental,
                         float swiglu_limit);

/* --------------------------------------------------------------------------
 *  Helpers for ds4 interop
 * -------------------------------------------------------------------------- */

/**
 * Convert a buffer of float -> bf16 in-place.
 */
void ktb_convert_f32_to_bf16(const float* src, void* dst, size_t n);

/**
 * Convert a buffer of bf16 -> float in-place.
 */
void ktb_convert_bf16_to_f32(const void* src, float* dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif
