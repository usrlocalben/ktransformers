/**
 * C bridge for kt-kernel  —  exposes CPUInfer + TP_MOE<T> as a plain C API.
 *
 * This file lives in ktransformers/kt-kernel/bridge/.
 * Compile it as a shared library alongside the main kt_kernel_ext module
 * so that external C projects (e.g. DwarfStar) can call the x86/NUMA MoE
 * kernels without embedding Python.
 */

#include "kt_bridge.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <unordered_map>

#include "llama.cpp/common/json.hpp"

#include "cpu_backend/cpuinfer.h"
#include "operators/common.hpp"

/* ------------------------------------------------------------------ */
/*  AMX / AVX512 backends                                             */
/* ------------------------------------------------------------------ */

#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)
#include "operators/amx/la/amx_kernels.hpp"
#include "operators/amx/moe.hpp"
#include "operators/amx/fp4-moe.hpp"
#include "operators/amx/fp8-moe.hpp"
#include "operators/amx/fp8-perchannel-moe.hpp"
#include "operators/amx/bf16-moe.hpp"
#include "operators/amx/k2-moe.hpp"
#include "operators/amx/moe_base.hpp"
#endif

/* ------------------------------------------------------------------ */
/*  AVX2                                                              */
/* ------------------------------------------------------------------ */

#if defined(__x86_64__)
#include "operators/avx2/bf16-moe.hpp"
#include "operators/avx2/fp8-moe.hpp"
#include "operators/avx2/gptq_int4-moe.hpp"
#include "operators/avx2/rawint4-moe.hpp"
#endif

/* ------------------------------------------------------------------ */
/*  ktb_engine                                                        */
/* ------------------------------------------------------------------ */

struct ktb_engine {
    std::unique_ptr<CPUInfer> cpuinfer;
    WorkerPoolConfig          wcfg;
};

ktb_engine_t ktb_engine_create(int n_threads, int n_numa_nodes) {
    if (n_threads <= 0 || n_numa_nodes <= 0) return nullptr;

    WorkerPoolConfig wcfg;
    wcfg.subpool_count = n_numa_nodes;
    for (int i = 0; i < n_numa_nodes; ++i) {
        wcfg.subpool_numa_map.push_back(i);
    }
    int base = n_threads / n_numa_nodes;
    int rem  = n_threads % n_numa_nodes;
    for (int i = 0; i < n_numa_nodes; ++i) {
        wcfg.subpool_thread_count.push_back(base + (i < rem ? 1 : 0));
    }

    try {
        auto e = new ktb_engine;
        e->wcfg = wcfg;
        e->cpuinfer = std::make_unique<CPUInfer>(wcfg);
        return e;
    } catch (const std::exception& ex) {
        fprintf(stderr, "[kt_bridge] engine create failed: %s\n", ex.what());
        return nullptr;
    }
}

void ktb_engine_destroy(ktb_engine_t e) {
    delete e;
}

/* ------------------------------------------------------------------ */
/*  ktb_safetensor                                                    */
/* ------------------------------------------------------------------ */

struct safetensor_file {
    std::string         path;
    int                 fd = -1;
    size_t              file_size = 0;
    const uint8_t*      mmap_base = nullptr;
    uint64_t            data_offset = 0;
    nlohmann::json      header;
};

struct ktb_safetensor {
    std::vector<std::unique_ptr<safetensor_file>> files;
    std::unordered_map<std::string, safetensor_file*> tensor_to_file;
    std::unordered_map<std::string, size_t>           tensor_to_file_offset;
};

static bool read_file_header(safetensor_file& f) {
    if (f.file_size < 8) return false;
    uint64_t header_len = 0;
    std::memcpy(&header_len, f.mmap_base, sizeof(header_len));
    if (8 + header_len > f.file_size) return false;
    try {
        f.header = nlohmann::json::parse(
            reinterpret_cast<const char*>(f.mmap_base + 8),
            reinterpret_cast<const char*>(f.mmap_base + 8 + header_len));
    } catch (...) {
        return false;
    }
    f.data_offset = 8 + header_len;
    return true;
}

ktb_safetensor_t ktb_safetensor_open(const char* dir_path) {
    std::vector<std::string> paths;
    DIR* d = opendir(dir_path);
    if (!d) {
        fprintf(stderr, "[kt_bridge] cannot open dir: %s\n", dir_path);
        return nullptr;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() > 12 &&
            name.substr(name.size() - 12) == ".safetensors") {
            paths.emplace_back(std::string(dir_path) + "/" + name);
        }
    }
    closedir(d);
    if (paths.empty()) {
        fprintf(stderr, "[kt_bridge] no .safetensors found in %s\n", dir_path);
        return nullptr;
    }
    std::sort(paths.begin(), paths.end());

    auto st = new ktb_safetensor;
    for (const auto& p : paths) {
        auto f = std::make_unique<safetensor_file>();
        f->path = p;
        f->fd = open(p.c_str(), O_RDONLY);
        if (f->fd < 0) continue;
        struct stat sb;
        if (fstat(f->fd, &sb) < 0) { close(f->fd); f->fd = -1; continue; }
        f->file_size = sb.st_size;
        f->mmap_base = reinterpret_cast<const uint8_t*>(
            mmap(nullptr, f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0));
        if (f->mmap_base == MAP_FAILED) { close(f->fd); f->fd = -1; continue; }

        if (!read_file_header(*f)) {
            munmap((void*)f->mmap_base, f->file_size);
            close(f->fd);
            f->fd = -1;
            continue;
        }

        for (auto& [key, val] : f->header.items()) {
            if (key == "__metadata__") continue;
            if (!val.is_object() || !val.contains("data_offsets")) continue;
            auto& off = val["data_offsets"];
            if (!off.is_array() || off.size() != 2) continue;
            size_t start = off[0].get<size_t>();
            size_t end   = off[1].get<size_t>();
            if (end <= start) continue;
            std::string full_key = key;
            st->tensor_to_file[full_key] = f.get();
            st->tensor_to_file_offset[full_key] = start;
        }
        st->files.push_back(std::move(f));
    }
    if (st->files.empty()) {
        delete st;
        return nullptr;
    }
    return st;
}

void ktb_safetensor_close(ktb_safetensor_t st) {
    for (auto& f : st->files) {
        if (f->mmap_base) munmap((void*)f->mmap_base, f->file_size);
        if (f->fd >= 0)   close(f->fd);
    }
    delete st;
}

bool ktb_safetensor_get(ktb_safetensor_t st, const char* name,
                        void** out_ptr, size_t* out_bytes) {
    auto it = st->tensor_to_file.find(name);
    if (it == st->tensor_to_file.end()) return false;
    auto fit = st->tensor_to_file_offset.find(name);
    if (fit == st->tensor_to_file_offset.end()) return false;
    safetensor_file* f = it->second;
    size_t start = fit->second;
    auto& info = f->header[name];
    size_t end = info["data_offsets"][1].get<size_t>();
    *out_ptr  = const_cast<uint8_t*>(f->mmap_base + f->data_offset + start);
    *out_bytes = end - start;
    return true;
}

/* ------------------------------------------------------------------ */
/*  ktb_moe  —  type-erased wrapper                                   */
/* ------------------------------------------------------------------ */

struct ktb_moe {
    /* Type-erased lifetime guard.
     * The concrete std::shared_ptr<TP_MOE<...>> is captured inside this
     * lambda so that the correct (non-virtual) destructor is called
     * when the bridge object is freed.                                   */
    std::function<void()>  on_destroy;
    MoE_Interface*         iface = nullptr;
    std::function<void()>  load_weights_fn;
    GeneralMOEConfig       config;

    /* For the per-expert pointer mode we stash copies of the
     * pointer vectors so that gate_projs / gate_scales etc.
     * remain valid as long as the ktb_moe exists.               */
    std::vector<std::vector<void*>> gate_projs;
    std::vector<std::vector<void*>> up_projs;
    std::vector<std::vector<void*>> down_projs;
    std::vector<std::vector<void*>> gate_scales;
    std::vector<std::vector<void*>> up_scales;
    std::vector<std::vector<void*>> down_scales;

    /* For MXFP4 the .safetensors store scales as F8_E8M0 (1 byte) but the
     * AMX kernels expect bf16 (2 bytes).  We convert on load and keep the
     * temporary buffers here so they live as long as the ktb_moe.       */
    std::vector<std::unique_ptr<uint16_t[]>> scale_temps;
};

/* Helper to fill the global std::vectors from flat C arrays. */
static void set_projs(std::vector<std::vector<void*>>& dst,
                      const void* const* src, int n) {
    dst.resize(1);
    dst[0].resize(n);
    for (int i = 0; i < n; ++i) dst[0][i] = const_cast<void*>(src[i]);
}

#define DISPATCH_AMX_CASE(T, C) \
    case C: { \
        using M = T; \
        auto concrete = std::make_shared<TP_MOE<M>>(cfg); \
        m->iface = concrete.get(); \
        m->load_weights_fn = [concrete, m]() { \
            concrete->config = m->config; \
            concrete->load_weights(); \
        }; \
        m->on_destroy = [concrete]() { }; \
    } break;

#define DISPATCH_AVX2_CASE(T, C) \
    case C: { \
        using M = T; \
        auto concrete = std::make_shared<TP_MOE<M>>(cfg); \
        m->iface = concrete.get(); \
        m->load_weights_fn = [concrete, m]() { \
            concrete->config = m->config; \
            concrete->load_weights(); \
        }; \
        m->on_destroy = [concrete]() { }; \
    } break;

ktb_moe_t ktb_moe_create(ktb_engine_t e,
                         int layer_idx,
                          int num_experts,
                          int num_experts_per_tok,
                          int hidden_size,
                          int intermediate_size,
                          ktb_method_t method,
                          float swiglu_limit) {
    if (!e || !e->cpuinfer) return nullptr;

    GeneralMOEConfig cfg(num_experts, num_experts_per_tok, hidden_size, intermediate_size);
    cfg.layer_idx = layer_idx;
    cfg.pool = e->cpuinfer->backend_;
    cfg.max_len = 8192;
    cfg.swiglu_limit = swiglu_limit;
    cfg.gpu_experts_mask = nullptr;
    cfg.num_gpu_experts = 0;

    /* group_size must be non-zero for int4/fp4/8 BuffersB that do k/group_size.  The
     * actual value is overwrriten by ktb_moe_load_weights_*.                 */
    if (method == KTB_MXFP4 || method == KTB_FP8 || method == KTB_FP8_PERCHANNEL ||
        method == KTB_RAWINT4 || method == KTB_AMXINT4 || method == KTB_GPTQ_INT4) {
        cfg.quant_config.group_size = 32;
    }

    auto m = new ktb_moe;
    m->config = cfg;

    try {
#if defined(__x86_64__) && defined(USE_AMX_AVX_KERNEL)
        switch (method) {
            DISPATCH_AMX_CASE(AMX_FP4_MOE_TP<amx::GemmKernel224MXFP4SmallKGroup>, KTB_MXFP4)
            DISPATCH_AMX_CASE(AMX_FP8_MOE_TP<amx::GemmKernel224FP8>,           KTB_FP8)
            DISPATCH_AMX_CASE(AMX_BF16_MOE_TP<amx::GemmKernel224BF16>,         KTB_BF16)
            DISPATCH_AMX_CASE(AMX_FP8_PERCHANNEL_MOE_TP<amx::GemmKernel224FP8PerChannel>, KTB_FP8_PERCHANNEL)
            DISPATCH_AMX_CASE(AMX_K2_MOE_TP<amx::GemmKernel224Int4SmallKGroup>, KTB_RAWINT4)
            DISPATCH_AMX_CASE(AMX_MOE_TP<amx::GemmKernel224Int4>,              KTB_AMXINT4)
            DISPATCH_AMX_CASE(AMX_MOE_TP<amx::GemmKernel224Int8>,              KTB_AMXINT8)
            default:
#if defined(__x86_64__)
                goto avx2_fallback;
#else
                throw std::runtime_error("Method not available on this platform");
#endif
        }
        if (m->iface) return m;  // AMX/AVX path succeeded, don't fall into AVX2
#else
        goto avx2_fallback;
#endif

#if defined(__x86_64__)
    avx2_fallback:
        switch (method) {
            DISPATCH_AVX2_CASE(AVX2_BF16_MOE_TP<avx2::GemmKernelAVX2BF16>,     KTB_BF16)
            DISPATCH_AVX2_CASE(AVX2_FP8_MOE_TP<avx2::GemmKernelAVX2FP8>,        KTB_FP8)
            DISPATCH_AVX2_CASE(AVX2_GPTQ_INT4_MOE_TP<avx2::GemmKernelAVX2GPTQInt4>, KTB_GPTQ_INT4)
            DISPATCH_AVX2_CASE(AVX2_RAW_INT4_MOE_TP<avx2::GemmKernelAVX2RawInt4>,   KTB_RAWINT4)
            default:
                throw std::runtime_error("Method not available on AVX2");
        }
#endif
    } catch (const std::exception& ex) {
        fprintf(stderr, "[kt_bridge] moe create (layer %d) failed: %s\n", layer_idx, ex.what());
        delete m;
        return nullptr;
    }
    return m;
}

#undef DISPATCH_AMX_CASE
#undef DISPATCH_AVX2_CASE

void ktb_moe_destroy(ktb_moe_t m) {
    delete m;
}

int ktb_moe_load_weights_ptrs(ktb_moe_t m,
                              int group_size,
                              const void* const* gate_weight,
                              const void* const* gate_scale,
                              const void* const* up_weight,
                              const void* const* up_scale,
                              const void* const* down_weight,
                              const void* const* down_scale,
                              const int64_t* physical_to_logical_map) {
    if (!m || !m->iface) return -1;

    int E = m->config.expert_num;
    std::vector<const void*> nulls(E, nullptr);

    set_projs(m->gate_projs,  gate_weight,  E);
    set_projs(m->up_projs,    up_weight,    E);
    set_projs(m->down_projs,  down_weight,  E);
    set_projs(m->gate_scales, gate_scale ? gate_scale : nulls.data(), E);
    set_projs(m->up_scales,   up_scale   ? up_scale   : nulls.data(), E);
    set_projs(m->down_scales, down_scale ? down_scale : nulls.data(), E);

    m->config.quant_config.group_size = group_size;
    m->config.quant_config.zero_point = false;

    m->config.gate_projs  = m->gate_projs;
    m->config.up_projs    = m->up_projs;
    m->config.down_projs  = m->down_projs;
    m->config.gate_scales = m->gate_scales;
    m->config.up_scales   = m->up_scales;
    m->config.down_scales = m->down_scales;

    if (physical_to_logical_map) {
        m->config.physical_to_logical_map = const_cast<void*>(static_cast<const void*>(physical_to_logical_map));
    } else {
        /* identity */
        static std::vector<uint64_t> id_map;
        id_map.resize(E);
        for (int i = 0; i < E; ++i) id_map[i] = i;
        m->config.physical_to_logical_map = id_map.data();
    }

    try {
        m->load_weights_fn();
    } catch (const std::exception& ex) {
        fprintf(stderr, "[kt_bridge] load_weights failed: %s\n", ex.what());
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Safetensors loading (V4 inference format)                         */
/* ------------------------------------------------------------------ */

int ktb_moe_load_weights_safetensors(ktb_moe_t m,
                                     ktb_safetensor_t st,
                                     int layer_idx,
                                     const int64_t* physical_to_logical_map) {
    if (!m || !st) return -1;

    m->scale_temps.clear();

    int E = m->config.expert_num;
    std::vector<void*> gate_w(E), up_w(E), down_w(E);
    std::vector<void*> gate_s(E), up_s(E), down_s(E);

    /*  DeepSeek-V4-Flash stores .scale as F8_E8M0 (1 byte per scale)
     *  but the AMX MXFP4 kernels expect ggml_bf16_t (2 bytes).         */
    auto maybe_convert = [st, m](const char* key, void*& ptr, size_t bytes) {
        if (bytes == 0) return;
        auto it = st->tensor_to_file.find(key);
        if (it == st->tensor_to_file.end()) return;
        auto& info = it->second->header[key];
        if (!info.is_object() || !info.contains("dtype")) return;
        std::string dtype = info["dtype"].get<std::string>();
        if (dtype == "F8_E8M0") {
            const uint8_t* s = (const uint8_t*)ptr;
            uint16_t* d = new uint16_t[bytes];
            for (size_t i = 0; i < bytes; ++i) d[i] = (uint16_t)(s[i]) << 7;
            m->scale_temps.emplace_back(std::unique_ptr<uint16_t[]>(d));
            ptr = d;
        }
    };

    auto try_load = [&](const char* prefix) -> bool {
        for (int e = 0; e < E; ++e) {
            char wname[256], sname[256];
            std::snprintf(wname, sizeof(wname), "%s.ffn.experts.%d.w1.weight", prefix, e);
            std::snprintf(sname, sizeof(sname), "%s.ffn.experts.%d.w1.scale",  prefix, e);
            size_t wb = 0, sb = 0;
            if (!ktb_safetensor_get(st, wname, &gate_w[e], &wb)) return false;
            if (!ktb_safetensor_get(st, sname, &gate_s[e], &sb)) return false;
            maybe_convert(sname, gate_s[e], sb);

            std::snprintf(wname, sizeof(wname), "%s.ffn.experts.%d.w3.weight", prefix, e);
            std::snprintf(sname, sizeof(sname), "%s.ffn.experts.%d.w3.scale",  prefix, e);
            if (!ktb_safetensor_get(st, wname, &up_w[e], &wb)) return false;
            if (!ktb_safetensor_get(st, sname, &up_s[e], &sb)) return false;
            maybe_convert(sname, up_s[e], sb);

            std::snprintf(wname, sizeof(wname), "%s.ffn.experts.%d.w2.weight", prefix, e);
            std::snprintf(sname, sizeof(sname), "%s.ffn.experts.%d.w2.scale",  prefix, e);
            if (!ktb_safetensor_get(st, wname, &down_w[e], &wb)) return false;
            if (!ktb_safetensor_get(st, sname, &down_s[e], &sb)) return false;
            maybe_convert(sname, down_s[e], sb);
        }
        return true;
    };

    bool ok = false;
    std::string p1 = std::string("model.layers.") + std::to_string(layer_idx);
    std::string p2 = std::string("layers.") + std::to_string(layer_idx);

    if (try_load(p1.c_str())) {
        ok = true;
    } else if (try_load(p2.c_str())) {
        ok = true;
    }
    if (!ok) {
        fprintf(stderr, "[kt_bridge] no V4-format MXFP4 experts found for layer %d\n", layer_idx);
        return -1;
    }

    /* Auto-detect group_size from the first scale tensor.
     * MXFP4: scale is [N, K / group_size]  where N=intermediate_size, K=hidden_size (gate/up)
     *        or [K, N / group_size]        for down.
     * We load the shape from the JSON to compute it.                       */
    int group_size = 32; /* MXFP4 default */
    {
        for (const auto& prefix : { p1, p2 }) {
            auto it = st->tensor_to_file.find(
                prefix + ".ffn.experts.0.w1.scale");
            if (it == st->tensor_to_file.end()) continue;
            try {
                auto& info = it->second->header[prefix + ".ffn.experts.0.w1.scale"];
                auto& shape = info["shape"];
                if (shape.is_array() && shape.size() == 2) {
                    int cols = shape[1].get<int>();
                    if (cols != 0) {
                        group_size = m->config.hidden_size / cols;
                        break;
                    }
                }
            } catch (...) {}
        }
    }

    return ktb_moe_load_weights_ptrs(m, group_size,
                                     gate_w.data(), gate_s.data(),
                                     up_w.data(),   up_s.data(),
                                     down_w.data(), down_s.data(),
                                     physical_to_logical_map);
}

/* ------------------------------------------------------------------ */
/*  Forward                                                           */
/* ------------------------------------------------------------------ */

void ktb_moe_forward(ktb_moe_t m,
                     int n_tokens,
                     const void* input,
                     void* output,
                     const int64_t* expert_ids,
                     const float* weights,
                     bool incremental,
                     float swiglu_limit) {
    if (!m || !m->iface) return;
    try {
        m->config.swiglu_limit = swiglu_limit;
        m->iface->forward(n_tokens,
                          m->config.num_experts_per_tok,
                          expert_ids,
                          weights,
                          input,
                          output,
                          incremental);
    } catch (const std::exception& ex) {
        fprintf(stderr, "[kt_bridge] forward failed: %s\n", ex.what());
    }
}

/* ------------------------------------------------------------------ */
/*  f32 convenience wrapper                                           */
/* ------------------------------------------------------------------ */

void ktb_moe_forward_f32(ktb_moe_t m,
                         int n_tokens,
                         const void*  input_f32,
                         void*        output_f32,
                         const int32_t* expert_ids_i32,
                         const float*   weights,
                         bool incremental,
                         float swiglu_limit) {
    if (!m || !m->iface) return;

    int H = m->config.hidden_size;
    int K = m->config.num_experts_per_tok;
    size_t n_elem = (size_t)n_tokens * H;

    /* Thread-local scratch (assume caller is single-threaded). */
    thread_local std::vector<ggml_bf16_t> scratch_in;
    thread_local std::vector<ggml_bf16_t> scratch_out;
    thread_local std::vector<int64_t>     scratch_ids;
    if (scratch_in.size() < n_elem)  scratch_in.resize(n_elem);
    if (scratch_out.size() < n_elem) scratch_out.resize(n_elem);
    if (scratch_ids.size() < (size_t)n_tokens * K) scratch_ids.resize((size_t)n_tokens * K);

    ktb_convert_f32_to_bf16((const float*)input_f32,  scratch_in.data(),  n_elem);
    for (int i = 0; i < n_tokens * K; ++i) {
        scratch_ids[i] = expert_ids_i32[i];
    }

    ktb_moe_forward(m, n_tokens, scratch_in.data(), scratch_out.data(),
                    scratch_ids.data(), weights, incremental, swiglu_limit);

    ktb_convert_bf16_to_f32(scratch_out.data(), (float*)output_f32, n_elem);
}

/* ------------------------------------------------------------------ */
/*  Conversion helpers                                                */
/* ------------------------------------------------------------------ */

void ktb_convert_f32_to_bf16(const float* src, void* dst, size_t n) {
    /* llama.cpp's ggml_fp32_to_bf16_row assumes 16-byte alignment on src/dst
     * and may segfault with GPU-mapped pointers.  Use the scalar fallback. */
    ggml_bf16_t* d = static_cast<ggml_bf16_t*>(dst);
    for (size_t i = 0; i < n; ++i) {
        d[i] = ggml_fp32_to_bf16(src[i]);
    }
}

void ktb_convert_bf16_to_f32(const void* src, float* dst, size_t n) {
    const ggml_bf16_t* s = static_cast<const ggml_bf16_t*>(src);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ggml_bf16_to_fp32(s[i]);
    }
}

