/**
 * @Description  : MXFP4 MoE operator — FP4 E2M1 weights × F32 activations
 *                 F32-native variant: no bf16 conversion at kernel boundaries.
 *                 Weights are still FP4 E2M1 nibble-packed; dequant is fused
 *                 into the dot-product loop as FP4→F32 via permute LUT.
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-06-04
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Based on fp4-moe.hpp (BF16 variant).  Key differences:
 *   Activation: F32 direct (no BF16 BufferA, no bf16→f32 shim in act_fn)
 *   Dot prod:   _mm512_fmadd_ps (F32×F32→F32) via permute LUT
 *   Weight:     FP4 E2M1 (same nibble-packed layout, same BufferB)
 *   Output:     F32 direct (no bf16 conversion in to_mat)
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_F32_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_F32_H

#include "la/amx_raw_buffers.hpp"
#include "moe_base.hpp"

namespace amx {

// ============================================================================
// F32 activation buffer — stores float instead of ggml_bf16_t
// ============================================================================
template <typename K>
struct BufferAF32Impl {
  float* a;
  int max_m, k;
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k) { return sizeof(float) * max_m * k; }

  BufferAF32Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    a = reinterpret_cast<float*>(ptr);
  }

  void set_data(void* new_ptr) { a = reinterpret_cast<float*>(new_ptr); }

  void from_mat(int m, float* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            float* s = src + (m_begin + i) * k + k_block_begin + k_begin;
            float* d = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
            // Store in even/odd-split layout so ActivationF32 needs no permute:
            //   d[0..15]  = s[0],s[2],...,s[30]  (even columns)
            //   d[16..31] = s[1],s[3],...,s[31]  (odd columns)
            alignas(64) static const int32_t ev_idx[16] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30};
            alignas(64) static const int32_t od_idx[16] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};
            __m512 v0 = _mm512_loadu_ps(s);
            __m512 v1 = _mm512_loadu_ps(s + 16);
            _mm512_storeu_ps(d,      _mm512_permutex2var_ps(v0, _mm512_load_si512(ev_idx), v1));
            _mm512_storeu_ps(d + 16, _mm512_permutex2var_ps(v0, _mm512_load_si512(od_idx), v1));
          }
        }
      }
    }
  }

  float* get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }
};

// ============================================================================
// F32 output buffer — stores float, to_mat writes float (no bf16 conversion)
// ============================================================================
template <typename K>
struct BufferCF32Impl {
  float* c;
  int max_m, n;
  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

  BufferCF32Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<float*>(ptr);
  }

  void set_data(void* new_ptr) { c = reinterpret_cast<float*>(new_ptr); }

  void to_mat(int m, float* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          float* s = c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP;
          float* d = dst + (m_begin + i) * n + n_block_begin + n_begin;
          for (int j = 0; j < N_STEP; j++) d[j] = s[j];
        }
      }
    }
  }

  float* get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }
};

// ============================================================================
// MXFP4 F32 kernel: FP4 E2M1 weights × F32 activations → F32 output (AVX512)
// ============================================================================
struct GemmKernel224MXFP4SmallKGroupF32 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "MXFP4_F32"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  // FP4 E2M1 → FP32 LUT (16 entries, for VPERMILPS within 512-bit lanes)
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  inline static const __m512 fp4_f32_lut = _mm512_setr_ps(
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);

  // Dequantize 16 packed FP4 bytes (32 nibbles = 1 K-group) → 32 FP32 values
  struct DequantizedWeightF32 {
    __m512 w_even;
    __m512 w_odd;
    inline static const __m128i lo_mask = _mm_set1_epi8(0x0F);

    __attribute__((always_inline)) DequantizedWeightF32(__m128i w) {
      __m128i lo = _mm_and_si128(w, lo_mask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), lo_mask);
      __m512i lo_32 = _mm512_cvtepu8_epi32(lo);
      __m512i hi_32 = _mm512_cvtepu8_epi32(hi);
      w_even = _mm512_permutexvar_ps(lo_32, fp4_f32_lut);
      w_odd  = _mm512_permutexvar_ps(hi_32, fp4_f32_lut);
    }
  };

  // F32 activation — BufferA stores even/odd pre-split, so just two loads.
  struct ActivationF32 {
    __m512 a_even;
    __m512 a_odd;

    __attribute__((always_inline)) ActivationF32(const float* ptr) {
      a_even = _mm512_loadu_ps(ptr);       // d[0..15]  = even columns
      a_odd  = _mm512_loadu_ps(ptr + 16);  // d[16..31] = odd columns
    }
  };

  // F32 × F32 dot product — matches original non-BF16 mxfp4_dot_bf16
  __attribute__((always_inline)) static inline __m512 mxfp4_dot_f32(const DequantizedWeightF32& w,
                                                                      const ActivationF32& act) {
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
  }

  // Buffers
  using BufferA = BufferAF32Impl<GemmKernel224MXFP4SmallKGroupF32>;
  using BufferB = BufferBInt4KGroupImpl<GemmKernel224MXFP4SmallKGroupF32>;
  using BufferC = BufferCF32Impl<GemmKernel224MXFP4SmallKGroupF32>;

  // 4-way horizontal reduce
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // mat-vec: M independent tokens, N dimension processed 4 rows at a time
  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      float* a_row = ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationF32 a(a_row + g * 32);
          const DequantizedWeightF32 d0(w0[g]);
          const DequantizedWeightF32 d1(w1[g]);
          const DequantizedWeightF32 d2(w2[g]);
          const DequantizedWeightF32 d3(w3[g]);
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp4_dot_f32(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp4_dot_f32(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp4_dot_f32(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp4_dot_f32(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationF32 a(a_row + g * 32);
          const DequantizedWeightF32 d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_f32(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

  // mat-mat: 4×4 register tile
  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      float* a_rows[MB] = {
          ba->get_submat(m, k, m_pos + 0, 0),
          ba->get_submat(m, k, m_pos + 1, 0),
          ba->get_submat(m, k, m_pos + 2, 0),
          ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const DequantizedWeightF32 d0(w0[g]);
          const DequantizedWeightF32 d1(w1[g]);
          const DequantizedWeightF32 d2(w2[g]);
          const DequantizedWeightF32 d3(w3[g]);
          const __m512 sv0 = _mm512_set1_ps(s0[g]);
          const __m512 sv1 = _mm512_set1_ps(s1[g]);
          const __m512 sv2 = _mm512_set1_ps(s2[g]);
          const __m512 sv3 = _mm512_set1_ps(s3[g]);

#define V_FMA_ROW_F32(M_I)                                                      \
  do {                                                                           \
    const ActivationF32 a(a_rows[M_I] + g * 32);                                \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp4_dot_f32(d0, a), acc[M_I][0]);      \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp4_dot_f32(d1, a), acc[M_I][1]);      \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp4_dot_f32(d2, a), acc[M_I][2]);      \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp4_dot_f32(d3, a), acc[M_I][3]);      \
  } while (0)
          V_FMA_ROW_F32(0);
          V_FMA_ROW_F32(1);
          V_FMA_ROW_F32(2);
          V_FMA_ROW_F32(3);
#undef V_FMA_ROW_F32
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationF32 a(a_rows[i] + g * 32);
            const DequantizedWeightF32 d(w[g]);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_f32(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
        }
      }
    }
    for (int mi = m_pos; mi < m; mi++) {
      float* c_row = bc->get_submat(m, n, mi, n_start);
      float* a_row = ba->get_submat(m, k, mi, 0);
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationF32 a(a_row + g * 32);
          const DequantizedWeightF32 d0(w0[g]);
          const DequantizedWeightF32 d1(w1[g]);
          const DequantizedWeightF32 d2(w2[g]);
          const DequantizedWeightF32 d3(w3[g]);
          a0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp4_dot_f32(d0, a), a0);
          a1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp4_dot_f32(d1, a), a1);
          a2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp4_dot_f32(d2, a), a2);
          a3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp4_dot_f32(d3, a), a3);
        }
        reduce4(a0, a1, a2, a3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationF32 a(a_row + g * 32);
          const DequantizedWeightF32 d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_f32(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }
};

// Dispatch functions for F32 kernel
inline void vec_mul_kgroup_f32(int m, int n, int k, int k_group_size,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferA> ba,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferB> bb,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroupF32::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup_f32(int m, int n, int k, int k_group_size,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferA> ba,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferB> bb,
                               std::shared_ptr<GemmKernel224MXFP4SmallKGroupF32::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroupF32::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// F32 MoE base class — uses float intermediate buffers throughout
// ============================================================================
template <class T, class Derived>
class AMX_MOE_BASE_F32 {
 public:
  int tp_part_idx = 0;

  float* m_local_input_ = nullptr;
  float* m_local_gate_output_ = nullptr;
  float* m_local_up_output_ = nullptr;
  float* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<float*> m_local_input_ptr_;
  std::vector<float*> m_local_gate_output_ptr_;
  std::vector<float*> m_local_up_output_ptr_;
  std::vector<float*> m_local_down_output_ptr_;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  GeneralMOEConfig config_;
  using input_t = float;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_MOE_BASE_F32(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    MemoryRequest mem_requests;
    const size_t ml = config_.max_len;
    const size_t k_tok = config_.num_experts_per_tok;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;
    mem_requests.append_pointer(&m_local_input_, sizeof(float) * k_tok * ml * H);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(float) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(float) * k_tok * ml * H);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(make_buffer_b(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
    }
    pool_count_ = (size_t)config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    gate_up_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AMX_MOE_BASE_F32() = default;

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<float> input(qlen * config_.hidden_size);
    std::vector<float> output(qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

  void load_weights() { derived()->load_weights(); }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) continue;

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);
    }

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) fn(i);
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    // Copy input (f32 → f32, no conversion)
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (float*)input + i * config_.hidden_size, sizeof(float) * config_.hidden_size);
      }
    });

    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    apply_activation(activated_expert, nth, qlen);

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Weighted sum: f32 input, f32 output — no bf16 conversion
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              float* src = m_local_down_output_ptr_[expert_ids[i * k + j]] +
                           m_local_pos_[i][j] * config_.hidden_size + e;
              x0 = _mm512_fmadd_ps(_mm512_loadu_ps(src), weight, x0);
              x1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 16), weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) continue;
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);
    }

    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      gate_up_ba_[expert_idx]->from_mat(qlen, (float*)input, 0, 1);
    }

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    apply_activation(activated_expert, nth, qlen);

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Weighted sum: f32 input, f32 output
    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[j])) continue;
        __m512 weight = _mm512_set1_ps(weights[j]);
        float* src = m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e;
        x0 = _mm512_fmadd_ps(_mm512_loadu_ps(src), weight, x0);
        x1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 16), weight, x1);
      }
      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  void derived_init() {}

  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  void apply_activation(int activated_expert, int nth, int qlen) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        float* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        float* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0 = _mm512_loadu_ps(gate_output_ptr + j);
          __m512 gate_val1 = _mm512_loadu_ps(gate_output_ptr + j + 16);
          __m512 up_val0   = _mm512_loadu_ps(up_output_ptr + j);
          __m512 up_val1   = _mm512_loadu_ps(up_output_ptr + j + 16);
          __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit);
          __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit);
          _mm512_storeu_ps(gate_output_ptr + j,      result0);
          _mm512_storeu_ps(gate_output_ptr + j + 16, result1);
        }
      }
    };

    if (activated_expert == 0) return;
    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) fn(task_id);
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// AMX_FP4_MOE_TP_F32 — F32-native MXFP4 MoE CRTP class
// ============================================================================
template <class T = amx::GemmKernel224MXFP4SmallKGroupF32>
class AMX_FP4_MOE_TP_F32 : public AMX_MOE_BASE_F32<T, AMX_FP4_MOE_TP_F32<T>> {
  using Base = AMX_MOE_BASE_F32<T, AMX_FP4_MOE_TP_F32<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP4_MOE_TP_F32() = default;
  AMX_FP4_MOE_TP_F32(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("MXFP4 F32 MoE only supports KGroup FP4");
    }
    printf("Creating AMX_FP4_MOE_TP_F32 %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_FP4_MOE_TP_F32() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup_f32(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup_f32(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup_f32(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                              down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup_f32(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                              down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("MXFP4 F32 MoE only supports KGroup FP4.");
    if (config_.gate_scale == nullptr) throw std::runtime_error("MXFP4 F32 MoE only supports load native weight.");

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;
          convert_or_copy(gate_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(up_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.up_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(down_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.down_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_BASE_F32 (merge_results — f32 native)
// ============================================================================
template <class T, class Derived>
class TP_MOE<AMX_MOE_BASE_F32<T, Derived>> : public TP_MOE_Common<AMX_MOE_BASE_F32<T, Derived>> {
 public:
  using TP_MOE_Common<AMX_MOE_BASE_F32<T, Derived>>::TP_MOE_Common;

  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0 = _mm512_loadu_ps((float*)output + token_nth * config.hidden_size + e);
          __m512 x1 = _mm512_loadu_ps((float*)output + token_nth * config.hidden_size + e + 16);
          _mm512_storeu_ps(merge_to + e,      _mm512_add_ps(_mm512_loadu_ps(merge_to + e),      x0));
          _mm512_storeu_ps(merge_to + e + 16, _mm512_add_ps(_mm512_loadu_ps(merge_to + e + 16), x1));
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          _mm512_storeu_ps(merge_to + e,
                           _mm512_add_ps(_mm512_loadu_ps(merge_to + e), _mm512_loadu_ps(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        _mm512_storeu_ps((float*)output + token_nth * config.hidden_size + e,      _mm512_loadu_ps(merge_to + e));
        _mm512_storeu_ps((float*)output + token_nth * config.hidden_size + e + 16, _mm512_loadu_ps(merge_to + e + 16));
      }
    };

    auto pool = config.pool;
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) fn(i);
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };
    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

// ============================================================================
// TP_MOE specialization for AMX_FP4_MOE_TP_F32 (load_weights)
// ============================================================================
template <typename K>
class TP_MOE<AMX_FP4_MOE_TP_F32<K>> : public TP_MOE<AMX_MOE_BASE_F32<K, AMX_FP4_MOE_TP_F32<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE_F32<K, AMX_FP4_MOE_TP_F32<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 F32 MoE only supports Packed FP4 with KGroup Scale");

    printf("From %s\n", use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "Packed FP4 with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);
              memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);
                memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (tpc.intermediate_size >> 1));
                  memcpy((ggml_bf16_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (ggml_bf16_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done (F32).\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (ggml_bf16_t*)(tpc.gate_scale);
      delete[] (ggml_bf16_t*)(tpc.up_scale);
      delete[] (ggml_bf16_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP4_MOE_F32_H
