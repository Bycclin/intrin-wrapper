/* wrapper/xmmintrin.h – SSE (single-precision) */
#ifndef UNIVERSAL_XMMINTRIN_H
#define UNIVERSAL_XMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <math.h>    // for fminf/fmaxf
#include <stdlib.h>  // for posix_memalign, free

// — x86-style SIMD types —
typedef float32x4_t __m128;
typedef int32x4_t   __m128i;
#define _MM_SHUFFLE(z,y,x,w) (((z)<<6)|((y)<<4)|((x)<<2)|(w))

// — Engine’s simd.h aliases —
typedef __m128  VecF;
typedef __m128i VecI;
#define AsVecF(x) (*(VecF*)&(x))
#define AsVecI(x) (*(VecI*)&(x))

// — Aligned malloc/free for util.h —
static inline void* _mm_malloc(size_t size, size_t align) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) ptr = NULL;
    return ptr;
}
static inline void _mm_free(void* p) {
    free(p);
}

// — Load / Store / Set —
static inline __m128 _mm_load_ss(const float *p)           { return vsetq_lane_f32(*p, vdupq_n_f32(0.0f), 0); }
static inline __m128 _mm_load_ps(const float *p)           { return vld1q_f32(p); }
static inline __m128 _mm_loadu_ps(const float *p)          { return vld1q_f32(p); }
static inline void   _mm_store_ss(float *p, __m128 a)      { vst1q_lane_f32(p, a, 0); }
static inline void   _mm_store_ps(float *p, __m128 a)      { vst1q_f32(p, a); }
static inline void   _mm_storeu_ps(float *p, __m128 a)     { vst1q_f32(p, a); }

static inline __m128 _mm_set_ss(float w)                   { return vsetq_lane_f32(w, vdupq_n_f32(0.0f), 0); }
static inline __m128 _mm_set_ps(float e3,float e2,float e1,float e0) {
    float aligned[4] __attribute__((aligned(16))) = { e0, e1, e2, e3 };
    return vld1q_f32(aligned);
}
static inline __m128 _mm_set1_ps(float w)                  { return vdupq_n_f32(w); }
static inline __m128 _mm_set_ps1(float w)                  { return _mm_set1_ps(w); }
static inline __m128 _mm_setzero_ps(void)                  { return vdupq_n_f32(0.0f); }

// — Move —
static inline __m128 _mm_move_ss(__m128 a, __m128 b) {
    return vsetq_lane_f32(vgetq_lane_f32(b,0), a, 0);
}

// — Arithmetic —
static inline __m128 _mm_add_ss(__m128 a, __m128 b) {
    return vsetq_lane_f32(vgetq_lane_f32(a,0) + vgetq_lane_f32(b,0), a, 0);
}
static inline __m128 _mm_add_ps(__m128 a, __m128 b)        { return vaddq_f32(a, b); }
static inline __m128 _mm_sub_ss(__m128 a, __m128 b) {
    return vsetq_lane_f32(vgetq_lane_f32(a,0) - vgetq_lane_f32(b,0), a, 0);
}
static inline __m128 _mm_sub_ps(__m128 a, __m128 b)        { return vsubq_f32(a, b); }
static inline __m128 _mm_mul_ss(__m128 a, __m128 b) {
    return vsetq_lane_f32(vgetq_lane_f32(a,0) * vgetq_lane_f32(b,0), a, 0);
}
static inline __m128 _mm_mul_ps(__m128 a, __m128 b)        { return vmulq_f32(a, b); }
static inline __m128 _mm_div_ss(__m128 a, __m128 b) {
    return vsetq_lane_f32(vgetq_lane_f32(a,0) / vgetq_lane_f32(b,0), a, 0);
}
static inline __m128 _mm_div_ps(__m128 a, __m128 b)        { return vdivq_f32(a, b); }
static inline __m128 _mm_sqrt_ss(__m128 a) {
    float tmp[2]; vst1_f32(tmp, vget_low_f32(a)); tmp[0] = sqrtf(tmp[0]);
    return vsetq_lane_f32(tmp[0], a, 0);
}
static inline __m128 _mm_sqrt_ps(__m128 a)                 { return vsqrtq_f32(a); }
static inline __m128 _mm_rcp_ss(__m128 a) {
    float32x2_t lo = vget_low_f32(a), r = vrecpe_f32(lo);
    return vsetq_lane_f32(vget_lane_f32(r,0), a, 0);
}
static inline __m128 _mm_rcp_ps(__m128 a)                  { return vrecpeq_f32(a); }
static inline __m128 _mm_rsqrt_ss(__m128 a) {
    float32x2_t lo = vget_low_f32(a), r = vrsqrte_f32(lo);
    return vsetq_lane_f32(vget_lane_f32(r,0), a, 0);
}
static inline __m128 _mm_rsqrt_ps(__m128 a)                { return vrsqrteq_f32(a); }

// — Min/Max —
static inline __m128 _mm_min_ss(__m128 a, __m128 b) {
    float fa = vgetq_lane_f32(a,0), fb = vgetq_lane_f32(b,0);
    return vsetq_lane_f32(fminf(fa,fb), a, 0);
}
static inline __m128 _mm_min_ps(__m128 a, __m128 b)        { return vminq_f32(a, b); }
static inline __m128 _mm_max_ss(__m128 a, __m128 b) {
    float fa = vgetq_lane_f32(a,0), fb = vgetq_lane_f32(b,0);
    return vsetq_lane_f32(fmaxf(fa,fb), a, 0);
}
static inline __m128 _mm_max_ps(__m128 a, __m128 b)        { return vmaxq_f32(a, b); }

// — Logical —
static inline __m128 _mm_and_ps(__m128 a, __m128 b) {
    return vreinterpretq_f32_s32(
        vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b))
    );
}
static inline __m128 _mm_andnot_ps(__m128 a, __m128 b) {
    return vreinterpretq_f32_s32(
        vbicq_s32(vreinterpretq_s32_f32(b), vreinterpretq_s32_f32(a))
    );
}
static inline __m128 _mm_or_ps(__m128 a, __m128 b) {
    return vreinterpretq_f32_s32(
        vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b))
    );
}
static inline __m128 _mm_xor_ps(__m128 a, __m128 b) {
    return vreinterpretq_f32_s32(
        veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b))
    );
}

// — Comparisons (scalar) —
static inline __m128 _mm_cmpeq_ss(__m128 a, __m128 b) {
    uint32_t c = (vgetq_lane_f32(a,0)==vgetq_lane_f32(b,0))?0xFFFFFFFFU:0U;
    uint32x4_t tmp = vsetq_lane_u32(c, vreinterpretq_u32_f32(a), 0);
    return vreinterpretq_f32_u32(tmp);
}
static inline __m128 _mm_cmplt_ss(__m128 a, __m128 b) {
    uint32_t c = (vgetq_lane_f32(a,0)< vgetq_lane_f32(b,0))?0xFFFFFFFFU:0U;
    uint32x4_t tmp = vsetq_lane_u32(c, vreinterpretq_u32_f32(a), 0);
    return vreinterpretq_f32_u32(tmp);
}
static inline __m128 _mm_cmple_ss(__m128 a, __m128 b) {
    uint32_t c = (vgetq_lane_f32(a,0)<=vgetq_lane_f32(b,0))?0xFFFFFFFFU:0U;
    uint32x4_t tmp = vsetq_lane_u32(c, vreinterpretq_u32_f32(a), 0);
    return vreinterpretq_f32_u32(tmp);
}
static inline __m128 _mm_cmpgt_ss(__m128 a, __m128 b) {
    uint32_t c = (vgetq_lane_f32(a,0)> vgetq_lane_f32(b,0))?0xFFFFFFFFU:0U;
    uint32x4_t tmp = vsetq_lane_u32(c, vreinterpretq_u32_f32(a), 0);
    return vreinterpretq_f32_u32(tmp);
}
static inline __m128 _mm_cmpge_ss(__m128 a, __m128 b) {
    uint32_t c = (vgetq_lane_f32(a,0)>=vgetq_lane_f32(b,0))?0xFFFFFFFFU:0U;
    uint32x4_t tmp = vsetq_lane_u32(c, vreinterpretq_u32_f32(a), 0);
    return vreinterpretq_f32_u32(tmp);
}

// — Comparisons (vector) —
static inline __m128 _mm_cmpeq_ps(__m128 a, __m128 b)       { return vreinterpretq_f32_u32(vceqq_f32(a,b)); }
static inline __m128 _mm_cmplt_ps(__m128 a, __m128 b)       { return vreinterpretq_f32_u32(vcltq_f32(a,b)); }
static inline __m128 _mm_cmple_ps(__m128 a, __m128 b)       { return vreinterpretq_f32_u32(vcleq_f32(a,b)); }
static inline __m128 _mm_cmpgt_ps(__m128 a, __m128 b)       { return vreinterpretq_f32_u32(vcgtq_f32(a,b)); }
static inline __m128 _mm_cmpge_ps(__m128 a, __m128 b)       { return vreinterpretq_f32_u32(vcgeq_f32(a,b)); }

// — Conversions —
static inline int    _mm_cvtss_si32(__m128 a)               { return (int)vgetq_lane_f32(a,0); }
static inline int    _mm_cvttss_si32(__m128 a)              { return (int)vgetq_lane_f32(a,0); }
static inline __m128 _mm_cvtsi32_ss(__m128 a, int b)        { return vsetq_lane_f32((float)b, a, 0); }

// — Shuffle / Unpack —
static inline __m128 _mm_shuffle_ps(__m128 a, __m128 b, int imm8) {
    float ta[4], tb[4], r[4];
    vst1q_f32(ta,a); vst1q_f32(tb,b);
    r[0]=ta[(imm8>>0)&3];  r[1]=ta[(imm8>>2)&3];
    r[2]=tb[(imm8>>4)&3];  r[3]=tb[(imm8>>6)&3];
    return vld1q_f32(r);
}
static inline __m128 _mm_unpackhi_ps(__m128 a, __m128 b) {
    float32x2_t a_hi = vget_high_f32(a), b_hi = vget_high_f32(b);
    return vcombine_f32(vzip1_f32(a_hi,b_hi), vzip2_f32(a_hi,b_hi));
}
static inline __m128 _mm_unpacklo_ps(__m128 a, __m128 b) {
    float32x2_t a_lo = vget_low_f32(a),  b_lo = vget_low_f32(b);
    return vcombine_f32(vzip1_f32(a_lo,b_lo), vzip2_f32(a_lo,b_lo));
}

// — Misc: movemask —
static inline int _mm_movemask_ps(__m128 a) {
    uint32x4_t s = vshrq_n_u32(vreinterpretq_u32_f32(a),31);
    uint32_t p[4]; vst1q_u32(p,s);
    return (p[0]&1)|((p[1]&1)<<1)|((p[2]&1)<<2)|((p[3]&1)<<3);
}

#if defined(__ARM_FEATURE_DOTPROD)
// map SSE4.1 dot-prod (_mm_dpbusd_epi32) → ARM vdotq_s32
static inline __m128i dpbusdEpi32(__m128i sum, __m128i x, __m128i y) {
    return vdotq_s32(sum,
                     vreinterpretq_s8_s32(x),
                     vreinterpretq_s8_s32(y));
}
#endif

#else  // x86
  #include <xmmintrin.h>
#endif

#endif // UNIVERSAL_XMMINTRIN_H
