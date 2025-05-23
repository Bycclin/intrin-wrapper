/* wrapper/pmmintrin.h – SSE3 (horizontal adds/subs, addsub, dups) */
#ifndef UNIVERSAL_PMMINTRIN_H
#define UNIVERSAL_PMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include "emmintrin.h"

// Horizontal add: {a0+a1,a2+a3, b0+b1,b2+b3}
static inline __m128 _mm_hadd_ps(__m128 a,__m128 b){
    float ta[4], tb[4], r[4];
    vst1q_f32(ta,a); vst1q_f32(tb,b);
    r[0]=ta[0]+ta[1]; r[1]=ta[2]+ta[3];
    r[2]=tb[0]+tb[1]; r[3]=tb[2]+tb[3];
    return vld1q_f32(r);
}
// Horizontal sub: {a0−a1,a2−a3, b0−b1,b2−b3}
static inline __m128 _mm_hsub_ps(__m128 a,__m128 b){
    float ta[4], tb[4], r[4];
    vst1q_f32(ta,a); vst1q_f32(tb,b);
    r[0]=ta[0]-ta[1]; r[1]=ta[2]-ta[3];
    r[2]=tb[0]-tb[1]; r[3]=tb[2]-tb[3];
    return vld1q_f32(r);
}

// addsub: {a0−b0,a1+b1,a2−b2,a3+b3}
static inline __m128 _mm_addsub_ps(__m128 a,__m128 b){
    const uint32x4_t m = {0x80000000,0,0x80000000,0};
    __m128 bn = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(b),m));
    return vaddq_f32(a,bn);
}

// move duplicates
static inline __m128 _mm_moveldup_ps(__m128 a){
    float32x4x2_t t = vuzpq_f32(a,a);
    return t.val[0];
}
static inline __m128 _mm_movehdup_ps(__m128 a){
    float32x4x2_t t = vuzpq_f32(a,a);
    return t.val[1];
}

// lddqu alias
static inline __m128i _mm_lddqu_si128(const __m128i *p){
    return _mm_loadu_si128(p);
}

#else  // x86
  #include <pmmintrin.h>
#endif

#endif // UNIVERSAL_PMMINTRIN_H
