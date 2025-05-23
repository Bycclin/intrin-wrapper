/* wrapper/emmintrin.h – SSE2 (double & integer) */
#ifndef UNIVERSAL_EMMINTRIN_H
#define UNIVERSAL_EMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include "xmmintrin.h"
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

// --- Double‐precision ---
typedef float64x2_t __m128d;

// Load/Store/Set
static inline __m128d _mm_load_pd(const double *p)    { return vld1q_f64(p); }
static inline __m128d _mm_loadu_pd(const double *p)   { return vld1q_f64(p); }
static inline void     _mm_store_pd(double *p,__m128d a){ vst1q_f64(p,a); }
static inline void     _mm_storeu_pd(double *p,__m128d a){ vst1q_f64(p,a); }
static inline __m128d _mm_set_pd(double e1,double e0){
    double d[2]={e0,e1}; return vld1q_f64(d);
}
static inline __m128d _mm_set1_pd(double v)          { return vdupq_n_f64(v); }
static inline __m128d _mm_setzero_pd(void)           { return vdupq_n_f64(0.0); }

// Arithmetic
static inline __m128d _mm_add_pd(__m128d a,__m128d b){ return vaddq_f64(a,b); }
static inline __m128d _mm_sub_pd(__m128d a,__m128d b){ return vsubq_f64(a,b); }
static inline __m128d _mm_mul_pd(__m128d a,__m128d b){ return vmulq_f64(a,b); }
static inline __m128d _mm_div_pd(__m128d a,__m128d b){ return vdivq_f64(a,b); }
static inline __m128d _mm_sqrt_pd(__m128d a)         { return vsqrtq_f64(a); }

// Min/Max/Compare
static inline __m128d _mm_min_pd(__m128d a,__m128d b){ return vminq_f64(a,b); }
static inline __m128d _mm_max_pd(__m128d a,__m128d b){ return vmaxq_f64(a,b); }
static inline __m128d _mm_cmpeq_pd(__m128d a,__m128d b){
    return vreinterpretq_f64_u64(vceqq_f64(a,b));
}

// --- Integer vectors ---
typedef int32x4_t __m128i;

// Load/Store/Set
static inline __m128i _mm_load_si128(const __m128i *p){
    return vreinterpretq_s32_s8(vld1q_s8((const int8_t*)p));
}
static inline __m128i _mm_loadu_si128(const __m128i *p){
    return vreinterpretq_s32_s8(vld1q_s8((const int8_t*)p));
}
static inline void _mm_store_si128(__m128i *p,__m128i v){
    vst1q_s8((int8_t*)p, vreinterpretq_s8_s32(v));
}
static inline void _mm_storeu_si128(__m128i *p,__m128i v){
    vst1q_s8((int8_t*)p, vreinterpretq_s8_s32(v));
}
static inline __m128i _mm_setzero_si128(void){
    return vreinterpretq_s32_u32(vdupq_n_u32(0));
}
static inline __m128i _mm_set_epi64x(long long e1,long long e0){
    int64_t d[2]={(int64_t)e0,(int64_t)e1};
    return vreinterpretq_s32_s64(vld1q_s64(d));
}
static inline __m128i _mm_set_epi32(int e3,int e2,int e1,int e0){
    int32_t d[4]={e0,e1,e2,e3};
    return vld1q_s32(d);
}

// Arithmetic
static inline __m128i _mm_add_epi32(__m128i a,__m128i b){ return vaddq_s32(a,b); }
static inline __m128i _mm_sub_epi32(__m128i a,__m128i b){ return vsubq_s32(a,b); }
static inline __m128i _mm_mullo_epi16(__m128i a,__m128i b){
    return vreinterpretq_s32_s16(vmulq_s16(vreinterpretq_s16_s32(a),
                                           vreinterpretq_s16_s32(b)));
}
static inline __m128i _mm_mullo_epi32(__m128i a,__m128i b){
    return vmulq_s32(a,b);
}

// Variable‐width shifts (manual fallback)
static inline __m128i _mm_slli_epi32(__m128i a,int imm){
    int32_t t[4]; vst1q_s32(t,a);
    for(int i=0;i<4;i++) t[i] <<= imm;
    return vld1q_s32(t);
}
static inline __m128i _mm_srli_epi32(__m128i a,int imm){
    uint32_t t[4]; vst1q_u32(t, vreinterpretq_u32_s32(a));
    for(int i=0;i<4;i++) t[i] >>= imm;
    return vreinterpretq_s32_u32(vld1q_u32(t));
}
static inline __m128i _mm_srai_epi32(__m128i a,int imm){
    int32_t t[4]; vst1q_s32(t,a);
    for(int i=0;i<4;i++) t[i] >>= imm;
    return vld1q_s32(t);
}

// Logical
static inline __m128i _mm_and_si128(__m128i a,__m128i b){ return vandq_s32(a,b); }

// Compare
static inline __m128i _mm_cmpeq_epi32(__m128i a,__m128i b){
    return vreinterpretq_s32_u32(vceqq_s32(a,b));
}

#else  // x86
  #include <emmintrin.h>
#endif

#endif // UNIVERSAL_EMMINTRIN_H
