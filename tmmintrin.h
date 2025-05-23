/* wrapper/tmmintrin.h – SSSE3 (abs, shuffle bytes, alignr) */
#ifndef UNIVERSAL_TMMINTRIN_H
#define UNIVERSAL_TMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include "pmmintrin.h"

// Absolute value
static inline __m128i _mm_abs_epi8(__m128i a)   { return vreinterpretq_s32_s8(vqabsq_s8(vreinterpretq_s8_s32(a))); }
static inline __m128i _mm_abs_epi16(__m128i a)  { return vreinterpretq_s32_s16(vqabsq_s16(vreinterpretq_s16_s32(a))); }
static inline __m128i _mm_abs_epi32(__m128i a)  { return vqabsq_s32(a); }

// Byte‐shuffle (pshufb)
static inline __m128i _mm_shuffle_epi8(__m128i a,__m128i m){
    uint8x16_t tb = vreinterpretq_u8_s32(a);
    uint8x16_t mb = vreinterpretq_u8_s32(m);
    uint8x16_t idx= vandq_u8(mb, vdupq_n_u8(0x0F));
    int8x16_t  sh = vqtbl1q_s8(vreinterpretq_s8_u8(tb), idx);
    uint8x16_t msb = vcgtq_u8(mb, vdupq_n_u8(127));
    uint8x16_t res = vbslq_u8(msb, vdupq_n_u8(0), vreinterpretq_u8_s8(sh));
    return vreinterpretq_s32_u8(res);
}

// Align-right (palignr) fallback
static inline __m128i _mm_alignr_epi8(__m128i a,__m128i b,int c){
    uint8_t tmp[32];
    vst1q_u8(tmp,      vreinterpretq_u8_s32(b));
    vst1q_u8(tmp + 16, vreinterpretq_u8_s32(a));
    uint8_t res[16];
    int start = c < 32 ? c : 32;
    for(int i=0;i<16;i++) res[i] = tmp[start + i];
    return vreinterpretq_s32_u8(vld1q_u8(res));
}

#else  // x86
  #include <tmmintrin.h>
#endif

#endif // UNIVERSAL_TMMINTRIN_H
