/* wrapper/smmintrin.h – SSE4.1 (blend, dp, extract/insert, round, pack) */
#ifndef UNIVERSAL_SMMINTRIN_H
#define UNIVERSAL_SMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include "tmmintrin.h"

// blendv
static inline __m128  _mm_blendv_ps(__m128 a,__m128 b,__m128 m){
    return vreinterpretq_f32_u32(
      vbslq_u32(vcgtq_s32(vreinterpretq_s32_f32(m), vdupq_n_s32(-1)),
                vreinterpretq_u32_f32(b),
                vreinterpretq_u32_f32(a))
    );
}
static inline __m128i _mm_blendv_epi8(__m128i a,__m128i b,__m128i m){
    return vreinterpretq_s32_u8(
      vbslq_u8(vcgtq_s8(vreinterpretq_s8_s32(m), vdupq_n_s8(-1)),
               vreinterpretq_u8_s32(b),
               vreinterpretq_u8_s32(a))
    );
}

// dot‐product
static inline __m128 _mm_dp_ps(__m128 a,__m128 b,int imm8){
    __m128 t=vmulq_f32(a,b);
    if(!(imm8&0x10)) t=vsetq_lane_f32(0.0f,t,0);
    if(!(imm8&0x20)) t=vsetq_lane_f32(0.0f,t,1);
    if(!(imm8&0x40)) t=vsetq_lane_f32(0.0f,t,2);
    if(!(imm8&0x80)) t=vsetq_lane_f32(0.0f,t,3);
    float32x2_t s=vpadd_f32(vget_low_f32(t),vget_high_f32(t));
    float tot=vget_lane_f32(s,0)+vget_lane_f32(s,1);
    __m128 res=_mm_setzero_ps();
    if(imm8&0x01) res=vsetq_lane_f32(tot,res,0);
    if(imm8&0x02) res=vsetq_lane_f32(tot,res,1);
    if(imm8&0x04) res=vsetq_lane_f32(tot,res,2);
    if(imm8&0x08) res=vsetq_lane_f32(tot,res,3);
    return res;
}

// extract/insert
static inline int   _mm_extract_epi32(__m128i a,int ndx){
    int32_t v[4]; vst1q_s32(v,a);
    return v[ndx & 3];
}
static inline int   _mm_extract_ps(__m128 a,int ndx){
    float f[4]; vst1q_f32(f,a);
    union{float f; int i;}u; u.f = f[ndx & 3];
    return u.i;
}
static inline __m128i _mm_insert_epi32(__m128i a,int b,int ndx){
    int32_t v[4]; vst1q_s32(v,a);
    v[ndx & 3] = b;
    return vld1q_s32(v);
}
static inline __m128  _mm_insert_ps(__m128 a,__m128 b,int imm8){
    float fa[4], fb[4];
    vst1q_f32(fa,a); vst1q_f32(fb,b);
    fa[imm8 & 3] = fb[(imm8>>4)&3];
    return vld1q_f32(fa);
}

// packus epi32→16
static inline __m128i _mm_packus_epi32(__m128i a,__m128i b){
    uint16x4_t pa=vqmovun_s32(a), pb=vqmovun_s32(b);
    return vreinterpretq_s32_u16(vcombine_u16(pa,pb));
}

// rounding
#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_TO_NEG_INF     0x01
#define _MM_FROUND_TO_POS_INF     0x02
#define _MM_FROUND_TO_ZERO        0x03
#define _MM_FROUND_CUR_DIRECTION  0x04
#define _MM_FROUND_RAISE_EXC      0x00
#define _MM_FROUND_NO_EXC         0x08

static inline __m128 _mm_round_ps(__m128 a,int mode){
    if((mode & 0x3) == _MM_FROUND_TO_ZERO){
        int32_t v[4]; vst1q_s32(v, vcvtq_s32_f32(a));
        return vcvtq_f32_s32(vld1q_s32(v));
    }
    return a;
}

#else  // x86
  #include <smmintrin.h>
#endif

#endif // UNIVERSAL_SMMINTRIN_H
