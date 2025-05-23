/* wrapper/nmmintrin.h – SSE4.2 (popcnt, CRC32, string‐compare stub) */
#ifndef UNIVERSAL_NMMINTRIN_H
#define UNIVERSAL_NMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

#include "smmintrin.h"

// popcnt
#if defined(__GNUC__)||defined(__clang__)
static inline int   _mm_popcnt_u32(unsigned int v){ return __builtin_popcount(v); }
#if defined(__aarch64__)
static inline long long _mm_popcnt_u64(unsigned long long v){ return __builtin_popcountll(v); }
#endif
#else
static inline int _mm_popcnt_u32(unsigned int n){
    uint8x8_t b=vcnt_u8(vdup_n_u8((uint8_t)n));
    return vaddv_u8(b);
}
#endif

// CRC32
#if (defined(__GNUC__)||defined(__clang__))&&(defined(__ARM_FEATURE_CRC32)||defined(__aarch64__))
  #include <arm_acle.h>
  static inline unsigned int       _mm_crc32_u8 (unsigned int c,unsigned char v){ return __crc32b(c,v); }
  static inline unsigned int       _mm_crc32_u16(unsigned int c,unsigned short v){ return __crc32h(c,v); }
  static inline unsigned int       _mm_crc32_u32(unsigned int c,unsigned int v){ return __crc32w(c,v); }
  #if defined(__aarch64__)
  static inline unsigned long long _mm_crc32_u64(unsigned long long c,unsigned long long v){ return __crc32d(c,v); }
  #endif
#else
#warning "NEON CRC32 not supported; using stub"
static inline unsigned int _mm_crc32_u8 (unsigned int c,unsigned char v){ return c ^ v; }
static inline unsigned int _mm_crc32_u16(unsigned int c,unsigned short v){ return c ^ v; }
static inline unsigned int _mm_crc32_u32(unsigned int c,unsigned int v){ return c ^ v; }
#if defined(__aarch64__)
static inline unsigned long long _mm_crc32_u64(unsigned long long c,unsigned long long v){ return c ^ v; }
#endif
#endif

// string‐compare stub
static inline int _mm_cmpestri(__m128i a,int la,__m128i b,int lb,int imm8){
    (void)a; (void)la; (void)b; (void)lb; (void)imm8;
    return 0;
}

#else  // x86
  #include <nmmintrin.h>
#endif

#endif // UNIVERSAL_NMMINTRIN_H
