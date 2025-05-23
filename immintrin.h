/* wrapper/immintrin.h – AVX & beyond umbrella */
#ifndef UNIVERSAL_IMMINTRIN_H
#define UNIVERSAL_IMMINTRIN_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)

// On ARM, pull in the SSE–SSE4.2 wrappers
#include "xmmintrin.h"
#include "emmintrin.h"
#include "pmmintrin.h"
#include "tmmintrin.h"
#include "smmintrin.h"
#include "nmmintrin.h"

// Simulate 256-bit vectors as two 128-bit lanes
typedef struct { __m128 lo, hi; } __m256;

// Load / store 8 floats
static inline __m256 _mm256_load_ps(const float *p) {
    __m256 r;
    r.lo = _mm_loadu_ps(p);
    r.hi = _mm_loadu_ps(p + 4);
    return r;
}
static inline void _mm256_store_ps(float *p, __m256 a) {
    _mm_storeu_ps(p,     a.lo);
    _mm_storeu_ps(p + 4, a.hi);
}

// Arithmetic
static inline __m256 _mm256_add_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = _mm_add_ps(a.lo, b.lo);
    r.hi = _mm_add_ps(a.hi, b.hi);
    return r;
}
static inline __m256 _mm256_mul_ps(__m256 a, __m256 b) {
    __m256 r;
    r.lo = _mm_mul_ps(a.lo, b.lo);
    r.hi = _mm_mul_ps(a.hi, b.hi);
    return r;
}

// You can add other _mm256_… fallbacks here as needed

#else  // x86

#include <immintrin.h>

#endif // ARCH

#endif // UNIVERSAL_IMMINTRIN_H
