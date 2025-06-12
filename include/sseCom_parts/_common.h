#pragma once

#include <emmintrin.h> // SSE2
#include <stdint.h>
#include <limits.h>

#define SSECOM_INLINE static inline

// Compilers that support the GNU C extention are suppose to defined `__GNUC__`
// However, some platforms (like Visual Studio) undefined them even when using a supported compiler
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    #define SSECOM_GNUC_EXTENTION
    #define SSECOM_FORCE_ORDER __asm__("")
#else
    #define SSECOM_FORCE_ORDER  // Nothing
#endif


// Get a integer vector with all bits set (very cheap!)
SSECOM_INLINE __m128i _setone_i128() {
    return _mm_cmpeq_epi16(_mm_setzero_si128(),_mm_setzero_si128());
}

// Uses a mask to set the output to `a` if true, or `b` if false (Works for all element sizes)
SSECOM_INLINE __m128i _either_i128(__m128i a, __m128i b, __m128i mask) {
    __m128i aToKeep = _mm_and_si128(a, mask);
    __m128i bToKeep = _mm_andnot_si128(mask, b);
    return _mm_or_si128(aToKeep, bToKeep);
}

// Uses a mask to set the output to `a` if true, or `b` if false
SSECOM_INLINE __m128 _either_f32x4(__m128 a, __m128 b, __m128 mask) {
    __m128 aToKeep = _mm_and_ps(a, mask);
    __m128 bToKeep = _mm_andnot_ps(mask, b);
    return _mm_or_ps(aToKeep, bToKeep);
}

// Uses a mask to set the output to `a` if true, or `b` if false
SSECOM_INLINE __m128d _either_f64x2(__m128d a, __m128d b, __m128d mask) {
    __m128d aToKeep = _mm_and_pd(a, mask);
    __m128d bToKeep = _mm_andnot_pd(mask, b);
    return _mm_or_pd(aToKeep, bToKeep);
}


// Fills all 8-bit elements with its MSB
SSECOM_INLINE __m128i _fillWithMSB_i8x16(__m128i input) {
    return _mm_cmplt_epi8(input, _mm_setzero_si128());
}

// Fills all bits in both 64-bit elements with its MSB
SSECOM_INLINE __m128i _fillWithMSB_i64x2(__m128i input) {
    return _mm_srai_epi32( _mm_shuffle_epi32(input, _MM_SHUFFLE(3, 3, 1, 1)), 31 );
}
