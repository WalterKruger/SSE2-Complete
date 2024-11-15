#pragma once

#include "_common.h"
#include <float.h>


// ====== 8-bit ======

// Negate signed 8-bit integers
SSECOM_INLINE __m128i _negate_i8x16(__m128i toNegate) {
    return _mm_sub_epi8(_mm_setzero_si128(), toNegate);
}

// Conditionaly negate signed 8-bit integers. The condition input must be either all 1s or 0s
SSECOM_INLINE __m128i _condNegate_i8x16(__m128i i8toNegate, __m128i condiationMask) {
    return _mm_sub_epi8(_mm_xor_si128(condiationMask, i8toNegate), condiationMask);
}


// ====== 16-bit ======

// Negate signed 16-bit integers
SSECOM_INLINE __m128i _negate_i16x8(__m128i toNegate) {
    return _mm_sub_epi16(_mm_setzero_si128(), toNegate);
}

// Conditionaly negate signed 16-bit integers. The condition input must be either all 1s or 0s
SSECOM_INLINE __m128i _condNegate_i16x8(__m128i i16toNegate, __m128i condiationMask) {
    return _mm_sub_epi16(_mm_xor_si128(condiationMask, i16toNegate), condiationMask);
}


// ====== 32-bit ======

// Negate signed 32-bit integers
SSECOM_INLINE __m128i _negate_i32x4(__m128i a) {
    return _mm_sub_epi32(_mm_setzero_si128(), a);
}

// Conditionaly negate signed 32-bit integers. The condition input must be either all 1s or 0s
SSECOM_INLINE __m128i _condNegate_i32x4(__m128i i32toNegate, __m128i condiationMask) {
    return _mm_sub_epi32(_mm_xor_si128(condiationMask, i32toNegate), condiationMask);
}


// ====== 64-bit ======

// Negate signed 64-bit integers
SSECOM_INLINE __m128i _negate_i64x2(__m128i a) {
    return _mm_sub_epi64(_mm_setzero_si128(), a);
}

// Conditionaly negate signed 64-bit integers. The condition input must be either all 1s or 0s
SSECOM_INLINE __m128i _condNegate_i64x2(__m128i i64toNegate, __m128i condiationMask) {
    return _mm_sub_epi64(_mm_xor_si128(condiationMask, i64toNegate), condiationMask);
}


// ====== Float types ======

// Negate 32-bit floats
SSECOM_INLINE __m128 _negate_f32x4(__m128 a) {
    return _mm_xor_ps(a, _mm_castsi128_ps( _mm_set1_epi32(1ULL << 31) ));
}

// Negate 64-bit floats
SSECOM_INLINE __m128d _negate_f64x2(__m128d a) {
    return _mm_xor_pd(a, _mm_castsi128_pd( _mm_set1_epi64x(1ULL << 63) ));
}