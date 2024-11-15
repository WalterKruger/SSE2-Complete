#pragma once

#include "_common.h"


// ======================
//      Movemask
// ======================

// Gets the most significant bit of each 8-bit integer element as a scalar bit array
SSECOM_INLINE int _getMsb_i8x16(__m128i a) {
    return _mm_movemask_epi8(a);
}

// Gets the most significant bit of each 16-bit integer element as a scalar bit array
SSECOM_INLINE int _getMsb_i16x8(__m128i a) {
    // When a value is saturated during the conversion, it will have the same sign/MSB 
    // Negative: INT8_MIN (0b10000000)   Positive: INT8_MAX (0b01111111)
    __m128i duplicatedTrunc = _mm_packs_epi16(a, a);

    // Uses less registers than truncating with a zero vector
    return _mm_movemask_epi8(duplicatedTrunc) >> 8;
}

// Gets the most significant bit of each 32-bit integer element as a scalar bit array
SSECOM_INLINE int _getMsb_i32x4(__m128i a) {
    return _mm_movemask_ps( _mm_castsi128_ps(a) );
}

// Gets the most significant bit of each 64-bit integer element as a scalar bit array
SSECOM_INLINE int _getMsb_i64x2(__m128i a) {
    return _mm_movemask_pd( _mm_castsi128_pd(a) );
}
