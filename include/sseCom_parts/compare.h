#pragma once

#include "_common.h"
#include "negation.h"

// ======================
//        Compare
// ======================

// ====== Unsigned 8-bit ======

// Compare unsigned 8-bit elements for greater-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrtEq_u8x16(__m128i isGrtEq, __m128i isLessEq) {
    return _mm_cmpeq_epi8(_mm_max_epu8(isGrtEq, isLessEq), isGrtEq);
}

// Compare unsigned 8-bit elements for less-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLssEq_u8x16(__m128i isLessEq, __m128i isGrtEq) {
    return _cmpGrtEq_u8x16(isGrtEq, isLessEq);
}

// Compare unsigned 8-bit elements for greater than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrt_u8x16(__m128i isGrt, __m128i isLess) {
    return _mm_cmpeq_epi8(_cmpLssEq_u8x16(isGrt, isLess), _mm_setzero_si128());
}

// Compare unsigned 8-bit elements for less than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLss_u8x16(__m128i isLess, __m128i isGrt) {
    return _mm_cmpeq_epi8(_cmpGrtEq_u8x16(isLess, isGrt), _mm_setzero_si128());
}


// ====== Unsigned 16-bit ======

// Compare unsigned 16-bit elements for greater-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrtEq_u16x8(__m128i isGrtEq, __m128i isLessEq) {
    return _mm_cmpeq_epi16(_mm_subs_epu16(isLessEq, isGrtEq), _mm_setzero_si128());
}

// Compare unsigned 16-bit elements for less-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLssEq_u16x8(__m128i isLessEq, __m128i isGrtEq) {
    return _cmpGrtEq_u16x8(isGrtEq, isLessEq);
}

// Compare unsigned 16-bit elements for greater-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrt_u16x8(__m128i isGrt, __m128i isLess) {
    return _mm_cmpeq_epi16(_cmpLssEq_u16x8(isGrt, isLess), _mm_setzero_si128());
}

// Compare unsigned 16-bit elements for less-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLss_u16x8(__m128i isLess, __m128i isGrt) {
    return _mm_cmpeq_epi16(_cmpGrtEq_u16x8(isLess, isGrt), _mm_setzero_si128());
}


// ====== Unsigned 32-bit ======

// Compare unsigned 32-bit elements for greater-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrt_u32x4(__m128i isGrt, __m128i isLess) {
    __m128i isGrt_s = _mm_xor_si128(isGrt, _mm_set1_epi32(INT32_MIN));
    __m128i isLess_s = _mm_xor_si128(isLess, _mm_set1_epi32(INT32_MIN));
    
    return _mm_cmpgt_epi32(isGrt_s, isLess_s);
}

// Compare unsigned 32-bit elements for less-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLss_u32x4(__m128i isLess, __m128i isGrt) {
    return _cmpGrt_u32x4(isGrt, isLess);
}

// Compare unsigned 32-bit elements for greater-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrtEq_u32x4(__m128i isGrtEq, __m128i isLessEq) {
    return _mm_cmpeq_epi32(_cmpLss_u32x4(isGrtEq, isLessEq), _mm_setzero_si128());
}

// Compare unsigned 32-bit elements for less-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLssEq_u32x4(__m128i isLessEq, __m128i isGrtEq) {
    return _mm_cmpeq_epi32(_cmpGrt_u32x4(isLessEq, isGrtEq), _mm_setzero_si128());
}

// ====== 64-bit ======

// Compare signed 64-bit elements for less-than and return a mask of all 1s if true
__m128i _cmpLss_i64x2(__m128i a, __m128i b) {
    __m128i signsDiffer_MSB = _mm_xor_si128(a, b);

    // Result must be true (a < b)
    __m128i aIsNeg_bIsPos_MSB = _mm_andnot_si128(b, a);

    // A difference is only negative if the minuend is less than the subtrahend
    __m128i difference = _mm_sub_epi64(a, b);

    // ...except when overflow (can only happen when signs differ)
    __m128i result_MSB = _mm_or_si128(_mm_andnot_si128(signsDiffer_MSB, difference), aIsNeg_bIsPos_MSB);
    
    return _fillWithMSB_i64x2(result_MSB);
}

// Compare signed 64-bit elements for greater-than and return a mask of all 1s if true
__m128i _cmpGrt_i64x2(__m128i a, __m128i b) {
    return _cmpLss_i64x2(b, a);
}

// Compare signed 64-bit elements for greater-than or equal-to and return a mask of all 1s if true
__m128i _cmpGrtEq_i64x2(__m128i a, __m128i b) {
    __m128i signsDiffer_MSB = _mm_xor_si128(a, b);
    __m128i aIsNeg_bIsPos_MSB = _mm_andnot_si128(b, a);

    __m128i difference = _mm_sub_epi64(a, b);
    __m128i result_MSB = _mm_or_si128(_mm_andnot_si128(signsDiffer_MSB, difference), aIsNeg_bIsPos_MSB);

    return _mm_cmpgt_epi32(_mm_shuffle_epi32(result_MSB, _MM_SHUFFLE(3,3,1,1)), _setone_i128());
}

// Compare signed 64-bit elements for less-than or equal-to and return a mask of all 1s if true
__m128i _cmpLssEq_i64x2(__m128i a, __m128i b) {
    return _cmpGrtEq_i64x2(b, a);
}

// Compare 64-bit integer elements for equality and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpEq_i64x2(__m128i a, __m128i b) {
    __m128i isEql32 = _mm_cmpeq_epi32(a, b);
    
    // [..., eqHi & eqLo, eqLo & eqHi]
    __m128i isEql32_rev = _mm_shuffle_epi32(isEql32, _MM_SHUFFLE(2,3,0,1));
    return _mm_and_si128(isEql32, isEql32_rev);
}


// Compare unsigned 64-bit elements for greater-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrt_u64x2(__m128i isGrt, __m128i isLss) {
    __m128i halfDiff = _mm_xor_si128(isGrt, isLss);
    __m128i halfBorrow = _mm_and_si128(halfDiff, isGrt); // isGrt & ~isLss
    __m128i subRight1 = _mm_sub_epi64(_mm_srli_epi64(halfDiff, 1), halfBorrow); 

    return _fillWithMSB_i64x2(subRight1);
}

// Compare unsigned 64-bit elements for less-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLss_u64x2(__m128i isLss, __m128i isGrt) {
    return _cmpGrt_u64x2(isGrt, isLss);
}

// Cmpeq size doesn't matter

// Compare unsigned 64-bit elements for less-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLssEq_u64x2(__m128i isLssEq, __m128i isGrtEq) {
    __m128i halfDiff = _mm_xor_si128(isLssEq, isGrtEq);
    __m128i halfBorrow = _mm_and_si128(halfDiff, isLssEq); // isLssEq & ~isGrtEq
    __m128i subRight1 = _mm_sub_epi64(_mm_srli_epi64(halfDiff, 1), halfBorrow);

    return _mm_cmpgt_epi32(_mm_shuffle_epi32(subRight1, _MM_SHUFFLE(3,3,1,1)), _setone_i128());
}

// Compare unsigned 64-bit elements for greater-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrtEq_u64x2(__m128i isGrtEq, __m128i isLssEq) {
    return _cmpLssEq_u64x2(isLssEq, isGrtEq);
}





// Return the larger element between the corresponding signed 32-bit integers
__m128i _max_i32x4(__m128i a, __m128i b) {
    __m128i isGrtMask = _mm_cmpgt_epi32(a, b);
    __m128i xorSum = _mm_xor_si128(a, b);

    // b ^ (a ^ b) = a
    return _mm_xor_si128(b, _mm_and_si128(xorSum, isGrtMask));
}

// Return the larger element between the corresponding signed 8-bit integers
SSECOM_INLINE __m128i _max_i8x16(__m128i a, __m128i b) {
    #if 0
        __m128i isGrtMask = _mm_cmpgt_epi8(a, b);
        return _either_i128(a, b, isGrtMask);
    #else
        __m128i aUnsigned = _mm_xor_si128(a, _mm_set1_epi8(CHAR_MIN));
        __m128i bUnsigned = _mm_xor_si128(b, _mm_set1_epi8(CHAR_MIN));

        __m128i unsignedMax = _mm_max_epu8(aUnsigned, bUnsigned);
        return _mm_xor_si128(unsignedMax, _mm_set1_epi8(CHAR_MIN));
    #endif
}
