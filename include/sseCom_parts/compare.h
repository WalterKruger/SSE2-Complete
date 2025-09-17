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

// Compare signed 64-bit elements for less-than or equal-to and return a mask of all 1s if true
__m128i _cmpLssEq_i64x2(__m128i a, __m128i b) {
    return _mm_cmpeq_epi32(_cmpGrt_i64x2(a, b), _mm_setzero_si128());
}

// Compare signed 64-bit elements for greater-than or equal-to and return a mask of all 1s if true
__m128i _cmpGrtEq_i64x2(__m128i a, __m128i b) {
    return _mm_cmpeq_epi32(_cmpLss_i64x2(a, b), _mm_setzero_si128());
}

// Compare 64-bit integer elements for equality and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpEq_i64x2(__m128i a, __m128i b) {
    __m128i isEql32 = _mm_cmpeq_epi32(a, b);
    
    // [..., eqHi & eqLo, eqLo & eqHi]
    __m128i isEql32_rev = _mm_shuffle_epi32(isEql32, _MM_SHUFFLE(2,3,0,1));
    return _mm_and_si128(isEql32, isEql32_rev);
}


#if 0
__m128i _cmpLss_u64x2(__m128i a, __m128i b) {
    __m128i isLess_32 = _cmpLss_u32x4(a, b);
    __m128i isEql_32 = _mm_cmpeq_epi32(a, b);

    __m128i loIsLess_inHi = _mm_slli_epi64(isLess_32, 32);

    // If the upper 32 is equal, then the lower determines the results
    __m128i result_hi = _mm_or_si128(
       isLess_32, _mm_and_si128(isEql_32, loIsLess_inHi)
    );

    return _mm_shuffle_epi32(result_hi, _MM_SHUFFLE(3,3,1,1));
}

__m128i _cmpGrt_u64x2(__m128i a, __m128i b) {
    return _cmpLss_u64x2(b, a);
}
#endif


// Compare unsigned 64-bit elements for less-than and return a mask of all 1s if true
__m128i _cmpLss_u64x2(__m128i a, __m128i b) {
    __m128i difference = _mm_sub_epi64(a, b);

    /*      Underflow based on MSB
        |___A___|___B___|__A-B__:__A<B__|
        |   0   |   0   |   0   :   0   |
        |   0   |   0   |   1   :   1   | (!A & D)
        |   0   |   1   |   0   :   1   | (B & !(!D & A))
        |   0   |   1   |   1   :   1   | (B & !(!D & A)) + (!A & D)
        |   1   |   0   |   0   :   0   |
        |   1   |   0   |   1   :   0   | 
        |   1   |   1   |   0   :   0   |
        |   1   |   1   |   1   :   1   | (B & !(!D & A))
    */
    // The cases where we know that the difference's MSB wasn't left over from `a`
    __m128i diffMSBNotFromA = _mm_andnot_si128(a, difference);
    // When MSB(b): a>b iff the `a-b` removed the MSB of `a` and a borrow didn't "restore" it
    __m128i bMSBExcludeDiffShowsAGrt = _mm_andnot_si128(_mm_andnot_si128(difference, a), b);

    return _fillWithMSB_i64x2(_mm_or_si128(diffMSBNotFromA, bMSBExcludeDiffShowsAGrt));
}

// Compare unsigned 64-bit elements for greater-than and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrt_u64x2(__m128i isGrt, __m128i isLss) {
    return _cmpLss_u64x2(isLss, isGrt);
}

// Cmpeq size doesn't matter

// Compare unsigned 64-bit elements for less-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpLssEq_u64x2(__m128i a, __m128i b) {
    return _mm_cmpeq_epi32(_cmpGrt_u64x2(a, b), _mm_setzero_si128());
}

// Compare unsigned 64-bit elements for greater-than or equal-to and return a mask of all 1s if true
SSECOM_INLINE __m128i _cmpGrtEq_u64x2(__m128i a, __m128i b) {
    return _mm_cmpeq_epi32(_cmpLss_u64x2(a, b), _mm_setzero_si128());
}





// Return the larger element between the corresponding signed 32-bit integers
__m128i _max_i32x4(__m128i a, __m128i b) {
    __m128i isGrtMask = _mm_cmpgt_epi32(a, b);
    return _either_i128(a, b, isGrtMask);
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
