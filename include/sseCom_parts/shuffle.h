#pragma once

#include <emmintrin.h>  // SSE2
#include <stdint.h>


// Compiler otherwise calculate a bunch of partial results then combine them, which is 25% slower...
#ifdef SSECOM_GNUC_EXTENTION
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index)\
    __asm__(".att_syntax;   pinsrw  %[idx],%[mem],%[out]" : [out] "+x" (vecOut) : [mem] "m" (memInp), [idx] "i" (index))
#else
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index) vecOut = _mm_insert_epi16(vecOut, memInp, index)
#endif


#define SSECOM_DUPE_N4(i) SSECOM_DUPE_Ni(i+0) SSECOM_DUPE_Ni(i+1) SSECOM_DUPE_Ni(i+2) SSECOM_DUPE_Ni(i+3)
#define SSECOM_DUPE_N16 SSECOM_DUPE_N4(0) SSECOM_DUPE_N4(4) SSECOM_DUPE_N4(8) SSECOM_DUPE_N4(12)




// ====== Shuffle by immediate ======

// Create an immediate for use with the `_shuffleLoHi` function
#define _MM_SHUFHALF(index1, index0) (((index1) << 2) | (index0))

// Set the lower-half of the result to 2 32-bit elements from loPart, and the high to 2 elements from hiPart
#define _shuffleLoHi_i32x4(loPart, loIndexes, hiPart, hiIndexes) (_mm_castps_si128(\
    _mm_shuffle_ps(_mm_castsi128_ps(loPart), _mm_castsi128_ps(hiPart), ((hiIndexes) << 4) | (loIndexes))\
))

// Shuffle 64-bit integers using an immediate control mask
// Output: { input[imm8.bit[0]], input[imm8.bit[1]] }
#define _shuffle_i64x2(toShuffle, mask) ( _mm_shuffle_epi32(toShuffle,\
    ((mask & 1)? _MM_SHUFHALF(3,2) : _MM_SHUFHALF(1,0)) | ((mask & 2)? _MM_SHUFHALF(3,2) << 4 : _MM_SHUFHALF(1,0) << 4)\
) )


// ====== Variable shuffles ======

// Shuffle 8-bit integers using the corresponding index
__m128i _shuffleVar_i8x16(__m128i toShuff, __m128i indexes) {
    const __m128i INDEX_MASK = _mm_set1_epi8(0x0f);

    indexes = _mm_and_si128(indexes, INDEX_MASK);

    static uint8_t shuffledVals[16], toShuff_s[16];
    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);

    __m128i nextIndexInLo = indexes;
    uint32_t curIndex32 = 0;

    // Need macro as `bsrli` expects an immediate and to encorage unrolling
    // The force order macro is to prevent a GCC/clang mis-optimization (~30% faster)
    #define SSECOM_DUPE_Ni(i)\
        curIndex32 = _mm_cvtsi128_si32(nextIndexInLo);\
        nextIndexInLo = _mm_bsrli_si128(indexes, ((i)<15)? (i)+1 : 0);\
        shuffledVals[i] = toShuff_s[(uint8_t)curIndex32];\
        SSECOM_FORCE_ORDER;
    
    SSECOM_DUPE_N16
    #undef CASE_Ni

    return _mm_loadu_si128((__m128i*)shuffledVals);
}

#undef SSECOM_DUPE_N4
#undef SSECOM_DUPE_N16


// Shuffle 16-bit integers using the corresponding index
__m128i _shuffleVar_i16x8(__m128i toShuff, __m128i indexes) {

    __m128i result = _mm_setzero_si128(); // Prevent UD
    indexes = _mm_and_si128(indexes, _mm_set1_epi16(0b111));

    uint16_t toShuff_s[8];
    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);

    // Two instructions per element (insert can use memory operand)
    SSECOM_FORCE_INSRET16(result, toShuff_s[_mm_extract_epi16(indexes,0)], 0);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,1)], 1);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,2)], 2);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,3)], 3);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,4)], 4);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,5)], 5);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,6)], 6);
    result = _mm_insert_epi16(result, toShuff_s[_mm_extract_epi16(indexes,7)], 7);

    return result;
}
#undef SSECOM_FORCE_INSRET16

// Shuffle 32-bit integers using the corresponding index
__m128i _shuffleVar_i32x4(__m128i toShuff, __m128i indexes) {
    indexes = _mm_and_si128(indexes, _mm_set1_epi32(0b11));

    uint32_t toShuffArr[4];
    _mm_storeu_si128((__m128i*)toShuffArr, toShuff);

    uint32_t index_0 = _mm_cvtsi128_si32(indexes);
    uint32_t index_1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(indexes, _MM_SHUFFLE(1,1,1,1)));
    uint32_t index_2 = _mm_cvtsi128_si32(_mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2,2,2)));
    uint32_t index_3 = _mm_cvtsi128_si32(_mm_shuffle_epi32(indexes, _MM_SHUFFLE(3,3,3,3)));

    __m128i res_0 = _mm_loadu_si32(toShuffArr + index_0);
    __m128i res_1 = _mm_loadu_si32(toShuffArr + index_1);
    __m128i resLower = _mm_unpacklo_epi32(res_0, res_1);

    __m128i res_2 = _mm_loadu_si32(toShuffArr + index_2);
    __m128i res_3 = _mm_loadu_si32(toShuffArr + index_3);
    __m128i resUpper = _mm_unpacklo_epi32(res_2, res_3);
    
    return _mm_unpacklo_epi64(resLower, resUpper);
}

// Shuffle 64-bit integers using the corresponding index
__m128i _shuffleVar_i64x2(__m128i toShuff, __m128i indexes) {
    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsHiMask = _mm_srai_epi32(_mm_slli_epi32(dupLo32, 31), 31);

    __m128i loDupe = _shuffle_i64x2(toShuff, _MM_SHUFFLE2(0, 0));
    __m128i hiDupe = _shuffle_i64x2(toShuff, _MM_SHUFFLE2(1, 1));

    return _either_i128(hiDupe, loDupe, selectsHiMask);
}
