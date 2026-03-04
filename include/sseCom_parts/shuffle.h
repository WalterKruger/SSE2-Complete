#pragma once

#include <emmintrin.h>  // SSE2
#include <stdint.h>

#include "_common.h"

// Compiler otherwise calculate a bunch of partial results then combine them, which is 25% slower...
#ifdef SSECOM_GNUC_EXTENTION
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index)\
    __asm__(".att_syntax;   pinsrw  %[idx],%[mem],%[out]" : [out] "+x" (vecOut) : [mem] "m" (memInp), [idx] "i" (index))
#else
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index) vecOut = _mm_insert_epi16(vecOut, memInp, index)
#endif



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
    __m128i loDupe = _mm_unpacklo_epi8(toShuff, toShuff);
    __m128i hiDupe = _mm_unpackhi_epi8(toShuff, toShuff);

    __m128i loloDupe = _mm_unpacklo_epi16(loDupe, loDupe);
    __m128i lohiDupe = _mm_unpackhi_epi16(loDupe, loDupe);
    __m128i hiloDupe = _mm_unpacklo_epi16(hiDupe, hiDupe);
    __m128i hihiDupe = _mm_unpackhi_epi16(hiDupe, hiDupe);

    __m128i A = _mm_shuffle_epi32(loloDupe, 0);
    __m128i E = _mm_shuffle_epi32(lohiDupe, 0);
    __m128i I = _mm_shuffle_epi32(hiloDupe, 0);
    __m128i M = _mm_shuffle_epi32(hihiDupe, 0);

    // ZERO, Low, Even, Hi
    __m128i loloParts = _mm_xor_si128(loloDupe, _mm_shuffle_epi32(loloDupe, _MM_SHUFFLE(2,0,0,0)));
    __m128i lohiParts = _mm_xor_si128(lohiDupe, _mm_shuffle_epi32(lohiDupe, _MM_SHUFFLE(2,0,0,0)));
    __m128i hiloParts = _mm_xor_si128(hiloDupe, _mm_shuffle_epi32(hiloDupe, _MM_SHUFFLE(2,0,0,0)));
    __m128i hihiParts = _mm_xor_si128(hihiDupe, _mm_shuffle_epi32(hihiDupe, _MM_SHUFFLE(2,0,0,0)));

    __m128i LOW_LOLO =  _mm_shuffle_epi32(loloParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_LOLO = _mm_shuffle_epi32(loloParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_LOLO = _mm_shuffle_epi32(loloParts, _MM_SHUFFLE(3,3,3,3));

    __m128i LOW_LOHI =  _mm_shuffle_epi32(lohiParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_LOHI = _mm_shuffle_epi32(lohiParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_LOHI = _mm_shuffle_epi32(lohiParts, _MM_SHUFFLE(3,3,3,3));

    __m128i LOW_HILO =    _mm_shuffle_epi32(hiloParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_HILO =   _mm_shuffle_epi32(hiloParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_HILO =   _mm_shuffle_epi32(hiloParts, _MM_SHUFFLE(3,3,3,3));

    __m128i LOW_HIHI =    _mm_shuffle_epi32(hihiParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_HIHI =   _mm_shuffle_epi32(hihiParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_HIHI =   _mm_shuffle_epi32(hihiParts, _MM_SHUFFLE(3,3,3,3));

    __m128i FULL_LOLO = _mm_xor_si128(LOW_LOLO, HIGH_LOLO);
    __m128i FULL_LOHI = _mm_xor_si128(LOW_LOHI, HIGH_LOHI);
    __m128i FULL_HILO = _mm_xor_si128(LOW_HILO, HIGH_HILO);
    __m128i FULL_HIHI = _mm_xor_si128(LOW_HIHI, HIGH_HIHI);


    __m128i isOdd64 = _fillWithMSB_i8x16(_mm_slli_epi32(indexes, 8-4));
    __m128i isOdd32 = _fillWithMSB_i8x16(_mm_slli_epi32(indexes, 8-3));
    __m128i isOdd16 = _fillWithMSB_i8x16(_mm_slli_epi32(indexes, 8-2));
    __m128i isOdd8 =  _fillWithMSB_i8x16(_mm_slli_epi32(indexes, 8-1));

    
    // abcd_efgh_ijkl_mnop
    __m128i sumOfHalf_lolo = _mm_xor_si128(LOW_LOLO, _mm_and_si128(FULL_LOLO, isOdd16));
    __m128i valIfEven_lolo = _mm_xor_si128(A, _mm_and_si128(EVEN_LOLO, isOdd16));

    __m128i sumOfHalf_lohi = _mm_xor_si128(LOW_LOHI, _mm_and_si128(FULL_LOHI, isOdd16));
    __m128i valIfEven_lohi = _mm_xor_si128(E, _mm_and_si128(EVEN_LOHI, isOdd16));

    __m128i sumOfHalf_hilo = _mm_xor_si128(LOW_HILO, _mm_and_si128(FULL_HILO, isOdd16));
    __m128i valIfEven_hilo = _mm_xor_si128(I, _mm_and_si128(EVEN_HILO, isOdd16));

    __m128i sumOfHalf_hihi = _mm_xor_si128(LOW_HIHI, _mm_and_si128(FULL_HIHI, isOdd16));
    __m128i valIfEven_hihi = _mm_xor_si128(M, _mm_and_si128(EVEN_HIHI, isOdd16));


    __m128i valIfLoLo = _mm_xor_si128(valIfEven_lolo, _mm_and_si128(sumOfHalf_lolo, isOdd8));
    __m128i valIfLoHi = _mm_xor_si128(valIfEven_lohi, _mm_and_si128(sumOfHalf_lohi, isOdd8));
    __m128i valIfHiLo = _mm_xor_si128(valIfEven_hilo, _mm_and_si128(sumOfHalf_hilo, isOdd8));
    __m128i valIfHiHi = _mm_xor_si128(valIfEven_hihi, _mm_and_si128(sumOfHalf_hihi, isOdd8));


    __m128i LOW =  _mm_xor_si128(valIfLoLo, valIfLoHi);
    __m128i HIGH = _mm_xor_si128(valIfHiLo, valIfHiHi);
    __m128i FULL = _mm_xor_si128(LOW, HIGH);
    __m128i EVEN = _mm_xor_si128(valIfLoLo, valIfHiLo);

    __m128i sumOfHalf = _mm_xor_si128(LOW, _mm_and_si128(FULL, isOdd64));
    __m128i valIfEven = _mm_xor_si128(valIfLoLo, _mm_and_si128(EVEN, isOdd64));

    return _mm_xor_si128(valIfEven, _mm_and_si128(sumOfHalf, isOdd32));
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
    // After getting the xor between the upper/lower, we can narrow it down
    // further as the xor identity either returns the even or
    // removes the odd part (as `even ^ BOTH = odd`)
    __m128i aaac = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(2,0,0,0));

    __m128i non_LO_EVEN_HI = _mm_xor_si128(toShuff, aaac);

    __m128i LO =    _mm_shuffle_epi32(non_LO_EVEN_HI, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN =  _mm_shuffle_epi32(non_LO_EVEN_HI, _MM_SHUFFLE(2,2,2,2));
    __m128i HI =    _mm_shuffle_epi32(non_LO_EVEN_HI, _MM_SHUFFLE(3,3,3,3));

    __m128i FULL = _mm_xor_si128(LO, HI);
    __m128i a = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(0,0,0,0));


    __m128i isInHi = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-2), 31);
    __m128i isInOdd = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-1), 31);

    // a ^ ((a^b) & isOdd) = (isOdd)? a^a^b : a
    __m128i part_half = _mm_xor_si128(LO, _mm_and_si128(FULL, isInHi));
    __m128i selIfEven =  _mm_xor_si128(a, _mm_and_si128(EVEN, isInHi));

    return _mm_xor_si128(selIfEven, _mm_and_si128(part_half, isInOdd));
}

// Shuffle 64-bit integers using the corresponding index
__m128i _shuffleVar_i64x2(__m128i toShuff, __m128i indexes) {
    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsHiMask = _mm_srai_epi32(_mm_slli_epi32(dupLo32, 31), 31);

    __m128i loDupe = _shuffle_i64x2(toShuff, _MM_SHUFFLE2(0, 0));
    __m128i hiDupe = _shuffle_i64x2(toShuff, _MM_SHUFFLE2(1, 1));

    return _either_i128(hiDupe, loDupe, selectsHiMask);
}
