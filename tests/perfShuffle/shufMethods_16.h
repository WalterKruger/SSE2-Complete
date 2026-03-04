#pragma once
#include "../_perfCommon.h"


NOINLINE __m128i switch_u16(__m128i toShuff, __m128i indexes) {
    
    uint16_t indexes_s[8], result[8];
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t n=0; n < 8; n++) {
        switch(indexes_s[n] & 0b111) {
            case 0: result[n] = _mm_extract_epi16(toShuff, 0); break;
            case 1: result[n] = _mm_extract_epi16(toShuff, 1); break;
            case 2: result[n] = _mm_extract_epi16(toShuff, 2); break;
            case 3: result[n] = _mm_extract_epi16(toShuff, 3); break;
            case 4: result[n] = _mm_extract_epi16(toShuff, 4); break;
            case 5: result[n] = _mm_extract_epi16(toShuff, 5); break;
            case 6: result[n] = _mm_extract_epi16(toShuff, 6); break;
            case 7: result[n] = _mm_extract_epi16(toShuff, 7); break;
        }
    }
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i memLoop_u16(__m128i toShuff, __m128i indexes) {
    uint16_t toShuff_s[8], indexes_s[8], result[8];

    indexes = _mm_and_si128(indexes, _mm_set1_epi16(0b111));

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 8; i++)
        result[i] = toShuff_s[indexes_s[i]];

    return _mm_loadu_si128((__m128i*)result);
}

#ifdef GNUC_EXTENTION
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index)\
    __asm__(".att_syntax;   pinsrw  %[idx],%[mem],%[out]" : [out] "+x" (vecOut) : [mem] "m" (memInp), [idx] "i" (index))
#else
    #define SSECOM_FORCE_INSRET16(vecOut, memInp, index) vecOut = _mm_insert_epi16(vecOut, memInp, index)
#endif

NOINLINE __m128i insertExtractMem_u16(__m128i toShuff, __m128i indexes) {

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

// Forces an extract that can't be reordered
#ifdef GNUC_EXTENTION
    #define ORD_EXTRACT16(x, idx) ({__asm__("");_mm_extract_epi16(x, idx);})
#else
    // FIXME: MSVC reorders...
    #define ORD_EXTRACT16(x, idx) _mm_extract_epi16(x, idx)
#endif

NOINLINE __m128i clang_u16(__m128i toShuff, __m128i indexes) {
    uint16_t toShuff_s[8];
    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);

    indexes = _mm_and_si128(indexes, _mm_set1_epi16(0b111));

    // Two instructions per element (insert can use memory operand)
    uint16_t el0 = ORD_EXTRACT16(indexes, 0);
    uint16_t el1 = ORD_EXTRACT16(indexes, 1);
    uint16_t el2 = ORD_EXTRACT16(indexes, 2);
    uint16_t el3 = ORD_EXTRACT16(indexes, 3);
    uint16_t el4 = ORD_EXTRACT16(indexes, 4);
    uint16_t el5 = ORD_EXTRACT16(indexes, 5);
    uint16_t el6 = ORD_EXTRACT16(indexes, 6);
    uint16_t el7 = ORD_EXTRACT16(indexes, 7);

    __m128i res7 = _mm_cvtsi32_si128(toShuff_s[el7]);
    __m128i res6 = _mm_cvtsi32_si128(toShuff_s[el6]);

    __m128i resHiHi = _mm_unpacklo_epi16(res6, res7);

    __m128i res5 = _mm_cvtsi32_si128(toShuff_s[el5]);
    __m128i res4 = _mm_cvtsi32_si128(toShuff_s[el4]);

    __m128i resHiLo = _mm_unpacklo_epi16(res4, res5);
    __m128i resHi = _mm_unpacklo_epi32(resHiLo, resHiHi);

    __m128i res3 = _mm_cvtsi32_si128(toShuff_s[el3]);
    __m128i res2 = _mm_cvtsi32_si128(toShuff_s[el2]);

    __m128i resLoHi = _mm_unpacklo_epi16(res2, res3);

    __m128i res1 = _mm_cvtsi32_si128(toShuff_s[el1]);
    __m128i res0 = _mm_cvtsi32_si128(toShuff_s[el0]);

    __m128i resLoLo = _mm_unpacklo_epi16(res0, res1);
    __m128i resLo = _mm_unpacklo_epi32(resLoLo, resLoHi);

    return _mm_unpacklo_epi64(resLo, resHi);    
}

NOINLINE __m128i viaXor_u16(__m128i toShuff, __m128i indexes) {
    #if 0
    __m128i a = _mm_shuffle_epi32(_mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(0,0,0,0)), _MM_SHUFFLE(0,0,0,0));
    __m128i b = _mm_shuffle_epi32(_mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(1,1,1,1)), _MM_SHUFFLE(0,0,0,0));
    __m128i c = _mm_shuffle_epi32(_mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(2,2,2,2)), _MM_SHUFFLE(0,0,0,0));
    __m128i d = _mm_shuffle_epi32(_mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(3,3,3,3)), _MM_SHUFFLE(0,0,0,0));

    __m128i E = _mm_shuffle_epi32(_mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(0,0,0,0)), _MM_SHUFFLE(2,2,2,2));
    __m128i F = _mm_shuffle_epi32(_mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(1,1,1,1)), _MM_SHUFFLE(2,2,2,2));
    __m128i G = _mm_shuffle_epi32(_mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(2,2,2,2)), _MM_SHUFFLE(2,2,2,2));
    __m128i H = _mm_shuffle_epi32(_mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(3,3,3,3)), _MM_SHUFFLE(2,2,2,2));

    #else

    __m128i aa_bb = _mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(1,1,0,0));
    __m128i cc_dd = _mm_shufflelo_epi16(toShuff, _MM_SHUFFLE(3,3,2,2));
    __m128i a = _mm_shuffle_epi32(aa_bb, _MM_SHUFFLE(0,0,0,0));
    __m128i b = _mm_shuffle_epi32(aa_bb, _MM_SHUFFLE(1,1,1,1));
    __m128i c = _mm_shuffle_epi32(cc_dd, _MM_SHUFFLE(0,0,0,0));
    __m128i d = _mm_shuffle_epi32(cc_dd, _MM_SHUFFLE(1,1,1,1));

    __m128i ee_ff = _mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(1,1,0,0));
    __m128i gg_hh = _mm_shufflehi_epi16(toShuff, _MM_SHUFFLE(3,3,2,2));
    __m128i E = _mm_shuffle_epi32(ee_ff, _MM_SHUFFLE(2,2,2,2));
    __m128i F = _mm_shuffle_epi32(ee_ff, _MM_SHUFFLE(3,3,3,3));
    __m128i G = _mm_shuffle_epi32(gg_hh, _MM_SHUFFLE(2,2,2,2));
    __m128i H = _mm_shuffle_epi32(gg_hh, _MM_SHUFFLE(3,3,3,3));

    #endif

    __m128i aXORe = _mm_xor_si128(a, E);

    __m128i even16Lo = _mm_xor_si128(a, b);
    __m128i even16 = _mm_xor_si128(even16Lo, _mm_xor_si128(E, F));

    __m128i even8Lo = _mm_xor_si128(a, c);
    __m128i even8 = _mm_xor_si128(even8Lo, _mm_xor_si128(E, G));

    __m128i LOW = _mm_xor_si128(even16Lo, _mm_xor_si128(c, d));
    __m128i FULL = _mm_xor_si128(LOW, _mm_xor_si128(_mm_xor_si128(E, F), _mm_xor_si128(G, H)));
    /*
    __m128i loDupe = _mm_unpacklo_epi16(toShuff, toShuff);
    __m128i hiDupe = _mm_unpackhi_epi16(toShuff, toShuff);

    __m128i aE_bF_cG_dH = _mm_xor_si128(loDupe, hiDupe);

    __m128i aXORe = _mm_shuffle_epi32(aE_bF_cG_dH, _MM_SHUFFLE(0,0,0,0));

    __m128i non_e16_e8_aEdH = _mm_xor_si128(aE_bF_cG_dH, aXORe);
    // FULL: aEdH ^ b^c^d ^ F^G

    __m128i even8 =     _mm_shuffle_epi32(non_e16_e8_aEdH, _MM_SHUFFLE(2,2,2,2));
    __m128i even16 =    _mm_shuffle_epi32(non_e16_e8_aEdH, _MM_SHUFFLE(1,1,1,1));
    */



    __m128i isOdd64 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-3), 15);
    
    __m128i cur =           _selectXorBoth_i128(LOW, FULL, isOdd64);
    __m128i toSelHalfLo =   _selectXorBoth_i128(even16Lo, even16, isOdd64);

    __m128i toSelHalfEven =     _selectXorBoth_i128(even8Lo, even8, isOdd64);
    __m128i toSelHalfEvenLo =   _selectXorBoth_i128(a, aXORe, isOdd64);


    __m128i isOdd32 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-2), 15);
    cur =           _selectXorBoth_i128(toSelHalfLo, cur, isOdd32);
    toSelHalfLo =   _selectXorBoth_i128(toSelHalfEvenLo, toSelHalfEven, isOdd32);


    __m128i isOdd16 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-1), 15);
    return _selectXorBoth_i128(toSelHalfLo, cur, isOdd16);
}

NOINLINE __m128i viaXorA_u16(__m128i toShuff, __m128i indexes) {
    __m128i loDupe = _mm_unpacklo_epi16(toShuff, toShuff);
    __m128i hiDupe = _mm_unpackhi_epi16(toShuff, toShuff);

    __m128i A = _mm_shuffle_epi32(loDupe, 0);
    __m128i E = _mm_shuffle_epi32(hiDupe, 0);

    // ZERO, Low, Even, Hi
    __m128i loParts = _mm_xor_si128(loDupe, _mm_shuffle_epi32(loDupe, _MM_SHUFFLE(2,0,0,0)));
    __m128i hiParts = _mm_xor_si128(hiDupe, _mm_shuffle_epi32(hiDupe, _MM_SHUFFLE(2,0,0,0)));

    __m128i LOW_LO =    _mm_shuffle_epi32(loParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_LO =   _mm_shuffle_epi32(loParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_LO =   _mm_shuffle_epi32(loParts, _MM_SHUFFLE(3,3,3,3));

    __m128i LOW_HI =    _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_HI =   _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(2,2,2,2));
    __m128i HIGH_HI =   _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(3,3,3,3));

    __m128i FULL_LO = _mm_xor_si128(LOW_LO, HIGH_LO);
    __m128i FULL_HI = _mm_xor_si128(LOW_HI, HIGH_HI);


    __m128i isOdd64 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-3), 15);
    __m128i isOdd32 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-2), 15);
    __m128i isOdd16 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-1), 15);

    // cond? a : b = b ^ (aXORb & cond)
    __m128i sumOfHalf_lo = _selectXorBoth_i128(LOW_LO, FULL_LO, isOdd32);
    __m128i valIfEven_lo = _selectXorBoth_i128(A, EVEN_LO, isOdd32);

    __m128i sumOfHalf_hi = _selectXorBoth_i128(LOW_HI, FULL_HI, isOdd32);
    __m128i valIfEven_hi = _selectXorBoth_i128(E, EVEN_HI, isOdd32);

    __m128i valIfLo = _selectXorBoth_i128(valIfEven_lo, sumOfHalf_lo, isOdd16);

    __m128i valIfHi = _selectXorBoth_i128(valIfEven_hi, sumOfHalf_hi, isOdd16);

    return _either_i128(valIfHi, valIfLo, isOdd64);
}

NOINLINE __m128i viaXorB_u16(__m128i toShuff, __m128i indexes) {
    __m128i loDupe = _mm_unpacklo_epi16(toShuff, toShuff);
    __m128i hiDupe = _mm_unpackhi_epi16(toShuff, toShuff);

    __m128i A = _mm_shuffle_epi32(loDupe, _MM_SHUFFLE(0,0,0,0));
    __m128i B = _mm_shuffle_epi32(loDupe, _MM_SHUFFLE(1,1,1,1));
    __m128i C = _mm_shuffle_epi32(loDupe, _MM_SHUFFLE(2,2,2,2));
    __m128i D = _mm_shuffle_epi32(loDupe, _MM_SHUFFLE(3,3,3,3));

    __m128i E = _mm_shuffle_epi32(hiDupe, _MM_SHUFFLE(0,0,0,0));
    __m128i F = _mm_shuffle_epi32(hiDupe, _MM_SHUFFLE(1,1,1,1));
    __m128i G = _mm_shuffle_epi32(hiDupe, _MM_SHUFFLE(2,2,2,2));
    __m128i H = _mm_shuffle_epi32(hiDupe, _MM_SHUFFLE(3,3,3,3));

    __m128i LOW_LO =    _mm_xor_si128(A, B);
    __m128i EVEN_LO =   _mm_xor_si128(A, C);
    __m128i FULL_LO =   _mm_xor_si128(LOW_LO, _mm_xor_si128(C, D));

    __m128i LOW_HI =    _mm_xor_si128(E, F);
    __m128i EVEN_HI =   _mm_xor_si128(E, G);
    __m128i FULL_HI =   _mm_xor_si128(LOW_HI, _mm_xor_si128(G, H));



    __m128i isOdd64 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-3), 15);
    __m128i isOdd32 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-2), 15);
    __m128i isOdd16 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-1), 15);

    // cond? a : b = b ^ (aXORb & cond)
    __m128i sumOfHalf_lo = _selectXorBoth_i128(LOW_LO, FULL_LO, isOdd32);
    __m128i valIfEven_lo = _selectXorBoth_i128(A, EVEN_LO, isOdd32);

    __m128i sumOfHalf_hi = _selectXorBoth_i128(LOW_HI, FULL_HI, isOdd32);
    __m128i valIfEven_hi = _selectXorBoth_i128(E, EVEN_HI, isOdd32);

    __m128i valIfLo = _selectXorBoth_i128(valIfEven_lo, sumOfHalf_lo, isOdd16);

    __m128i valIfHi = _selectXorBoth_i128(valIfEven_hi, sumOfHalf_hi, isOdd16);

    return _either_i128(valIfHi, valIfLo, isOdd64);
}

NOINLINE __m128i viaXorC_u16(__m128i toShuff, __m128i indexes) {
    // abcd efgh
    // 00ab 00ef
    
    // _a_c _e_g
    // _0_a _0_e
    __m128i valEvenOdd = _mm_xor_si128(toShuff, _mm_slli_epi64(toShuff, 32));
    __m128i partsBoth16 = _mm_xor_si128(valEvenOdd, _mm_slli_epi32(valEvenOdd, 16));

    // 1st, Low, Even, Full
    __m128i loParts = _mm_unpacklo_epi16(partsBoth16, partsBoth16);
    __m128i hiParts = _mm_unpackhi_epi16(partsBoth16, partsBoth16);

    __m128i A =         _mm_shuffle_epi32(loParts, _MM_SHUFFLE(0,0,0,0));
    __m128i LOW_LO =    _mm_shuffle_epi32(loParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_LO =   _mm_shuffle_epi32(loParts, _MM_SHUFFLE(2,2,2,2));
    __m128i FULL_LO =   _mm_shuffle_epi32(loParts, _MM_SHUFFLE(3,3,3,3));

    __m128i E =         _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(0,0,0,0));
    __m128i LOW_HI =    _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(1,1,1,1));
    __m128i EVEN_HI =   _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(2,2,2,2));
    __m128i FULL_HI =   _mm_shuffle_epi32(hiParts, _MM_SHUFFLE(3,3,3,3));



    __m128i isOdd64 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-3), 15);
    __m128i isOdd32 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-2), 15);
    __m128i isOdd16 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-1), 15);

    // cond? a : b = b ^ (aXORb & cond)
    __m128i sumOfHalf_lo = _selectXorBoth_i128(LOW_LO, FULL_LO, isOdd32);
    __m128i valIfEven_lo = _selectXorBoth_i128(A, EVEN_LO, isOdd32);

    __m128i sumOfHalf_hi = _selectXorBoth_i128(LOW_HI, FULL_HI, isOdd32);
    __m128i valIfEven_hi = _selectXorBoth_i128(E, EVEN_HI, isOdd32);

    __m128i valIfLo = _selectXorBoth_i128(valIfEven_lo, sumOfHalf_lo, isOdd16);

    __m128i valIfHi = _selectXorBoth_i128(valIfEven_hi, sumOfHalf_hi, isOdd16);

    return _either_i128(valIfHi, valIfLo, isOdd64);
}