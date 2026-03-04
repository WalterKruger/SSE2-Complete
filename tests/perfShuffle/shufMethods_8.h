#pragma once
#include "../_perfCommon.h"


NOINLINE __m128i loopMemIndex_u8(__m128i toShuff, __m128i indexes) {
    uint8_t toShuff_s[16], indexes_s[16], result[16];

    //indexes = _mm_and_si128(indexes, *(__m128i*)&SHUFFLE_MASK_8.vec);
    //indexes = _mm_and_si128(indexes, _mm_set1_epi8(0x0f));

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 16; i++)
        result[i] = toShuff_s[indexes_s[i] & 0x0f];

    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i scalarForce_u8(__m128i toShuff, __m128i indexes) {
    uint8_t toShuff_s[16], indexes_s[16], result[16];

    //indexes = _mm_and_si128(indexes, *(__m128i*)&SHUFFLE_MASK_8.vec);
    //indexes = _mm_and_si128(indexes, _mm_set1_epi8(0x0f));

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    #if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
        #pragma unroll 0
    #elif defined(__GNUC__)
        #pragma GCC unroll 0
    #endif
    for (size_t i=0; i < 16; i++)
        result[i] = toShuff_s[indexes_s[i] & 0x0f];

    return _mm_loadu_si128((__m128i*)result);
}


#define DUPE_N4(i) CASE_Ni(i+0) CASE_Ni(i+1) CASE_Ni(i+2) CASE_Ni(i+3)
#define DUPE_N16 DUPE_N4(0) DUPE_N4(4) DUPE_N4(8) DUPE_N4(12)

NOINLINE __m128i indexUnroll_u8(__m128i toShuff, __m128i indexes) {
    const __m128i INDEX_MASK = _mm_set1_epi8(0x0f);

    static uint8_t indexes_s[16], shuffledVals[16], toShuff_s[16];

    indexes = _mm_and_si128(indexes, INDEX_MASK);
    //indexes = _mm_slli_epi16(indexes, 3); // * 8

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    #define CASE_Ni(i) shuffledVals[i] = toShuff_s[indexes_s[i]];
    DUPE_N16
    #undef CASE_Ni

    return _mm_loadu_si128((__m128i*)shuffledVals);
}

NOINLINE __m128i shiftUnroll_u8(__m128i toShuff, __m128i indexes) {
    const __m128i INDEX_MASK = _mm_set1_epi8(0x0f);

    static uint64_t toShuff_s[2];
    static uint8_t indexes_s[16], shuffledVals[16];

    indexes = _mm_and_si128(indexes, INDEX_MASK);
    indexes = _mm_slli_epi16(indexes, 3); // * 8

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    uint64_t shuffleSection; 

    #define CASE_Ni(i) \
        shuffleSection = toShuff_s[0];\
        shuffleSection  = (indexes_s[i] >= 64)? toShuff_s[1] : shuffleSection;\
        shuffledVals[i]  = shuffleSection >> (indexes_s[i] % 64);
    DUPE_N16
    #undef CASE_Ni

    return _mm_loadu_si128((__m128i*)shuffledVals);
}


NOINLINE __m128i shiftUnNoMem_u8(__m128i toShuff, __m128i indexes) {
    const __m128i INDEX_MASK = _mm_set1_epi8(0x0f);

    __m128i result = _mm_setzero_si128();

    indexes = _mm_and_si128(indexes, INDEX_MASK);
    indexes = _mm_slli_epi16(indexes, 3); // * 8

    uint64_t toShuffLo = _mm_cvtsi128_si64(toShuff);
    uint64_t toShuffHi = _mm_cvtsi128_si64(_mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2,3,2)));

    uint64_t shuffleSection;
    uint8_t curIndex;
    uint8_t curShuffed;
    uint8_t prevShuffed;

    #define CASE_Ni(i) \
        curIndex = _mm_extract_epi16(indexes, (i)/2) >> (((i)%2)? 8 : 0);\
        shuffleSection = toShuffLo;\
        shuffleSection  = (curIndex >= 64)? toShuffHi : shuffleSection;\
        curShuffed = shuffleSection >> (curIndex % 64);\
        if ((i) % 2) {result = _mm_insert_epi16(result, ((int)curShuffed << 8) | prevShuffed, (i)/2);}\
        prevShuffed = curShuffed;
    DUPE_N16
    #undef CASE_Ni

    return result;
}


NOINLINE __m128i msvc_u8(__m128i toShuff, __m128i indexes) {
    const __m128i INDEX_MASK = _mm_set1_epi8(0x0f);

    static uint8_t shuffledVals[16], toShuff_s[16];

    indexes = _mm_and_si128(indexes, INDEX_MASK);

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);

    __m128i indexInLo = indexes;
    uint32_t lo32 = 0;

    #define CASE_Ni(i)\
    lo32 = _mm_cvtsi128_si32(indexInLo);\
    indexInLo = _mm_bsrli_si128(indexes, (i)+((i) != 15));\
    shuffledVals[i] = toShuff_s[(uint8_t)lo32]; _GNUC_ONLY(__asm__(""));
    DUPE_N16
    #undef CASE_Ni

    return _mm_loadu_si128((__m128i*)shuffledVals);
}

#ifdef __GNUC__
volatile const union {uint64_t qwords[2]; __m128i vec;} SHUFFLE_MASK_8 = {{0x0f0f0f0f0f0f0f0full,0x0f0f0f0f0f0f0f0full}};

__attribute__((optimize("O3")))
NOINLINE __m128i gccAutoVec_u8(__m128i toShuff, __m128i indexes) {
    uint8_t toShuff_s[16], indexes_s[16], result[16];

    //indexes = _mm_and_si128(indexes, *(__m128i*)&SHUFFLE_MASK_8.vec);
    //indexes = _mm_and_si128(indexes, _mm_set1_epi8(0x0f));

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 16; i++)
        result[i] = toShuff_s[indexes_s[i] & 0x0f];

    return _mm_loadu_si128((__m128i*)result);
}
#endif


_GNUC_ONLY( __attribute__((target("ssse3"))) )
NOINLINE __m128i SSSE3_u8(__m128i toShuff, __m128i indexes) {
    return _mm_shuffle_epi8(toShuff, _mm_and_si128(indexes, _mm_set1_epi8(0x0f)));

}


NOINLINE __m128i viaXor_u8(__m128i toShuff, __m128i indexes) {
    __m128i loDupe = _mm_unpacklo_epi8(toShuff, toShuff);
    __m128i hiDupe = _mm_unpackhi_epi8(toShuff, toShuff);

    __m128i LO_aa_bb = _mm_shufflelo_epi16(loDupe, _MM_SHUFFLE(1,1,0,0));
    __m128i LO_cc_dd = _mm_shufflelo_epi16(loDupe, _MM_SHUFFLE(3,3,2,2));
    __m128i HI_ee_ff = _mm_shufflehi_epi16(loDupe, _MM_SHUFFLE(1,1,0,0));
    __m128i HI_gg_hh = _mm_shufflehi_epi16(loDupe, _MM_SHUFFLE(3,3,2,2));

    __m128i LO_ii_jj = _mm_shufflelo_epi16(hiDupe, _MM_SHUFFLE(1,1,0,0));
    __m128i LO_kk_ll = _mm_shufflelo_epi16(hiDupe, _MM_SHUFFLE(3,3,2,2));
    __m128i HI_mm_nn = _mm_shufflehi_epi16(hiDupe, _MM_SHUFFLE(1,1,0,0));
    __m128i HI_oo_pp = _mm_shufflehi_epi16(hiDupe, _MM_SHUFFLE(3,3,2,2));

    __m128i a = _mm_shuffle_epi32(LO_aa_bb, _MM_SHUFFLE(0,0,0,0));
    __m128i b = _mm_shuffle_epi32(LO_aa_bb, _MM_SHUFFLE(1,1,1,1));
    __m128i c = _mm_shuffle_epi32(LO_cc_dd, _MM_SHUFFLE(0,0,0,0));
    __m128i d = _mm_shuffle_epi32(LO_cc_dd, _MM_SHUFFLE(1,1,1,1));

    __m128i e = _mm_shuffle_epi32(HI_ee_ff, _MM_SHUFFLE(2,2,2,2));
    __m128i f = _mm_shuffle_epi32(HI_ee_ff, _MM_SHUFFLE(3,3,3,3));
    __m128i g = _mm_shuffle_epi32(HI_gg_hh, _MM_SHUFFLE(2,2,2,2));
    __m128i h = _mm_shuffle_epi32(HI_gg_hh, _MM_SHUFFLE(3,3,3,3));

    __m128i i = _mm_shuffle_epi32(LO_ii_jj, _MM_SHUFFLE(0,0,0,0));
    __m128i j = _mm_shuffle_epi32(LO_ii_jj, _MM_SHUFFLE(1,1,1,1));
    __m128i k = _mm_shuffle_epi32(LO_kk_ll, _MM_SHUFFLE(0,0,0,0));
    __m128i l = _mm_shuffle_epi32(LO_kk_ll, _MM_SHUFFLE(1,1,1,1));

    __m128i m = _mm_shuffle_epi32(HI_mm_nn, _MM_SHUFFLE(2,2,2,2));
    __m128i n = _mm_shuffle_epi32(HI_mm_nn, _MM_SHUFFLE(3,3,3,3));
    __m128i o = _mm_shuffle_epi32(HI_oo_pp, _MM_SHUFFLE(2,2,2,2));
    __m128i p = _mm_shuffle_epi32(HI_oo_pp, _MM_SHUFFLE(3,3,3,3));


    __m128i toSel16FF_lo = a;
    __m128i toSel16FF_HL = _mm_xor_si128(toSel16FF_lo, i);

    __m128i toSel16FT_lo = _mm_xor_si128(a, e);
    __m128i toSel16FT_HL = _mm_xor_si128(toSel16FT_lo, _mm_xor_si128(i,m));

    __m128i toSel16TF_lo = _mm_xor_si128(a, c);
    __m128i toSel16TF_HL = _mm_xor_si128(toSel16TF_lo, _mm_xor_si128(i,k));

    __m128i toSel16TT_lo = _mm_xor_si128(_mm_xor_si128(a,c), _mm_xor_si128(e,g));
    __m128i toSel16TT_HL = _mm_xor_si128(toSel16TT_lo, _mm_xor_si128(_mm_xor_si128(i,k), _mm_xor_si128(m,o)));

    __m128i toSel32F_lo = _mm_xor_si128(a, b);
    __m128i toSel32F_HL = _mm_xor_si128(toSel32F_lo, _mm_xor_si128(i,j));

    __m128i toSel32T_lo = _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(e, f));
    __m128i toSel32T_HL = _mm_xor_si128(toSel32T_lo, _mm_xor_si128(_mm_xor_si128(i, j), _mm_xor_si128(m, n)));


    __m128i toSel64_lo = _mm_xor_si128(_mm_xor_si128(a,b), _mm_xor_si128(c,d));
    __m128i toSel64_HL = _mm_xor_si128(toSel64_lo, _mm_xor_si128(_mm_xor_si128(i,j), _mm_xor_si128(k,l)) );

    __m128i LOW = _mm_xor_si128(toSel64_lo, _mm_xor_si128(_mm_xor_si128(e,f), _mm_xor_si128(g,h)));
    __m128i FULL = _mm_xor_si128(LOW, _mm_xor_si128(_mm_xor_si128(i,j), _mm_xor_si128(k,l)) );
    FULL = _mm_xor_si128(FULL, _mm_xor_si128(_mm_xor_si128(m,n), _mm_xor_si128(o,p)) );



    
    __m128i isEven64 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-4), _mm_setzero_si128());

    __m128i cur =       _selectXorBoth_i128(LOW, FULL, isEven64);
    __m128i toSel64 =   _selectXorBoth_i128(toSel64_lo, toSel64_HL, isEven64);

    __m128i toSel32T =  _selectXorBoth_i128(toSel32T_lo, toSel32T_HL, isEven64);
    __m128i toSel32F =  _selectXorBoth_i128(toSel32F_lo, toSel32F_HL, isEven64);

    __m128i toSel16TT = _selectXorBoth_i128(toSel16TT_lo, toSel16TT_HL, isEven64);
    __m128i toSel16TF = _selectXorBoth_i128(toSel16TF_lo, toSel16TF_HL, isEven64);
    __m128i toSel16FT = _selectXorBoth_i128(toSel16FT_lo, toSel16FT_HL, isEven64);
    __m128i toSel16FF = _selectXorBoth_i128(toSel16FF_lo, toSel16FF_HL, isEven64);


    __m128i isEven32 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-3), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel64, cur, isEven32);
    __m128i toSel32 =   _selectXorBoth_i128(toSel32F, toSel32T, isEven32);
    __m128i toSel16T =  _selectXorBoth_i128(toSel16TF, toSel16TT, isEven32);
    __m128i toSel16F =  _selectXorBoth_i128(toSel16FF, toSel16FT, isEven32);


    __m128i isEven16 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-2), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel32, cur, isEven16);
    __m128i toSel16 = _selectXorBoth_i128(toSel16F, toSel16T, isEven16);


    __m128i isEven8 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-1), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel16, cur, isEven8);

    return cur;
}


NOINLINE __m128i viaXorA_u8(__m128i toShuff, __m128i indexes) {
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
    __m128i sumOfHalf_lolo = _selectXorBoth_i128(LOW_LOLO, FULL_LOLO, isOdd16);
    __m128i valIfEven_lolo = _selectXorBoth_i128(A, EVEN_LOLO, isOdd16);

    __m128i sumOfHalf_lohi = _selectXorBoth_i128(LOW_LOHI, FULL_LOHI, isOdd16);
    __m128i valIfEven_lohi = _selectXorBoth_i128(E, EVEN_LOHI, isOdd16);

    __m128i sumOfHalf_hilo = _selectXorBoth_i128(LOW_HILO, FULL_HILO, isOdd16);
    __m128i valIfEven_hilo = _selectXorBoth_i128(I, EVEN_HILO, isOdd16);

    __m128i sumOfHalf_hihi = _selectXorBoth_i128(LOW_HIHI, FULL_HIHI, isOdd16);
    __m128i valIfEven_hihi = _selectXorBoth_i128(M, EVEN_HIHI, isOdd16);


    __m128i valIfLoLo = _selectXorBoth_i128(valIfEven_lolo, sumOfHalf_lolo, isOdd8);
    __m128i valIfLoHi = _selectXorBoth_i128(valIfEven_lohi, sumOfHalf_lohi, isOdd8);
    __m128i valIfHiLo = _selectXorBoth_i128(valIfEven_hilo, sumOfHalf_hilo, isOdd8);
    __m128i valIfHiHi = _selectXorBoth_i128(valIfEven_hihi, sumOfHalf_hihi, isOdd8);


    __m128i valIfLo = _either_i128(valIfLoHi, valIfLoLo, isOdd32);
    __m128i valIfHi = _either_i128(valIfHiHi, valIfHiLo, isOdd32);

    return _either_i128(valIfHi, valIfLo, isOdd64);
}


NOINLINE __m128i viaXorB_u8(__m128i toShuff, __m128i indexes) {
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
    __m128i sumOfHalf_lolo = _selectXorBoth_i128(LOW_LOLO, FULL_LOLO, isOdd16);
    __m128i valIfEven_lolo = _selectXorBoth_i128(A, EVEN_LOLO, isOdd16);

    __m128i sumOfHalf_lohi = _selectXorBoth_i128(LOW_LOHI, FULL_LOHI, isOdd16);
    __m128i valIfEven_lohi = _selectXorBoth_i128(E, EVEN_LOHI, isOdd16);

    __m128i sumOfHalf_hilo = _selectXorBoth_i128(LOW_HILO, FULL_HILO, isOdd16);
    __m128i valIfEven_hilo = _selectXorBoth_i128(I, EVEN_HILO, isOdd16);

    __m128i sumOfHalf_hihi = _selectXorBoth_i128(LOW_HIHI, FULL_HIHI, isOdd16);
    __m128i valIfEven_hihi = _selectXorBoth_i128(M, EVEN_HIHI, isOdd16);


    __m128i valIfLoLo = _selectXorBoth_i128(valIfEven_lolo, sumOfHalf_lolo, isOdd8);
    __m128i valIfLoHi = _selectXorBoth_i128(valIfEven_lohi, sumOfHalf_lohi, isOdd8);
    __m128i valIfHiLo = _selectXorBoth_i128(valIfEven_hilo, sumOfHalf_hilo, isOdd8);
    __m128i valIfHiHi = _selectXorBoth_i128(valIfEven_hihi, sumOfHalf_hihi, isOdd8);


    __m128i LOW =  _mm_xor_si128(valIfLoLo, valIfLoHi);
    __m128i HIGH = _mm_xor_si128(valIfHiLo, valIfHiHi);
    __m128i FULL = _mm_xor_si128(LOW, HIGH);
    __m128i EVEN = _mm_xor_si128(valIfLoLo, valIfHiLo);

    __m128i sumOfHalf = _selectXorBoth_i128(LOW, FULL, isOdd64);
    __m128i valIfEven = _selectXorBoth_i128(valIfLoLo, EVEN, isOdd64);

    return _selectXorBoth_i128(valIfEven, sumOfHalf, isOdd32);
}


NOINLINE __m128i bitByBit_u8(__m128i toShuff, __m128i indexes) {
    // 0b_hgfe_dcba
    // 0b_0001_0000
    
    __m128i indexBitSet = _mm_max_epu8(
        _mm_set1_epi8(1), _mm_and_si128(_mm_set1_epi8(1<<4), _mm_slli_epi32(indexes, 2)) 
    );
    __m128i mskCount = _mm_slli_epi32(indexes, 6);

    __m128i shiftsBy2 = _fillWithMSB_i8x16(mskCount);
    indexBitSet = _mm_max_epu8(_mm_and_si128(shiftsBy2, _mm_slli_epi32(indexBitSet, 2)), indexBitSet);
    mskCount = _mm_add_epi8(mskCount,mskCount);

    indexBitSet = _mm_add_epi8(indexBitSet, _mm_and_si128(indexBitSet, _fillWithMSB_i8x16(mskCount)));


    toShuff = _mm_move_epi64(toShuff); // Saves on movemask cast
    __m128i result = _mm_setzero_si128();

    #pragma GCC unroll 8
    for (size_t i = 0; i < 8; i++) {
        __m128i eachElementBit = _mm_set1_epi32(_mm_movemask_epi8(toShuff) * 0x01010101);
        //__m128i eachElementBit = _mm_set1_epi8((char)_mm_movemask_epi8(toShuff));

        __m128i targetBit = _mm_cmpeq_epi8(_mm_andnot_si128(eachElementBit, indexBitSet), _mm_setzero_si128());

        result = _mm_sub_epi8(_mm_add_epi8(result, result), targetBit);

        toShuff = _mm_add_epi8(toShuff, toShuff); // << 1
    }

    return result;
}


NOINLINE __m128i bitByBitA_u8(__m128i toShuff, __m128i indexes) {
    // 0b_hgfe_dcba
    // 0b_0001_0000
    
    __m128i indexBitSet = _mm_max_epu8(
        _mm_set1_epi8(1), _mm_and_si128(_mm_set1_epi8(1<<4), _mm_slli_epi32(indexes, 2)) 
    );
    __m128i mskCount = _mm_slli_epi32(indexes, 6);

    __m128i shiftsBy2 = _fillWithMSB_i8x16(mskCount);
    indexBitSet = _mm_max_epu8(_mm_and_si128(shiftsBy2, _mm_slli_epi32(indexBitSet, 2)), indexBitSet);
    mskCount = _mm_add_epi8(mskCount,mskCount);

    indexBitSet = _mm_add_epi8(indexBitSet, _mm_and_si128(indexBitSet, _fillWithMSB_i8x16(mskCount)));


    toShuff = _mm_move_epi64(toShuff); // Saves on movemask cast
    __m128i bitPos = _mm_cvtsi32_si128(7);
    __m128i result = _mm_setzero_si128();

    #pragma GCC unroll 8
    for (size_t i = 0; i < 8; i++) {
        int slice = _mm_movemask_epi8(_mm_sll_epi64(toShuff, bitPos));
        __m128i eachElementBit = _mm_set1_epi32(slice * 0x01010101);

        __m128i targetBit = _mm_cmpeq_epi8(_mm_andnot_si128(eachElementBit, indexBitSet), _mm_setzero_si128());
        result = _mm_avg_epu8(result, targetBit);

        bitPos = _mm_add_epi64(bitPos, _mm_set1_epi64x(-1));
    }

    return result;
}

