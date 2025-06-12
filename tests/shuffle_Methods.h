#pragma once

#include <stdint.h>
#include <immintrin.h> // sse2 & later

#include "../include/sseComplete.h"
#include "_perfCommon.h"

#define sAsI128(x) _mm_castps_si128(x)
#define dAsI128(x) _mm_castpd_si128(x)
#define iAsPd(x) _mm_castsi128_pd(x)
#define iAsPs(x) _mm_castsi128_ps(x)


// Uses a condition mask to select between two values, when one of them is a precomputed xor between the two
// cond? a : b = a ^ (aXORb & cond)
static inline __m128i _selectXorBoth_i128(__m128i valIfNot, __m128i bothValXor, __m128i conditionMask) {
    return _mm_xor_si128(valIfNot, _mm_and_si128(bothValXor, conditionMask));
}




// To auto-generate the 256 case switch statement/asm jump table 
#define DUP_N4(n) DUP_Ni(n+0) DUP_Ni(n+1) DUP_Ni(n+2) DUP_Ni(n+3)
#define DUP_N8(  n) DUP_N4( n+0) DUP_N4( n+4)
#define DUP_N16( n) DUP_N8( n+0) DUP_N8( n+8)
#define DUP_N32( n) DUP_N16(n+0) DUP_N16(n+16)
#define DUP_N64( n) DUP_N32(n+0) DUP_N32(n+32)
#define DUP_N128(n) DUP_N64(n+0) DUP_N64(n+64)

#define DUPLICATE_256 DUP_N128(0) DUP_N128(128)

#define PRIMATIVE_STR(x) #x
#define STR(x) PRIMATIVE_STR(x)




// ====== 64-bit ======



NOINLINE __m128i scalar_u64(__m128i toShuff, __m128i indexes) {
    uint64_t toShuff_s[2], indexes_s[2], result[2];

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 2; i++)
        result[i] = toShuff_s[indexes_s[i] & 0b1];

    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i cmov_u64(__m128i toShuff, __m128i indexes) {
    
    __m128i toShuffHi = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2, 3,2));
    __m128i indexesHi = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(3,2, 3,2));

    uint64_t partLo = _mm_cvtsi128_si64(toShuff);
    uint64_t partHi = _mm_cvtsi128_si64(toShuffHi);

    uint64_t indexLo = _mm_cvtsi128_si64(indexes);
    uint64_t indexHi = _mm_cvtsi128_si64(indexesHi);

    uint64_t resLo = partLo; resLo = (indexLo & 1)? partHi : partLo;
    uint64_t resHi = partLo; resHi = (indexHi & 1)? partHi : partLo;

    return _mm_unpacklo_epi64(_mm_cvtsi64_si128(resLo), _mm_cvtsi64_si128(resHi));
}


NOINLINE __m128i selectHiLo_u64(__m128i toShuff, __m128i indexes) {

    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsHi = _mm_srai_epi32(_mm_slli_epi32(dupLo32, 31), 31);

    __m128i loDupe = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 1,0));
    __m128i hiDupe = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2, 3,2));

    return _either_i128(hiDupe, loDupe, selectsHi);
}

NOINLINE __m128i selecRev_u64(__m128i toShuff, __m128i indexes) {
    
    // Shift clears all but LSB
    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsRev = _mm_cmpeq_epi32( _mm_slli_epi32(dupLo32, 31), _mm_setr_epi32(1<<31, 1<<31, 0, 0) );

    __m128i toShuffRev = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 3,2));

    return _either_i128(toShuffRev, toShuff, selectsRev);
}



// ====== 32-bit ======



NOINLINE __m128i switch_u32(__m128i toShuff, __m128i indexes) {
    // 
    __m128i twoLSB_inMSB = _mm_slli_epi32(indexes, 32-2);
    __m128i twoLSB_inMSB_of16Lane = _mm_srai_epi32(twoLSB_inMSB, 16-1);

    int indexes_lowBits = _getMsb_i16x8(twoLSB_inMSB_of16Lane);

    switch(indexes_lowBits) {
        #define DUP_Ni(i) case (i): return _mm_shuffle_epi32(toShuff, i);
        DUPLICATE_256
        #undef DUP_Ni
    }

    // To Prevent warnings
    return toShuff;
}

// For switches compiler don't jump into the instruction
#ifdef GNUC_EXTENTION
NOINLINE __m128i instructionJmp_u32(__m128i toShuff, __m128i indexes) {
    __m128i twoLSB_inMSB = _mm_slli_epi32(indexes, 32-2);
    __m128i twoLSB_inMSB_of16Lane = _mm_srai_epi32(twoLSB_inMSB, 16-1);

    int indexes_lowBits = _getMsb_i16x8(twoLSB_inMSB_of16Lane);
    
    uint64_t tablePtrStorage = 0;

    __asm__ volatile(
        // Calculate the absolute address based on the instruction pointer
        "leaq    shufJmpTable_%=(%%rip), %[tablePtr];"
        "leaq    (%[tablePtr], %[idx], 8), %[tablePtr];" // 6 without padding

        "jmpq     *%[tablePtr];"
    ".align 64; shufJmpTable_%=:  ;"
        #define DUP_Ni(i) "pshufd  $("STR(i)"), %[toShuff], %[toShuff]; retq;""nop;nop;"
        DUPLICATE_256
        #undef DUP_Ni
    
     : [tablePtr] "=&r" (tablePtrStorage) 
     : [idx] "r" ((uint64_t)indexes_lowBits), [toShuff] "Yz" (toShuff)
    );

    // To Prevent warnings
    return toShuff;
}
#endif


NOINLINE __m128i viaXor_u32(__m128i toShuff, __m128i indexes) {
    __m128i a = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(0,0,0,0));
    __m128i b = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,1,1,1));
    __m128i c = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(2,2,2,2));
    __m128i d = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,3,3,3));

    __m128i LO = _mm_xor_si128(a, b);
    __m128i HI = _mm_xor_si128(c, d);
    __m128i FULL = _mm_xor_si128(LO, HI);

    #if 0
    // Mask out only the indexes
    indexes = _mm_and_si128(indexes, _mm_set1_epi32(0b11));


    __m128i isInHi = _mm_cmpgt_epi32(indexes, _mm_set1_epi32(1));
    __m128i part_half = _selectXorBoth_i128(LO, FULL, isInHi);

    
    __m128i isInOdd = _mm_srai_epi32(_mm_slli_epi32(indexes, 31), 31);
    #else

    __m128i isInHi = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-2), 31);
    __m128i part_half = _selectXorBoth_i128(LO, FULL, isInHi);

    
    __m128i isInOdd = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-1), 31);
    #endif

    // Remove the part that isn't selected
    __m128i ifInHiRemove = _selectXorBoth_i128(d, HI, isInOdd);
    __m128i ifInLoRemove = _selectXorBoth_i128(b, LO, isInOdd);

    return _mm_xor_si128(part_half, _either_i128(ifInHiRemove, ifInLoRemove, isInHi));
}


_GNUC_ONLY(__attribute__((target("avx2"))))
NOINLINE __m128i AVX_u32(__m128i toShuff, __m128i indexes) {
    __m128 result_ps = _mm_permutevar_ps(_mm_castsi128_ps(toShuff), indexes);
    
    return _mm_castps_si128(result_ps);
}


NOINLINE __m128i shift_u32(__m128i toShuff, __m128i indexes) {
    uint32_t indexes_s[4], result[4];
    uint64_t toShuff_half[2];

    indexes = _mm_and_si128(indexes, _mm_set1_epi32(0b11));
    indexes = _mm_slli_epi32(indexes, 5); // * 32

    _mm_storeu_si128((__m128i*)toShuff_half, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    uint64_t half0 = toShuff_half[0]; half0 = (indexes_s[0] >= 64)? toShuff_half[1] : half0;
    uint64_t half1 = toShuff_half[0]; half1 = (indexes_s[1] >= 64)? toShuff_half[1] : half1;
    uint64_t half2 = toShuff_half[0]; half2 = (indexes_s[2] >= 64)? toShuff_half[1] : half2;
    uint64_t half3 = toShuff_half[0]; half3 = (indexes_s[3] >= 64)? toShuff_half[1] : half3;
        
    result[0] = half0 >> (indexes_s[0] % 64);
    result[1] = half1 >> (indexes_s[1] % 64);
    result[2] = half2 >> (indexes_s[2] % 64);
    result[3] = half3 >> (indexes_s[3] % 64);

    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i gcc_u32(__m128i toShuff, __m128i indexes) {
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

//FIXME: Doesn't work...
NOINLINE __m128i clang_u32(__m128i toShuff, __m128i indexes) {
    indexes = _mm_slli_epi32(indexes, 5);
    indexes = _mm_and_si128(indexes, _mm_set1_epi32(0b11<<5));

    __m128i isGrt64 = _mm_cmpgt_epi32(indexes, _mm_set1_epi32(64));

    __m128i isGrt_00_11 = _mm_shuffle_epi32(isGrt64, 250);
    __m128i isGrt_22_33 = _mm_shuffle_epi32(isGrt64, 80);

    __m128i hiDupe = _mm_shuffle_epi32(toShuff, 238);
    __m128i loDupe = _mm_shuffle_epi32(toShuff, 68);


    __m128i xmm3 = _either_i128(hiDupe, loDupe, isGrt_00_11);
    __m128i xmm0 = _either_i128(hiDupe, loDupe, isGrt_22_33);

    __m128i xmm4 = _mm_unpackhi_epi32(indexes, _mm_setzero_si128());

    __m128i xmm5 = _mm_srl_epi64(xmm3, xmm4);

    xmm4 = _mm_bsrli_si128(indexes, 12);

    xmm3 = _mm_srl_epi64(xmm3, xmm4);

    xmm4 = _mm_srl_epi64( xmm0, sAsI128(_mm_move_ss(_mm_setzero_ps(), iAsPs(indexes))) );

    indexes = _mm_srli_epi64(indexes, 32);
    xmm0 = _mm_srl_epi64(xmm0, indexes);

    xmm3 = dAsI128( _mm_move_sd(iAsPd(xmm3), iAsPd(xmm5)) );
    xmm0 = dAsI128( _mm_move_sd(iAsPd(xmm0), iAsPd(xmm4)) );

    return sAsI128(  _mm_shuffle_ps(iAsPs(xmm0), iAsPs(xmm3), 136) );
}


NOINLINE __m128i shiftSIMD_u32(__m128i toShuff, __m128i indexes) {
    indexes = _mm_slli_epi32(indexes, 5);
    indexes = _mm_and_si128(indexes, _mm_set1_epi32(0b11<<5));

    __m128i isGrt64 = _mm_cmpgt_epi32(indexes, _mm_set1_epi32(64-1));

    __m128i loPart = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0,1,0));
    __m128i hiPart = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2,3,2));

    __m128i loIsGrt64 = _mm_shuffle_epi32(isGrt64, _MM_SHUFFLE(1,1,0,0));
    __m128i hiIsGrt64 = _mm_shuffle_epi32(isGrt64, _MM_SHUFFLE(3,3,2,2));

    __m128i loSelected = _either_i128(hiPart, loPart, loIsGrt64);
    __m128i hiSelected = _either_i128(hiPart, loPart, hiIsGrt64);

    uint32_t indexesArr[4];
    _mm_storeu_si128((__m128i*)indexesArr, indexes);

    uint32_t resLoLo = _mm_cvtsi128_si64(loSelected) >> indexesArr[0];
    uint32_t resLoHi = _mm_cvtsi128_si64(_mm_bsrli_si128(loSelected,8)) >> indexesArr[1];
    uint32_t resHiLo = _mm_cvtsi128_si64(hiSelected) >> indexesArr[2];
    uint32_t resHiHi = _mm_cvtsi128_si64(_mm_bsrli_si128(hiSelected,8)) >> indexesArr[3];

    return _mm_setr_epi32(resLoLo, resLoHi, resHiLo, resHiHi);
}


NOINLINE __m128i scalar_u32(__m128i toShuff, __m128i indexes) {
    uint32_t toShuff_s[4], indexes_s[4], result[4];

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 4; i++)
        result[i] = toShuff_s[indexes_s[i] & 0b11];

    return _mm_loadu_si128((__m128i*)result);
}



// ====== 16-bit ======



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



// ====== 8-bit ======



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
    uint8_t x[16]; _mm_storeu_si128((__m128i*)x, toShuff);
    /*
    even64: 1^2^3^4^5^6^7^8
    odd64: 9^10^11^12^13^14^15^16

    even32: 1^2^3^4 ^ 9^10^11^12
    odd32: 5^6^7^8 ^ 13^14^15^16

    even16: 1^2 ^ 5^6 ^ 9^10 ^ 13^14
    odd16: 3^4 ^ 7^8 ^ 11^12 ^ 15^16
    */
   // Set all 8-bit lanes to the cumlative XOR between all 8-bit elements within each even/odd N-bit lane

    __m128i even64 = _mm_set1_epi8(x[0]^x[1]^x[2]^x[3]^x[4]^x[5]^x[6]^x[7]);
    __m128i odd64 = _mm_set1_epi8(x[8]^x[9]^x[10]^x[11]^x[12]^x[13]^x[14]^x[15]);

    // The `T` suffex indicates values that overlap with the pervious odd values
    // FIXME: During the o/e16 it gains two additional terms instead of removing them
    __m128i evenT32 = _mm_set1_epi8(x[8]^x[9]^x[10]^x[11]);
    __m128i evenF32 = _mm_set1_epi8(x[0]^x[1]^x[2]^x[3]);
    __m128i oddT32 = _mm_set1_epi8(x[12]^x[13]^x[14]^x[15]);
    __m128i oddF32 = _mm_set1_epi8(x[4]^x[5]^x[6]^x[7]);

    __m128i evenT16 = _mm_set1_epi8(x[4]^x[5] ^ x[12]^x[13]);
    __m128i evenF16 = _mm_set1_epi8(x[0]^x[1] ^ x[8]^x[9]);
    __m128i oddT16 = _mm_set1_epi8(x[6]^x[7] ^ x[14]^x[15]);
    __m128i oddF16 = _mm_set1_epi8(x[2]^x[3] ^ x[10]^x[11]);

    __m128i evenT8 = _mm_set1_epi8(x[2] ^ x[6] ^ x[10] ^ x[14]);
    __m128i evenF8 = _mm_set1_epi8(x[0] ^ x[4] ^ x[8] ^ x[12]);
    __m128i oddT8 = _mm_set1_epi8(x[3] ^ x[7] ^ x[11] ^ x[15]);
    __m128i oddF8 = _mm_set1_epi8(x[1] ^ x[5] ^ x[9] ^ x[13]);

    // Indexes account for bits that are in range
    indexes = _mm_and_si128(indexes, _mm_set1_epi8(0x0F));

    __m128i isOdd64 = _mm_cmpgt_epi8(indexes, _mm_set1_epi8(7));
    __m128i isOdd32 = _mm_cmpeq_epi8(_mm_and_si128(indexes, _mm_set1_epi8(1<<2)), _mm_setzero_si128());
    __m128i isOdd16 = _mm_cmpeq_epi8(_mm_and_si128(indexes, _mm_set1_epi8(1<<1)), _mm_setzero_si128());
    __m128i isOdd8 = _mm_cmpeq_epi8(_mm_and_si128(indexes, _mm_set1_epi8(1)), _mm_setzero_si128());

    // Returns the values to remove (oposite) => conditionIsOdd? even  : odd
    struct sectionRemove {__m128i curF_PrevF; __m128i curF_PrevT; __m128i curT_PrevF; __m128i curT_PrevT; __m128i curCond;}
    SECTIONS[] = {
        //{odd64, even64, isOdd64},
        {oddF32, oddT32, evenF32, evenT32, isOdd32},
        {oddF16, oddT16, evenF16, evenT16, isOdd16},
        {oddF8, oddT8, evenF8, evenT8, isOdd8}
    };

    
    __m128i partialShuff = _either_i128(odd64, even64, isOdd64);
    __m128i prevCond = isOdd64;
    for (size_t i=0; i<3; i++) {
        struct sectionRemove curToRemove = SECTIONS[i];

        __m128i removeIfPrevF = _either_i128(curToRemove.curT_PrevF, curToRemove.curF_PrevF, curToRemove.curCond);
        __m128i removeIfPrevT = _either_i128(curToRemove.curT_PrevT, curToRemove.curF_PrevT, curToRemove.curCond);

        partialShuff = _mm_xor_si128(partialShuff, _either_i128(removeIfPrevT, removeIfPrevF, prevCond));
        prevCond = curToRemove.curCond;
    }

    return partialShuff; 
}
