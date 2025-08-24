#pragma once

#include <stdint.h>
#include <emmintrin.h> // SSE2

#include "../include/sseComplete.h"


#define SSE_LITERAL_UNION_DBL static const volatile union {double array[2]; __m128d vec;}


// ===== 32-bit int to float32 =====

NOINLINE __m128 magicExpo_u32ToF32(__m128i u32_toCvt) {
    // Magic method based on clang (when converting a uint64 => double)
    const uint32_t MAGIC_EXP_LO = (FLT_MAX_EXP + FLT_MANT_DIG - 2) << (FLT_MANT_DIG-1);
    const uint32_t MAGIC_EXP_HI = (FLT_MAX_EXP + FLT_MANT_DIG+16 - 2) << (FLT_MANT_DIG-1);

    const float MAGIC_TERM_LO =  1ull << (FLT_MANT_DIG-1);
    const float MAGIC_TERM_HI =  1ull << (FLT_MANT_DIG-1+16);


    __m128i input_lo = _mm_and_si128(u32_toCvt, _mm_set1_epi32(UINT16_MAX));
    __m128i input_hi = _mm_srli_epi32(u32_toCvt, 16);

    // [..., MAGIC, inp1.lo, MAGIC, inp0.lo]
    __m128 magicFlt_lo = _mm_castsi128_ps(_mm_or_si128(_mm_set1_epi32(MAGIC_EXP_LO), input_lo));
    __m128 magicFlt_hi = _mm_castsi128_ps(_mm_or_si128(_mm_set1_epi32(MAGIC_EXP_HI), input_hi));

    magicFlt_lo = _mm_sub_ps(magicFlt_lo, _mm_set1_ps(MAGIC_TERM_LO));
    magicFlt_hi = _mm_sub_ps(magicFlt_hi, _mm_set1_ps(MAGIC_TERM_HI));

    return _mm_add_ps(magicFlt_hi, magicFlt_lo);
}

NOINLINE __m128 scaleBranchless_u32ToF32(__m128i u32_toCvt) {
    const __m128i LSB_MASK = _mm_set1_epi32(1);

    // Since conversion function assumes a signed input, we need to conditionally scale large input
    __m128i needToScale = _mm_srai_epi32(u32_toCvt, 31);

    // Half as int so in range then restore original value by doubling as float
    __m128i halfDown = _mm_srli_epi32(u32_toCvt, 1);

    __m128i toCvt_LSB = _mm_and_si128(u32_toCvt, LSB_MASK);

    // At this range the float's ULP > 1, so conversion is exact but only if halfing rounds up
    // Round up: `(x >> 1) | (x & 1)` Do nothing `[ (x>>1) + (x>>1) ] | (x & 1)`
    __m128i halfDownOrRemoveLSB = _mm_add_epi32(halfDown, _mm_andnot_si128(needToScale, halfDown));
    __m128i halfUpOrInput = _mm_or_si128(halfDownOrRemoveLSB, toCvt_LSB);
    

    __m128 cvted_or_cvtedHalf = _mm_cvtepi32_ps(halfUpOrInput);

    // Doubling restores original value, but only if we previously halfed it
    __m128 halfedAsInt_flt = _mm_castsi128_ps(needToScale);
    return _mm_add_ps(cvted_or_cvtedHalf, _mm_and_ps(halfedAsInt_flt, cvted_or_cvtedHalf));
}

NOINLINE __m128 halfElement_u32ToF32(__m128i u32_toCvt) {
    // Convert the high and low 16-bit seperatly, then recombind
    __m128i low16Bits = _mm_and_si128(u32_toCvt, _mm_set1_epi32(UINT16_MAX));
    __m128i hi16Bits = _mm_srli_epi32(u32_toCvt, 16);

    __m128 hiCvted =  _mm_cvtepi32_ps(hi16Bits);
    __m128 lowCvted = _mm_cvtepi32_ps(low16Bits);

    // lowCvt + (hiCvt << 16)
    return _mm_add_ps(lowCvted, _mm_mul_ps(hiCvted, _mm_set1_ps(1<<16)));
}

NOINLINE __m128 halfElementA_u32ToF32(__m128i u32_toCvt) {
    __m128i low16Bits = _mm_and_si128(u32_toCvt, _mm_set1_epi32(UINT16_MAX));
    __m128i hi16Bits = _mm_srli_epi32(u32_toCvt, 16);

    __m128 hiBitsCvted =  _mm_cvtepi32_ps(hi16Bits);
    __m128 lowBitsCvted = _mm_cvtepi32_ps(low16Bits);

    __m128i hiCvtedInt = _mm_castps_si128(hiBitsCvted);

    // lowCvt + (hiCvt << 16)
    __m128i hiShifted = _mm_add_epi32(
        hiCvtedInt, _mm_and_si128( _mm_castps_si128(_mm_cmpneq_ps(hiBitsCvted, _mm_setzero_ps() )), _mm_set1_epi32(16 << (FLT_MANT_DIG-1) ))
        );
    return _mm_add_ps(lowBitsCvted, _mm_castsi128_ps(hiShifted));
}

NOINLINE __m128 compiler_u32ToF32(__m128i u32_toCvt) {
    uint32_t u32_array[4];
    _mm_store_si128((__m128i*)u32_array, u32_toCvt);

    return _mm_setr_ps((float)u32_array[0], (float)u32_array[1], (float)u32_array[2], (float)u32_array[3]);
}


// ===== Unsigned 32-bit int to float64 =====

// Alternative method: signed32ToDbl(x) + ( ((int32_t)x < 0)? (1ull<<32) : 0 )
NOINLINE __m128d correctAfterCvt_u32ToF64(__m128i u32_toCvt) {
    __m128d CORRECT_TERM = _mm_set1_pd(1ull << 32);

    __m128d signedCvt = _mm_cvtepi32_pd(u32_toCvt);
    __m128i needToCorrect = _mm_srai_epi32(_mm_shuffle_epi32(u32_toCvt, _MM_SHUFFLE(1, 1, 0, 0)), 31);

    return _mm_add_pd(signedCvt, _mm_and_pd(CORRECT_TERM, _mm_castsi128_pd(needToCorrect)));
}

// Compilers misoptimize by replacing `sub(or(x, MAGIC), MAGIC)` with `add(or(x, MAGIC1), MAGIC2)`
// Which requires an additional 128-bit load or two 64-bit loads + broadcast to vector (GCC)
SSE_LITERAL_UNION_DBL _convMethod_u32Cvt_MAGIC = {
    {1ull << (DBL_MANT_DIG-1), 1ull << (DBL_MANT_DIG-1)}
};

NOINLINE __m128d mantissaDepo_u32ToF64(__m128i u32_toCvt) {
    // Setting the exponent to 2**52 "shifts" the entire mantissa into the integer part
    // However, this also shifts over the implict leading one

    // Constant is volatile, so this prevents double loading 
    __m128d MAGIC_LOCAL = _convMethod_u32Cvt_MAGIC.vec;

    __m128d toCvt_u64 = _mm_castsi128_pd( _zeroExtendLo_u32x4_i64x2(u32_toCvt) );
    return _mm_sub_pd(_mm_or_pd(toCvt_u64, MAGIC_LOCAL), MAGIC_LOCAL);
}

NOINLINE __m128d scalarInstruction_u32ToF64(__m128i u32_toCvt) {
    // Statement ordering used to manipulate asm output
    __m128i UpperInLow = _mm_shuffle_epi32(u32_toCvt, _MM_SHUFFLE(0, 0, 0, 1));
    
    // Use the signed int64 => float64 instruction since uint32 in range of int64
    uint32_t u32_lo = _mm_cvtsi128_si32(u32_toCvt);
    __m128d dbl_lo = _mm_cvtsi64_sd(_mm_setzero_pd(), u32_lo);
    
    int64_t u32_hi = _mm_cvtsi128_si32(UpperInLow);
    __m128d dbl_hi = _mm_cvtsi64_sd(_mm_setzero_pd(), u32_hi);

    return _mm_unpacklo_pd(dbl_lo, dbl_hi);
}


NOINLINE __m128d compiler_u32ToF64(__m128i u32_toCvt) {
    uint32_t u32_array[4];
    _mm_store_si128((__m128i*)u32_array, u32_toCvt);

    return _mm_setr_pd((double)u32_array[0], (double)u32_array[1]);
}



// ===== Signed 64-bit int to float64 =====

NOINLINE __m128 scalarInstruction_i64ToF32(__m128i i64_toCvt) {
    #ifdef __clang__
    // Clang produces multiple redundent zero vectors with the other code...
    int64_t i64_array[2];
    _mm_store_si128((__m128i*)i64_array, i64_toCvt);

    return _mm_setr_ps((float)i64_array[0], (float)i64_array[1], 0, 0);

    #else
    // Statement ordering used to manipulate asm output
    __m128i hiInLow = _mm_shuffle_epi32(i64_toCvt, 0b00001110);
    
    int64_t i64_lo = _mm_cvtsi128_si64(i64_toCvt);
    __m128 dbl_lo = _mm_cvtsi64_ss(_mm_setzero_ps(), i64_lo);
    
    int64_t i64_hi = _mm_cvtsi128_si64(hiInLow);
    __m128 dbl_hi = _mm_cvtsi64_ss(_mm_setzero_ps(), i64_hi);

    return _mm_unpacklo_ps(dbl_lo, dbl_hi);
    #endif
}


NOINLINE __m128 compiler_i64ToF32(__m128i i64_toCvt) {
    int64_t i64_array[2];
    _mm_store_si128((__m128i*)i64_array, i64_toCvt);

    return _mm_setr_ps((float)i64_array[0], (float)i64_array[1], 0, 0);
}



// ===== Unsigned 64-bit int to float32 =====

NOINLINE __m128 scaleBranchless_u64ToF32(__m128i u64_toCvt) {
    const __m128i LSB_MASK = _mm_set1_epi64x(1);

    // Since conversion function assumes a signed input, we need to conditionally scale large input
    __m128i needToScale = _fillWithMSB_i64x2(u64_toCvt);

    // Half as int so in range then restore original value by doubling as float
    __m128i halfDown = _mm_srli_epi64(u64_toCvt, 1);

    __m128i toCvt_LSB = _mm_and_si128(u64_toCvt, LSB_MASK);

    // At this range the float's ULP > 1, so conversion is exact but only if halfing rounds up
    // Round up: `(x >> 1) | (x & 1)` Do nothing `[ (x>>1) + (x>>1) ] | (x & 1)`
    __m128i halfDownOrRemoveLSB = _mm_add_epi64(halfDown, _mm_andnot_si128(needToScale, halfDown));
    __m128i halfUpOrInput = _mm_or_si128(halfDownOrRemoveLSB, toCvt_LSB);
    

    __m128 cvted_or_cvtedHalf = _convert_i64x2_f32x4(halfUpOrInput);

    // Doubling restores original value, but only if we previously halfed it
    __m128 halfedAsInt_flt = _mm_castsi128_ps( _mm_bsrli_si128(needToScale, 4) ); // 64-bit mask => 32-bit
    return _mm_add_ps(cvted_or_cvtedHalf, _mm_and_ps(halfedAsInt_flt, cvted_or_cvtedHalf));
}


NOINLINE __m128 scaleLin_u64ToF32(__m128i u64_toCvt) {
    __m128 cvted_lo, cvted_hi;

    uint64_t u64_hi = _mm_cvtsi128_si64(
        _mm_shuffle_epi32(u64_toCvt, _MM_SHUFFLE(0,0,3,2))
    );

    uint64_t u64_lo = _mm_cvtsi128_si64(u64_toCvt);

    if ((int64_t)u64_lo >= 0) {
        cvted_lo = _mm_cvtsi64_ss(_mm_setzero_ps(), u64_lo);
    } else {
        int64_t halfRoundUp = (u64_lo >> 1) | (u64_lo & 1);

        __m128 halfCvted = _mm_cvtsi64_ss(_mm_setzero_ps(), halfRoundUp);
        cvted_lo = _mm_add_ps(halfCvted, halfCvted);
    }

    if ((int64_t)u64_hi >= 0) {
        cvted_hi = _mm_cvtsi64_ss(_mm_setzero_ps(), u64_hi);
    } else {
        int64_t halfRoundUp = (u64_hi >> 1) | (u64_hi & 1);
        
        __m128 halfCvted = _mm_cvtsi64_ss(_mm_setzero_ps(), halfRoundUp);
        cvted_hi = _mm_add_ps(halfCvted, halfCvted);
    }

    return _mm_unpacklo_ps(cvted_lo, cvted_hi);
}

NOINLINE __m128 doubleRound_u64ToF32(__m128i toCvt) {
    // Works by doing a cheapish `u64 => f64` conversion, then hardware supported `f64 => f32`
    // However, double rounding isn't exact so we check for that case and correct

    const __m128i MANT_TRUNC_MASK = _mm_set1_epi64x((1 << (DBL_MANT_DIG - FLT_MANT_DIG)) - 1);
    const __m128i DBL_TIED_MANT =   _mm_set1_epi64x(1 << (DBL_MANT_DIG - FLT_MANT_DIG - 1));

    const __m128i DBL_MANT_MASK =_mm_set1_epi64x((1ull << (DBL_MANT_DIG-1)) - 1);
    const __m128i FLT_MANTDIG_IN_DBLEXPO = _mm_set1_epi64x((uint64_t)FLT_MANT_DIG << (DBL_MANT_DIG-1));

    //return _mm_cvtpd_ps(_convert_u64x2_f64x2(toCvt));

    // This specific conversion is cheap
    __m128i dblCvtedBits = _mm_castpd_si128(_convert_u64x2_f64x2(toCvt));

    // Cheaper to duplicate lower32 (only need to check those)
    __m128i dblToFltConvertTies = _mm_shuffle_epi32(
        _mm_cmpeq_epi32(_mm_and_si128(dblCvtedBits, MANT_TRUNC_MASK), DBL_TIED_MANT),
        _MM_SHUFFLE(0,0, 2,2)
    );

    __m128d powOf2BelowFltMant = _mm_castsi128_pd(_mm_andnot_si128(DBL_MANT_MASK, _mm_sub_epi64(dblCvtedBits, FLT_MANTDIG_IN_DBLEXPO)));
    __m128i bitPosThatRoundedToTie = _convert_f64x2_i64x2(powOf2BelowFltMant); // Signed doesn't matter as >= 2^(63-24)

    __m128i roundedDownOrEqual = _cmpEq_i64x2(_mm_and_si128(toCvt, bitPosThatRoundedToTie), bitPosThatRoundedToTie);

    __m128i bitsBelowBitPos = _mm_sub_epi64(bitPosThatRoundedToTie, _mm_set1_epi64x(1)); // x-1
    __m128i isExactlyTied = _cmpEq_i64x2(_mm_and_si128(toCvt, bitsBelowBitPos), _mm_setzero_si128());

    // Rounded up to tie: -1ULP, Rounded down to tie: +1ULP (nothing if at tie without rounding)
    dblCvtedBits = _mm_add_epi64(dblCvtedBits, _mm_andnot_si128(roundedDownOrEqual, dblToFltConvertTies));
    dblCvtedBits = _mm_sub_epi64(dblCvtedBits, _mm_andnot_si128(isExactlyTied, _mm_and_si128(dblToFltConvertTies, roundedDownOrEqual) ));

    return _mm_cvtpd_ps(_mm_castsi128_pd(dblCvtedBits));
}

NOINLINE __m128 thirdFusedSum_u64ToF32(__m128i u64toCvt) {
    // No SIMD `u64 => float` convertion, so convert a third at a time then sum those together

    const __m128i MANT21_MASK = _mm_set1_epi64x((1<<21) - 1);
    const __m128i MANT_MASK_NOMSB = _mm_set1_epi32((1 << (FLT_MANT_DIG - 2)) - 1);
    const __m128i FLT_SIGN_MASK = _mm_set1_epi32(1 << 31); 

    const __m128 MID_OFFSET_TERM = _mm_set1_ps((float)(1ull << 21));
    const __m128 HI_OFFSET_TERM =  _mm_set1_ps((float)(1ull << 42));


    __m128 low = _mm_cvtepi32_ps(_mm_and_si128(u64toCvt, MANT21_MASK));
    __m128 mid = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi64(u64toCvt, 21), MANT21_MASK));
    __m128 hi  = _mm_cvtepi32_ps(_mm_srli_epi64(u64toCvt, 21*2));

    mid = _mm_mul_ps(mid, MID_OFFSET_TERM);
    hi  = _mm_mul_ps(hi, HI_OFFSET_TERM);

    // Adding all three values directly fails due to intermediate rounding
    // so perform a fused-add-add using method from:
    //      `Graillat & Muller (2025): hal-04575249v2`

    // Fast2Sum (values differ in magnitude)
    __m128 sum_lowMid = _mm_add_ps(low, mid);
    __m128 error_lowMid = _mm_sub_ps(low, _mm_sub_ps(sum_lowMid, mid));

    __m128 sum_total = _mm_add_ps(sum_lowMid, hi);
    __m128 error_lm_hi = _mm_sub_ps(sum_lowMid, _mm_sub_ps(sum_total, hi));

    // `error_lm_h` should always be larger or zero
    __m128 error_total = _mm_add_ps(error_lowMid, error_lm_hi);
    __m128 error_final = _mm_sub_ps(error_lowMid, _mm_sub_ps(error_total, error_lm_hi));


    // `1 * 2**n`: zero mantissa, `3 * 2**n`: only MSB set
    __m128i is1Or3XpowOf2 = _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(error_total), MANT_MASK_NOMSB), _mm_setzero_si128());
    __m128 errorTotalNotExact = _mm_cmpneq_ps(error_final, _mm_setzero_ps());

    __m128 needsCorrection = _mm_and_ps(_mm_castsi128_ps(is1Or3XpowOf2), errorTotalNotExact);

    __m128i signsDifferMSB = _mm_and_si128(_mm_castps_si128(_mm_xor_ps(error_final, error_total)), FLT_SIGN_MASK);


    // If a correction is needed, we have to scale by a term
    // NoCorrection: `8/8` (no scaling)
    __m128 correctionTerm = _mm_set1_ps(8.0f / 8.0f);

    // SignsDiffers: `7/8`, SignsSame: `9/8`
    __m128 correctionTermOffset = _mm_and_ps(_mm_set1_ps(1.0f / 8.0f), needsCorrection);

    // Setting MSB subtracts rather then add
    correctionTerm = _mm_add_ps(correctionTerm, _mm_or_ps(correctionTermOffset, _mm_castsi128_ps(signsDifferMSB)));


    __m128 converted = _mm_add_ps(sum_total, _mm_mul_ps(correctionTerm, error_total));

    return _mm_shuffle_ps(converted, _mm_setzero_ps(), _MM_SHUFFLE(0,0, 2,0));
}

NOINLINE __m128 thirdFusedSumA_u64ToF32(__m128i u64toCvt) {
    // No SIMD `u64 => float` convertion, so convert a third at a time then sum those together

    const __m128i MANT21_MASK = _mm_set1_epi64x((1<<21) - 1);
    const __m128i DBL_NONTRUNC_MASK = _mm_set1_epi64x(~((1ull << (DBL_MANT_DIG - FLT_MANT_DIG)) - 1));
    const __m128i DBL_ODDCVT_MANT = _mm_set1_epi64x(1ull << (DBL_MANT_DIG - FLT_MANT_DIG));

    const __m128 MID_OFFSET_TERM = _mm_set1_ps((float)(1ull << 21));
    const __m128 HI_OFFSET_TERM =  _mm_set1_ps((float)(1ull << 42));


    __m128 low = _mm_cvtepi32_ps(_mm_and_si128(u64toCvt, MANT21_MASK));
    __m128 mid = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi64(u64toCvt, 21), MANT21_MASK));
    __m128 hi  = _mm_cvtepi32_ps(_mm_srli_epi64(u64toCvt, 21*2));

    mid = _mm_mul_ps(mid, MID_OFFSET_TERM);
    hi  = _mm_mul_ps(hi, HI_OFFSET_TERM);

    low = _mm_shuffle_ps(low, _mm_setzero_ps(), _MM_SHUFFLE(0,0, 2,0));
    mid = _mm_shuffle_ps(mid, _mm_setzero_ps(), _MM_SHUFFLE(0,0, 2,0));
    hi  = _mm_shuffle_ps(hi,  _mm_setzero_ps(), _MM_SHUFFLE(0,0, 2,0));

    // Adding all three values directly fails due to intermediate rounding
    // so perform a fused-add-add using method from:
    //      `Boldo & Melquiond (2009)`

    // Fast2Sum (values differ in magnitude)
    __m128 sum_lowMid = _mm_add_ps(low, mid);
    __m128 error_lowMid = _mm_sub_ps(low, _mm_sub_ps(sum_lowMid, mid));

    __m128 sum_total = _mm_add_ps(sum_lowMid, hi);
    __m128 error_lm_hi = _mm_sub_ps(sum_lowMid, _mm_sub_ps(sum_total, hi));


    __m128d dblErrorTotal = _mm_add_pd(_mm_cvtps_pd(error_lm_hi), _mm_cvtps_pd(error_lowMid));
    __m128d dblTotalRoundDown = _mm_and_pd(dblErrorTotal, _mm_castsi128_pd(DBL_NONTRUNC_MASK));

    __m128d addIsntExact = _mm_cmpneq_pd(dblErrorTotal, dblTotalRoundDown);
    dblTotalRoundDown = _mm_or_pd(dblTotalRoundDown, _mm_and_pd(addIsntExact, _mm_castsi128_pd(DBL_ODDCVT_MANT)));

    return _mm_add_ps(sum_total, _mm_cvtpd_ps(dblTotalRoundDown));
}

NOINLINE __m128 dblHalf2Sum_u64ToF32(__m128i u64toCvt) {

    const __m128d MANT_AS_LOINT = _mm_set1_pd((double)(1ull << (DBL_MANT_DIG-1)));
    const __m128d MANT_AS_HIINT = _mm_set1_pd(0x1p84); // MANT_BITS + 32

    const __m128i EXPO_INTERLEAVE = _shuffleLoHi_i32x4(
        _mm_castpd_si128(MANT_AS_LOINT), _MM_SHUFHALF(3, 1), _mm_castpd_si128(MANT_AS_HIINT), _MM_SHUFHALF(3, 1)
    );

    __m128i lohi_interleave = _mm_shuffle_epi32(u64toCvt, _MM_SHUFFLE(3,1, 2,0));

    __m128d lo_magicExpo = _mm_castsi128_pd(_mm_unpacklo_epi32(lohi_interleave, EXPO_INTERLEAVE));
    __m128d hi_magicExpo = _mm_castsi128_pd(_mm_unpackhi_epi32(lohi_interleave, EXPO_INTERLEAVE));

    __m128d loCvted = _mm_sub_pd(lo_magicExpo, MANT_AS_LOINT);
    __m128d hiCvted = _mm_sub_pd(hi_magicExpo, MANT_AS_HIINT);

    // Fast2Sum
    __m128d sum_loHi = _mm_add_pd(loCvted, hiCvted);
    __m128d error_loHi = _mm_sub_pd(loCvted, _mm_sub_pd(sum_loHi, hiCvted));


    // If the sum rounded to a value that would tie when converting,
    // then a double-rounding error would occur

    // Needlessly offsetting is only incorrect when it offsets to a tie
    __m128i zeroOrNaNIfLSB = _mm_srai_epi32(_mm_slli_epi64(_mm_castpd_si128(sum_loHi), 63) , 31);

    // NaN is always false, else compare with zero
    __m128i hasNegError = _mm_castpd_si128(_mm_cmplt_pd(error_loHi, _mm_castsi128_pd(zeroOrNaNIfLSB)));
    __m128i hasPosError = _mm_castpd_si128(_mm_cmpgt_pd(error_loHi, _mm_castsi128_pd(zeroOrNaNIfLSB)));

    __m128i sumNoDoubleRound = _mm_castpd_si128(sum_loHi);
    sumNoDoubleRound = _mm_add_epi64(sumNoDoubleRound, hasNegError);
    sumNoDoubleRound = _mm_sub_epi64(sumNoDoubleRound, hasPosError);

    return _mm_cvtpd_ps(_mm_castsi128_pd(sumNoDoubleRound));
}

NOINLINE __m128 dblHalf2SumA_u64ToF32(__m128i u64toCvt) {

    const __m128d MANT_AS_LOINT = _mm_set1_pd((double)(1ull << (DBL_MANT_DIG-1)));
    const __m128d MANT_AS_HIINT = _mm_set1_pd(0x1p84); // MANT_BITS + 32

    const __m128i EXPO_INTERLEAVE = _shuffleLoHi_i32x4(
        _mm_castpd_si128(MANT_AS_LOINT), _MM_SHUFHALF(3, 1), _mm_castpd_si128(MANT_AS_HIINT), _MM_SHUFHALF(3, 1)
    );

    __m128i lohi_interleave = _mm_shuffle_epi32(u64toCvt, _MM_SHUFFLE(3,1, 2,0));

    __m128d lo_magicExpo = _mm_castsi128_pd(_mm_unpacklo_epi32(lohi_interleave, EXPO_INTERLEAVE));
    __m128d hi_magicExpo = _mm_castsi128_pd(_mm_unpackhi_epi32(lohi_interleave, EXPO_INTERLEAVE));

    __m128d loCvted = _mm_sub_pd(lo_magicExpo, MANT_AS_LOINT);
    __m128d hiCvted = _mm_sub_pd(hi_magicExpo, MANT_AS_HIINT);

    // Fast2Sum
    __m128d sum_loHi = _mm_add_pd(loCvted, hiCvted);
    __m128d error_loHi = _mm_sub_pd(loCvted, _mm_sub_pd(sum_loHi, hiCvted));


    // If the sum rounded to a value that would tie when converting,
    // then a double-rounding error would occur

    __m128i errorMoreThanZeroMask = _mm_castpd_si128(_mm_cmpgt_pd(error_loHi, _mm_setzero_pd()));

    // PosError: -1 + sign(+), NegError: 0 + sign(-), NoError: 0 + sign(+)
    __m128i sumNegULPOffset = _mm_add_epi64(
        errorMoreThanZeroMask, _mm_srli_epi64(_mm_castpd_si128(error_loHi), 63)
    );

    // Faster then only offsetting if it is already tied
    __m128i offsetCantCauseTie = _mm_shuffle_epi32(
        _mm_cmpeq_epi32( _mm_slli_epi32(_mm_castpd_si128(sum_loHi), 31), _mm_setzero_si128() ),
        _MM_SHUFFLE(2,2, 0,0)
    );

    __m128d sumNoDoubleRound = _mm_castsi128_pd(
        _mm_sub_epi64(_mm_castpd_si128(sum_loHi), _mm_and_si128(sumNegULPOffset, offsetCantCauseTie))
    );

    return _mm_cvtpd_ps(sumNoDoubleRound);
}


NOINLINE __m128 viaF80_u64ToF32(__m128i u64toCvt) {
    static const float CORRECTION_LO[] = {0, 0x1p64f, 0, 0x1p64f};
    static const float CORRECTION_HI[] = {0, 0, 0x1p64f, 0x1p64f};

    int needsScale = _getMsb_i64x2(u64toCvt);

    uint64_t low, high;
    _mm_storeu_si64(&low, u64toCvt);
    _mm_storeh_pi((__m64*)&high, _mm_castsi128_ps(u64toCvt));

    union {float val; uint32_t asInt;} cvted[2];
    cvted[0].val = (long double)(int64_t)low +  CORRECTION_LO[needsScale];
    cvted[1].val = (long double)(int64_t)high + CORRECTION_HI[needsScale];

    return _mm_castsi128_ps(_mm_cvtsi64_si128(
        ((uint64_t)cvted[1].asInt << 32) | cvted[0].asInt
    ));
}

NOINLINE __m128 compiler_u64ToF32(__m128i u64_toCvt) {
    uint64_t u64_array[2];
    _mm_store_si128((__m128i*)u64_array, u64_toCvt);

    return _mm_setr_ps((float)u64_array[0], (float)u64_array[1], 0, 0);
}

// ===== Unsigned 64-bit int to float64 =====

NOINLINE __m128d magicExpo_u64ToF64(__m128i u64_toCvt) {
    /*
    magicLow[0:63] =  u64[0:31] | double(2**BITS_IN_MANT)

    magicHigh[0:63] = u64[32:63] | double(2**(BITS_IN_MANT+32))

    tmpLow[0:63] =  doubleSub(magicLow,  2**BITS_IN_MANT)
    tmpHigh[0:63] = doubleSub(magicHigh, 2**(BITS_IN_MANT+32))

    return doubleAdd(tmpLow, tmpHigh)
    */

    // Magic method based on clang (when converting a uint64 => double)
    const uint32_t MAGIC_EXP_LO = DBL_MAX_EXP + DBL_MANT_DIG - 2;
    const uint32_t MAGIC_EXP_HI = DBL_MAX_EXP + DBL_MANT_DIG+32 - 2;

    const __m128i MAGIC_EXP = _mm_setr_epi32(
        MAGIC_EXP_LO << (DBL_MANT_DIG-33), MAGIC_EXP_LO << (DBL_MANT_DIG-33),
        MAGIC_EXP_HI << (DBL_MANT_DIG-33), MAGIC_EXP_HI << (DBL_MANT_DIG-33)
    );

    // This prevents GCC from producing two 64-bit load + broadcast
    // Using a memory location during the subtraction is faster
    SSE_LITERAL_UNION_DBL MAGIC_TERM_LO = {{1ull << (DBL_MANT_DIG-1), 1ull << (DBL_MANT_DIG-1)}};
    SSE_LITERAL_UNION_DBL MAGIC_TERM_HI = {{1.9342813113834067e+25, 1.9342813113834067e+25}}; // 2**(DBL_MANT_DIG-1+32);

    // [inp1.hi, inp0.hi, inp1.lo, inp0.lo]
    __m128i inp_hihi_lolo = _mm_shuffle_epi32(u64_toCvt, 0b11011000);

    // [MAGIC, inp1.lo, MAGIC, inp0.lo]
    __m128d magicFlt_lo = _mm_castsi128_pd(_mm_unpacklo_epi32(inp_hihi_lolo, MAGIC_EXP));
    __m128d magicFlt_hi = _mm_castsi128_pd(_mm_unpackhi_epi32(inp_hihi_lolo, MAGIC_EXP));

    // Promotes using a memory argument rather than a 128-bit to register
    magicFlt_lo = _mm_sub_pd(magicFlt_lo, *(__m128d*)&MAGIC_TERM_LO.vec);
    magicFlt_hi = _mm_sub_pd(magicFlt_hi, *(__m128d*)&MAGIC_TERM_HI.vec);

    return _mm_add_pd(magicFlt_hi, magicFlt_lo);
}


NOINLINE __m128d compiler_u64ToF64(__m128i u64_toCvt) {
    uint64_t u64_array[2];
    _mm_store_si128((__m128i*)u64_array, u64_toCvt);

    return _mm_setr_pd((double)u64_array[0], (double)u64_array[1]);
}

// ===== Signed 64-bit int to float64 =====

__m128d scalarInstruction_i64ToF64(__m128i i64_toCvt) {
    // Statement ordering used to manipulate asm output
    __m128i hiInLow = _mm_shuffle_epi32(i64_toCvt, _MM_SHUFFLE(0, 0, 3, 2));
    
    int64_t i64_lo = _mm_cvtsi128_si64(i64_toCvt);
    __m128d dbl_lo = _mm_cvtsi64_sd(_mm_setzero_pd(), i64_lo);
    
    int64_t i64_hi = _mm_cvtsi128_si64(hiInLow);
    __m128d dbl_hi = _mm_cvtsi64_sd(_mm_setzero_pd(), i64_hi);

    return _mm_unpacklo_pd(dbl_lo, dbl_hi);
}

NOINLINE __m128d compiler_i64ToF64(__m128i i64_toCvt) {
    int64_t i64_array[2];
    _mm_store_si128((__m128i*)i64_array, i64_toCvt);

    return _mm_setr_pd((double)i64_array[0], (double)i64_array[1]);
}
