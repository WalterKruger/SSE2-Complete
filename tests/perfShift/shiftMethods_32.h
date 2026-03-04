#pragma once
#include "../_perfCommon.h"

#include "../../include/sseCom_parts/conversion.h"
#include "../../include/sseCom_parts/multiply.h"
#include "../../include/sseCom_parts/shuffle.h"

#define DUPLICATE_4 CASE_Ni(0) CASE_Ni(1) CASE_Ni(2) CASE_Ni(3)



// =======================
// ===== 32-bit left =====
// =======================


NOINLINE __m128i scalar_32(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi32(0b11111));

    uint32_t toShift_s[4], amount_s[4], result[4];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 4; i++) {
        result[i] = toShift_s[i] << (amount_s[i]);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}


NOINLINE __m128i directBitByBit_32(__m128i toShift, __m128i amount) {

    amount = _mm_slli_epi32(amount, 32-5);

    #define CASE_Ni(i)\
    toShift = _either_i128(_mm_slli_epi32(toShift,1<<(4-(i))), toShift, _mm_srai_epi32(amount, 31));\
    amount = _mm_add_epi32(amount,amount);
    
    DUPLICATE_4
    #undef CASE_Ni

    toShift = _mm_add_epi32(toShift, _mm_and_si128(toShift, _mm_srai_epi32(amount, 31))); // Doubling is cheaper
    
    return toShift;
}


NOINLINE __m128i powOf2BitByBit_32(__m128i toShift, __m128i amount) {

    __m128i powOf2 = _mm_set1_epi32(1);

    #define ISBITSET_32(x, i) _mm_srai_epi32(_mm_slli_epi32(x, 31-(i)), 31)
    
    powOf2 = _either_i128( _mm_slli_epi32(powOf2,1<<0), powOf2, ISBITSET_32(amount, 0) );
    powOf2 = _either_i128( _mm_slli_epi32(powOf2,1<<1), powOf2, ISBITSET_32(amount, 1) );
    powOf2 = _either_i128( _mm_slli_epi32(powOf2,1<<2), powOf2, ISBITSET_32(amount, 2) );
    powOf2 = _either_i128( _mm_slli_epi32(powOf2,1<<3), powOf2, ISBITSET_32(amount, 3) );
    powOf2 = _either_i128( _mm_slli_epi32(powOf2,1<<4), powOf2, ISBITSET_32(amount, 4) );

    return _mulLo_u32x4(toShift, powOf2);
}

NOINLINE __m128i powOf2Float_32(__m128i toShift, __m128i amount) {
    const __m128 ONE_FLT = _mm_set1_ps(1.0f);

    __m128i powOf2Offset = _mm_srli_epi32( _mm_slli_epi32(amount, 32-5), 32-5 - (FLT_MANT_DIG-1) );

    __m128i powOf2Flt = _mm_add_epi32(powOf2Offset, _mm_castps_si128(ONE_FLT));
    // Can overflow, but that sets it to the correct value anyway
    __m128i powOf2Int = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt));

    return _mulLo_u32x4(toShift, powOf2Int);
}

NOINLINE __m128i shiftSIMD_32(__m128i toShift, __m128i amount) {
    amount = _mm_and_si128(amount, _mm_set1_epi32(31));

    // Since we masked out the lower bits, the odd 16-bit lanes are all zeros
    // (The last `shuffelo_16` index is the one that matters)

    __m128i index_loHi = _mm_shufflelo_epi16(amount, _MM_SHUFFLE(3,3,3,2));
    __m128i shifted_loHi = _mm_sll_epi32(toShift, index_loHi);

    __m128i index_loLo = _mm_shufflelo_epi16(amount, _MM_SHUFFLE(1,1,1,0));
    __m128i shifted_loLo = _mm_sll_epi32(toShift, index_loLo);


    __m128i amountHiHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));

    __m128i index_hiHi = _mm_shufflelo_epi16(amountHiHi, _MM_SHUFFLE(3,3,3,2));
    __m128i shifted_hiHi = _mm_sll_epi32(toShift, index_hiHi);

    __m128i index_hiLo = _mm_shufflelo_epi16(amountHiHi, _MM_SHUFFLE(1,1,1,0));
    __m128i shifted_hiLo = _mm_sll_epi32(toShift, index_hiLo);

    // ##Lo: In first 32 lane, ##Hi: In second 32-bit lane
    __m128i shifted_lo = _mm_unpacklo_epi32(shifted_loLo, shifted_loHi);
    __m128i shifted_hi = _mm_unpacklo_epi32(shifted_hiLo, shifted_hiHi);

    // [##Hi, ???, ???, ##Lo]
    return _shuffleLoHi_i32x4(shifted_lo, _MM_SHUFHALF(3,0), shifted_hi, _MM_SHUFHALF(3,0));
}



// ========================
// ===== 32-bit right =====
// ========================


NOINLINE __m128i scalarR_32(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi16(0b11111));

    uint32_t toShift_s[4], amount_s[4], result[4];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 4; i++) {
        result[i] = toShift_s[i] >> (amount_s[i] % 32);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i floatScaleR_32(__m128i toShift, __m128i amount) {
    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i amountPackedHi32 = _mm_srli_epi32( _mm_slli_epi32(amount, 32 - 5), 64-5 - (DBL_MANT_DIG-1) );

    __m128i expoOffset_lo = _mm_unpacklo_epi32(_mm_setzero_si128(), amountPackedHi32);
    __m128i expoOffset_hi = _mm_unpackhi_epi32(_mm_setzero_si128(), amountPackedHi32);

    __m128d toShiftFlt_lo = _convertLo_u32x4_f64x2(toShift);
    __m128d toShiftFlt_hi = _convertHi_u32x4_f64x2(toShift);

    // Note: This creates NaN when toShift = 0. That gets converted to `INT64_MAX` which is truncated away
    __m128d shiftedFlt_lo = _mm_castsi128_pd( _mm_sub_epi64(_mm_castpd_si128(toShiftFlt_lo), expoOffset_lo) );
    __m128d shiftedFlt_hi = _mm_castsi128_pd( _mm_sub_epi64(_mm_castpd_si128(toShiftFlt_hi), expoOffset_hi) );
    
    #if 0
    // Otherwise saturates when powOf2Flt = 1
    __m128i shiftedLo = _convert_f64x2_u32x4(shiftedFlt_lo);
    __m128i shiftedHi = _convert_f64x2_u32x4(shiftedFlt_hi);

    return _shuffleLoHi_i32x4(shiftedLo, _MM_SHUFHALF(1,0), shiftedHi, _MM_SHUFHALF(1,0));
    #else
    // May overflow or convert NaN
    __m128i shiftedLo = _mm_cvttpd_epi32(shiftedFlt_lo);
    __m128i shiftedHi = _mm_cvttpd_epi32(shiftedFlt_hi);

    // Normal values: [0, SIGNED32_MAX] (MSB=0), or overflow error val: "80000000H" (MSB=1)
    __m128i resultMayOverflow = _shuffleLoHi_i32x4(shiftedLo, _MM_SHUFHALF(1,0), shiftedHi, _MM_SHUFHALF(1,0));
    __m128i hasOverflowed = _mm_srai_epi32(resultMayOverflow, 31);

    // Clear MSB, as NaN gets converted to the error value as well
    resultMayOverflow = _mm_srli_epi32(_mm_add_epi32(resultMayOverflow, resultMayOverflow), 1);

    // Restore correct value when shift by zero
    return _mm_or_si128(resultMayOverflow, _mm_and_si128(toShift, hasOverflowed));
    #endif
}

NOINLINE __m128i shiftSIMDR_32(__m128i toShift, __m128i amount) {
    amount = _mm_and_si128(amount, _mm_set1_epi32(31));

    // Since we masked out the lower bits, the odd 16-bit lanes are all zeros
    // (The last `shuffelo_16` index is the one that matters)

    __m128i index_loHi = _mm_shufflelo_epi16(amount, _MM_SHUFFLE(3,3,3,2));
    __m128i shifted_loHi = _mm_srl_epi32(toShift, index_loHi);

    __m128i index_loLo = _mm_shufflelo_epi16(amount, _MM_SHUFFLE(1,1,1,0));
    __m128i shifted_loLo = _mm_srl_epi32(toShift, index_loLo);


    __m128i amountHiHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));

    __m128i index_hiHi = _mm_shufflelo_epi16(amountHiHi, _MM_SHUFFLE(3,3,3,2));
    __m128i shifted_hiHi = _mm_srl_epi32(toShift, index_hiHi);

    __m128i index_hiLo = _mm_shufflelo_epi16(amountHiHi, _MM_SHUFFLE(1,1,1,0));
    __m128i shifted_hiLo = _mm_srl_epi32(toShift, index_hiLo);

    // ##Lo: In first 32 lane, ##Hi: In second 32-bit lane
    __m128i shifted_lo = _mm_unpacklo_epi32(shifted_loLo, shifted_loHi);
    __m128i shifted_hi = _mm_unpacklo_epi32(shifted_hiLo, shifted_hiHi);

    // [##Hi, ???, ???, ##Lo]
    return _shuffleLoHi_i32x4(shifted_lo, _MM_SHUFHALF(3,0), shifted_hi, _MM_SHUFHALF(3,0));
}

NOINLINE __m128i powOf2FloatR_32(__m128i toShift, __m128i amount) {
    const __m128 ONE_FLT = _mm_set1_ps(1.0f);

    // A mulhi can act like a right shift: mulhi(x, 1 << 2) = (x << 2) >> 32 = x >> 30
    // (0 - x) % 32: [0, 1, 2, ..., 31] => [0, 31, 30, ..., 1]
    amount = _mm_sub_epi32(_mm_setzero_si128(), amount);
    __m128i powOf2Offset = _mm_srli_epi32( _mm_slli_epi32(amount, 32-5), 32-5 - (FLT_MANT_DIG-1) );

    __m128i powOf2Flt = _mm_add_epi32(powOf2Offset, _mm_castps_si128(ONE_FLT));
    // Can overflow, but that sets it to the correct value anyway
    __m128i powOf2Int = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt));

    // When amount = 0, the powOf2 will be 1 so the product will be entirely in the low part
    // (powOf2Offset is zero, when modulo the shift size)
    __m128i shiftZeroCorrection = _mm_and_si128(toShift, _mm_cmpeq_epi32(powOf2Offset, _mm_setzero_si128()));

    return _mm_or_si128(_mulHi_u32x4(toShift, powOf2Int), shiftZeroCorrection);
}
