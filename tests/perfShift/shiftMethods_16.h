#pragma once
#include "../_perfCommon.h"

#include "../../include/sseCom_parts/conversion.h"
#include "../../include/sseCom_parts/arithmetic.h"

#define DUPLICATE_3 CASE_Ni(0) CASE_Ni(1) CASE_Ni(2)



// =======================
// ===== 16-bit left =====
// =======================


NOINLINE __m128i scalar_16(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi16(0b1111));

    uint16_t toShift_s[8], amount_s[8], result[8];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 8; i++) {
        result[i] = toShift_s[i] << (amount_s[i] % 16);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i insertExtractMem_16(__m128i toShift, __m128i amount) {

    __m128i result = _mm_setzero_si128(); // Prevent UD
    amount = _mm_and_si128(amount, _mm_set1_epi16(0b1111));

    // Two instructions per element (insert can use memory operand)
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,0) << _mm_extract_epi16(amount,0), 0);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,1) << _mm_extract_epi16(amount,1), 1);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,2) << _mm_extract_epi16(amount,2), 2);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,3) << _mm_extract_epi16(amount,3), 3);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,4) << _mm_extract_epi16(amount,4), 4);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,5) << _mm_extract_epi16(amount,5), 5);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,6) << _mm_extract_epi16(amount,6), 6);
    result = _mm_insert_epi16(result, _mm_extract_epi16(toShift,7) << _mm_extract_epi16(amount,7), 7);

    return result;
}


NOINLINE __m128i powOf2BitByBit_16(__m128i toShift, __m128i amount) {

    __m128i powOf2 = _mm_set1_epi16(1);

    #define ISBITSET_16(x, i) _mm_srai_epi16( _mm_slli_epi16(x, 15 - (i)), 15 )

    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<0), powOf2, ISBITSET_16(amount, 0));
    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<1), powOf2, ISBITSET_16(amount, 1));
    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<2), powOf2, ISBITSET_16(amount, 2));
    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<3), powOf2, ISBITSET_16(amount, 3));

    return _mm_mullo_epi16(toShift, powOf2);
}

NOINLINE __m128i directBitByBit_16(__m128i toShift, __m128i amount) {

    amount = _mm_slli_epi16(amount, 16-4);

    #define CASE_Ni(i)\
    toShift = _either_i128(_mm_slli_epi16(toShift,1<<(3-(i))), toShift, _mm_srai_epi16(amount, 15));\
    amount = _mm_add_epi16(amount,amount);
    
    DUPLICATE_3
    #undef CASE_Ni

    // Conditional doubling is cheaper
    toShift = _mm_add_epi16(toShift, _mm_and_si128(toShift, _mm_srai_epi16(amount, 15)));
    
    return toShift;
}



NOINLINE __m128i powOf2Float_16(__m128i toShift, __m128i amount) {
    const union {float asVal; uint32_t bitRep;} ONE_FLT = {1.0f};


    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );

    #if 1
    __m128i expoOffset_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoPackedHi16);
    __m128i expoOffset_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoPackedHi16);

    __m128i powOf2Flt_lo = _mm_add_epi32(expoOffset_lo, _mm_set1_epi32(ONE_FLT.bitRep));
    __m128i powOf2Flt_hi = _mm_add_epi32(expoOffset_hi, _mm_set1_epi32(ONE_FLT.bitRep));
    #else

    __m128i expoOffsetHi16 = _mm_add_epi16(expoPackedHi16, _mm_set1_epi16(ONE_FLT.bitRep >> 16));

    __m128i powOf2Flt_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoOffsetHi16);
    __m128i powOf2Flt_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoOffsetHi16);

    #endif

    __m128i powOf2Int_lo = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_lo));
    __m128i powOf2Int_hi = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_hi));
    
    // Otherwise saturates when powOf2 = 1 << 15
    __m128i powOf2Int = _trunc_u32x4_u16x8(powOf2Int_lo, powOf2Int_hi);


    return _mm_mullo_epi16(toShift, powOf2Int);
}

NOINLINE __m128i powOf2FloatA_16(__m128i toShift, __m128i amount) {
    const union {float asVal; uint32_t bitRep;} NEGONE_FLT = {-1.0f};


    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );
    __m128i expoOffsetHi16 = _mm_add_epi16(expoPackedHi16, _mm_set1_epi16(NEGONE_FLT.bitRep >> 16));

    __m128i negPow2Flt_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoOffsetHi16);
    __m128i negPow2Flt_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoOffsetHi16);

    __m128i negPow2Int_lo = _mm_cvttps_epi32(_mm_castsi128_ps(negPow2Flt_lo));
    __m128i negPow2Int_hi = _mm_cvttps_epi32(_mm_castsi128_ps(negPow2Flt_hi));

    // A positive power would have saturated with: 1 << 15
    __m128i negPow2Int_underflow = _mm_packs_epi32(negPow2Int_lo, negPow2Int_hi);

    // `INT_MIN` is equal to desired unsigned power and remains the same when negated
    return _mm_mullo_epi16(_negate_i16x8(toShift), negPow2Int_underflow);
}


NOINLINE __m128i floatScale_16(__m128i toShift, __m128i amount) {
    const union {float asVal; uint32_t bitRep;} NEGONE_FLT = {-1.0f};

    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );
    __m128i expoOffsetHi16 = _mm_add_epi16(expoPackedHi16, _mm_set1_epi16(NEGONE_FLT.bitRep >> 16));

    __m128i negPow2Flt_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoOffsetHi16);
    __m128i negPow2Flt_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoOffsetHi16);

    __m128i negPow2Int_lo = _mm_cvttps_epi32(_mm_castsi128_ps(negPow2Flt_lo));
    __m128i negPow2Int_hi = _mm_cvttps_epi32(_mm_castsi128_ps(negPow2Flt_hi));
    
    // A positive power would have saturated with: 1 << 15
    __m128i negPow2Int = _mm_packs_epi32(negPow2Int_lo, negPow2Int_hi);

    // `INT_MIN` is equal to desired unsigned power and remains the same when negated
    return _mm_mullo_epi16(_negate_i16x8(toShift), negPow2Int);
}

_GNUC_ONLY(__attribute__((target("ssse3"))))
NOINLINE __m128i ssse3LUT_16(__m128i toShift, __m128i amount) {
    const __m128i POW2LUT_LO = _mm_cvtsi64_si128(0x8040201008040201ll);
    const __m128i POW2LUT_HI = _mm_bslli_si128(POW2LUT_LO, 8);
    const __m128i LOW_HALF_MASK = _mm_set1_epi16(0x00ff);

    // So amount is modulo 16
    amount = _mm_and_si128(amount, _mm_set1_epi16(0x000f));

    // The LUT value outside of low/high are zero
    __m128i powOf2_lo = _mm_and_si128(_mm_shuffle_epi8(POW2LUT_LO, amount), LOW_HALF_MASK);
    __m128i powOf2_hi = _mm_shuffle_epi8(POW2LUT_HI, _mm_slli_epi16(amount, 8));

    __m128i powOf2_full = _mm_or_si128(powOf2_lo, powOf2_hi);

    return _mm_mullo_epi16(toShift, powOf2_full);
}




// ========================
// ===== 16-bit right =====
// ========================


NOINLINE __m128i scalarR_16(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi16(0b1111));

    uint16_t toShift_s[8], amount_s[8], result[8];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 8; i++) {
        result[i] = toShift_s[i] >> (amount_s[i] % 16);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i floatScaleR_16(__m128i toShift, __m128i amount) {
    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );

    __m128i expoOffset_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoPackedHi16);
    __m128i expoOffset_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoPackedHi16);

    __m128 toShiftFlt_lo = _convertLo_u16x8_f32x4(toShift);
    __m128 toShiftFlt_hi = _convertHi_u16x8_f32x4(toShift);

    // Note: This creates NaN when toShift = 0. That gets converted to `INT32_MAX` which is truncated away
    __m128 shiftedFlt_lo = _mm_castsi128_ps( _mm_sub_epi32(_mm_castps_si128(toShiftFlt_lo), expoOffset_lo) );
    __m128 shiftedFlt_hi = _mm_castsi128_ps( _mm_sub_epi32(_mm_castps_si128(toShiftFlt_hi), expoOffset_hi) );
    
    // Otherwise saturates when powOf2Flt = 1
    return _trunc_u32x4_u16x8(
        _mm_cvttps_epi32(shiftedFlt_lo), _mm_cvttps_epi32(shiftedFlt_hi)
    );
}

NOINLINE __m128i powOf2FloatR_16(__m128i toShift, __m128i amount) {
    const __m128 ONE_FLT = _mm_set1_ps(1.0f);

    // A mulhi can act like a right shift: mulhi(x, 1 << 2) = (x << 2) >> 16 = x >> 14
    // (0 - x) % 16: [0, 1, 2, ..., 15] => [0, 15, 14, ..., 1]
    amount = _mm_sub_epi16(_mm_setzero_si128(), amount);

    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );

    __m128i expoOffset_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoPackedHi16);
    __m128i expoOffset_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoPackedHi16);

    __m128i powOf2Flt_lo = _mm_add_epi32(expoOffset_lo, _mm_castps_si128(ONE_FLT));
    __m128i powOf2Flt_hi = _mm_add_epi32(expoOffset_hi, _mm_castps_si128(ONE_FLT));

    __m128i powOf2Int_lo = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_lo));
    __m128i powOf2Int_hi = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_hi));
    
    // Otherwise saturates when powOf2 = 1 << 15
    __m128i powOf2Int = _trunc_u32x4_u16x8(powOf2Int_lo, powOf2Int_hi);

    // When amount = 0, the powOf2 will be 1 so the product will be entirely in the low part
    // (expoPackedHi16 is zero, when modulo the shift size)
    __m128i shiftZeroCorrection = _mm_and_si128(toShift, _mm_cmpeq_epi16(expoPackedHi16, _mm_setzero_si128()));

    return _mm_or_si128(_mm_mulhi_epu16(toShift, powOf2Int), shiftZeroCorrection);
}


NOINLINE __m128i directBitByBitR_16(__m128i toShift, __m128i amount) {

    amount = _mm_slli_epi16(amount, 16-4);

    #define CASE_Ni(i)\
    toShift = _either_i128(_mm_srli_epi16(toShift,1<<(3-(i))), toShift, _mm_srai_epi16(amount, 15));\
    amount = _mm_add_epi16(amount,amount);
    
    DUPLICATE_3
    #undef CASE_Ni

    #if 0
    toShift = _either_i128(_mm_srli_epi16(toShift,1<<0), toShift, _mm_srai_epi16(amount, 15));
    #else
    // avg(a,b): (a + b + 1) >> 1
    __m128i ceilDiv2 = _mm_avg_epu16(toShift, _mm_setzero_si128());
    // x - ceil(x/2) = x>>1
    toShift = _mm_sub_epi16(toShift, _mm_and_si128(ceilDiv2, _mm_srai_epi16(amount, 15)));
    #endif
    
    return toShift;
}
