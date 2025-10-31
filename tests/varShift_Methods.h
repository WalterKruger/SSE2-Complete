#pragma once

#include <emmintrin.h> // SSE2
#include <float.h>
#include <stdint.h>

#include "_perfCommon.h"
#include "shuffle_Methods.h"
#include "../include/sseComplete.h"





#define dAsI128(x) _mm_castpd_si128(x)
#define iAsPd(x) _mm_castsi128_pd(x)


#define DUPLICATE_3 CASE_Ni(0) CASE_Ni(1) CASE_Ni(2)
#define DUPLICATE_4 DUPLICATE_3 CASE_Ni(3)
#define DUPLICATE_5 DUPLICATE_4 CASE_Ni(4)


// ========================
// ===== 64-bit right =====
// ========================


NOINLINE __m128i scalarR_64(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi64x(0b111111));

    uint64_t toShift_s[2], amount_s[2], result[2];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 2; i++) {
        result[i] = toShift_s[i] >> (amount_s[i]);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i clangR_64(__m128i toShift, __m128i amount) {
    const __m128i AMOUNT_MASK = _mm_set1_epi64x(64-1);
    
    amount = _mm_and_si128(amount, AMOUNT_MASK);

    __m128i shiftedLo = _mm_srl_epi64(toShift, amount);

    __m128i amountHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));
    __m128i shiftedHi = _mm_srl_epi64(toShift, amountHi);

    //return _shuffleLoHi_i32x4(shiftedLo, _MM_SHUFHALF(1,0), shiftedHi, _MM_SHUFHALF(3,2));
    return _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shiftedHi), _mm_castsi128_pd(shiftedLo)
    ));
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


// =======================
// ===== 8-bit right =====
// =======================


NOINLINE __m128i scalarR_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    uint8_t toShift_s[16], amount_s[16], result[16];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 16; i++) {
        result[i] = toShift_s[i] >> (amount_s[i] % 8);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i directBitByBitR_8(__m128i toShift, __m128i amount) {

    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);

    toShift = _either_i128(_shiftR_u8x16(toShift,1<<2), toShift, _fillWithMSB_i8x16(amount));
    amount = _mm_add_epi8(amount,amount);

    toShift = _either_i128(_shiftR_u8x16(toShift,1<<1), toShift, _fillWithMSB_i8x16(amount));
    amount = _mm_add_epi8(amount,amount);

    #if 0
    toShift = _either_i128(_shiftR_u8x16(toShift,1<<0), toShift, _fillWithMSB_i8x16(amount));
    #else
    // avg(a,b): (a + b + 1) >> 1
    __m128i ceilDiv2 = _mm_avg_epu8(toShift, _mm_setzero_si128());
    // x - ceil(x/2) = x>>1
    toShift = _mm_sub_epi8(toShift, _mm_and_si128(ceilDiv2, _fillWithMSB_i8x16(amount)));
    #endif

    return toShift;
}


NOINLINE __m128i directBitByBitAR_8(__m128i toShift, __m128i amount) {

    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);
    __m128i shiftBy2_2ndMSB = _mm_and_si128(amount, _mm_set1_epi8(1<<6));

    toShift = _either_i128(_shiftR_u8x16(toShift,1<<2), toShift, _fillWithMSB_i8x16(amount));
    amount = _mm_add_epi8(amount,amount);

    // avg(a,b): (a + b + 1) >> 1
    // x - ceil(x/2) = x>>1
    __m128i shiftBy2 = _fillWithMSB_i8x16(amount);
    #if 0
    toShift = _mm_sub_epi8(toShift, _mm_and_si128(shiftBy2, _mm_avg_epu8(toShift, _mm_setzero_si128())));
    toShift = _mm_sub_epi8(toShift, _mm_and_si128(shiftBy2, _mm_avg_epu8(toShift, _mm_setzero_si128())));
    #else
    // (shiftBy2)? (x + 0x100) >> 1 : ((x << 1) + 1) >> 1 
    __m128i maxOrToShift = _mm_or_si128(shiftBy2, toShift);
    toShift = _mm_avg_epu8( _mm_avg_epu8(toShift, maxOrToShift), maxOrToShift);
    toShift = _mm_add_epi8(toShift, shiftBy2_2ndMSB); // Shifting added two MSBs

    #endif
    amount = _mm_add_epi8(amount,amount);

    __m128i ceilDiv2 = _mm_avg_epu8(toShift, _mm_setzero_si128());
    toShift = _mm_sub_epi8(toShift, _mm_and_si128(ceilDiv2, _fillWithMSB_i8x16(amount)));

    return toShift;
}

NOINLINE __m128i repeatedHalfR_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    __m128i partialShift = toShift;
    for (size_t i=0; i < 7; i++) {
        __m128i finishedShifting = _mm_cmpeq_epi8(amount, _mm_setzero_si128());

        // x - ceil(x/2) = x>>1
        __m128i ceilDiv2 = _mm_avg_epu8(partialShift, _mm_setzero_si128());
        partialShift = _mm_sub_epi8(partialShift, _mm_andnot_si128(finishedShifting, ceilDiv2));

        amount = _mm_subs_epu8(amount, _mm_set1_epi8(1));
    }

    return partialShift;
}

NOINLINE __m128i repeatedHalfAR_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    __m128i partialShift = toShift;
    __m128i truncatedLength = _mm_set1_epi8(1);

    for (size_t i=0; i < 7; i++) {
        // avg(a,b) = (a + b + 1) >> 1      avg(a,0xff) = (a + 0x100) >> 1
        __m128i needsShift = _mm_cmpgt_epi8(amount, _mm_setzero_si128());

        partialShift = _mm_avg_epu8(partialShift, _mm_andnot_si128(needsShift, partialShift));
        truncatedLength = _mm_add_epi8(truncatedLength, _mm_and_si128(needsShift, truncatedLength));

        amount = _mm_add_epi8(amount, _setone_i128());
    }

    // If any bits where truncated, the resulted was rounded up!
    __m128i truncatedBits = _mm_and_si128(toShift, _mm_add_epi8(truncatedLength, _setone_i128()));
    return _mm_add_epi8(partialShift, _mm_cmpgt_epi8(truncatedBits, _mm_setzero_si128()));
}



NOINLINE __m128i repeatedHalfArithmeticShift_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    __m128i partialShift = toShift;
    __m128i maskToRemMSB = _mm_or_si128(toShift, _mm_set1_epi8(0b01111111));

    for (size_t i=0; i < 7; i++) {
        // avg(a,b) = (a + b + 1) >> 1      avg(a,0xff) = (a + 0x100) >> 1
        __m128i needsShift = _mm_cmpgt_epi8(amount, _mm_setzero_si128());
        partialShift = _mm_avg_epu8(partialShift, _mm_or_si128(needsShift, partialShift));

        // Avg sets the MSB. Only keep if input was set
        partialShift = _mm_and_si128(partialShift, maskToRemMSB);

        amount = _mm_add_epi8(amount, _setone_i128());
    }

    // Each averaging shifts in a set bit
    return partialShift;
}




// =======================
// ===== 64-bit left =====
// =======================


NOINLINE __m128i scalar_64(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi64x(0b111111));

    uint64_t toShift_s[2], amount_s[2], result[2];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 2; i++) {
        result[i] = toShift_s[i] << (amount_s[i]);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i clang_64(__m128i toShift, __m128i amount) {
    #if 0
    static volatile const union {uint64_t qword[2]; __m128i vec;} SHIFT_MASK = {{0b111111, 0b111111}};
    amount = _mm_and_si128(amount, *(__m128i*)&SHIFT_MASK);
    #else
    amount = _mm_and_si128(amount, _mm_set1_epi64x(0b111111));
    #endif

    __m128i shiftedLo = _mm_sll_epi64(toShift, amount);

    __m128i amountHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));
    __m128i shiftedHi = _mm_sll_epi64(toShift, amountHi);

    //return _shuffleLoHi_i32x4(shiftedLo, _MM_SHUFHALF(1,0), shiftedHi, _MM_SHUFHALF(3,2));
    return dAsI128(_mm_move_sd(iAsPd(shiftedHi), iAsPd(shiftedLo)));
}

NOINLINE __m128i directBitByBit_64(__m128i toShift, __m128i amount) {

    amount = _mm_slli_epi64(amount, 64-6);

    #define CASE_Ni(i)\
    toShift = _either_i128(_mm_slli_epi64(toShift,1<<(5-(i))), toShift, _fillWithMSB_i64x2(amount));\
    amount = _mm_add_epi64(amount,amount);
    
    DUPLICATE_5
    #undef CASE_Ni

    // Doubling is cheaper
    toShift = _mm_add_epi64(toShift, _mm_and_si128(toShift, _fillWithMSB_i64x2(amount)));
    
    return toShift;
}


NOINLINE __m128i powOf2BitByBit_64(__m128i toShift, __m128i amount) {

    __m128i powOf2 = _mm_set1_epi64x(1);

    #define ISBITSET_64(x, i) _fillWithMSB_i64x2(_mm_slli_epi64(x, 63 - (i)))

    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<0), powOf2, ISBITSET_64(amount, 0));
    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<1), powOf2, ISBITSET_64(amount, 1));
    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<2), powOf2, ISBITSET_64(amount, 2));
    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<3), powOf2, ISBITSET_64(amount, 3));
    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<4), powOf2, ISBITSET_64(amount, 4));
    powOf2 = _either_i128(_mm_slli_epi64(powOf2,1<<5), powOf2, ISBITSET_64(amount, 5));

    return _mulLo_u64x2(toShift, powOf2);
}

NOINLINE __m128i powOf2Float_64(__m128i toShift, __m128i amount) {
    const __m128d ONE_FLT = _mm_set1_pd(1.0f);

    __m128i powOf2Offset = _mm_srli_epi64( _mm_slli_epi64(amount, 64-6), 64-6 - (DBL_MANT_DIG-1) );

    __m128i powOf2Flt = _mm_add_epi64(powOf2Offset, _mm_castpd_si128(ONE_FLT));
    // Signed convertion is faster, and overflow error value is the same as the true conversion
    __m128i powOf2Int = _convert_f64x2_i64x2(_mm_castsi128_pd(powOf2Flt));

    return _mulLo_u64x2(toShift, powOf2Int);
}


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

    #if 0
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

    // Clears upper bits and shifts into place so that zero padding places it into the exponent 
    __m128i expoPackedHi16 = _mm_srli_epi16( _mm_slli_epi16(amount, 16 - 4), 32-4 - (FLT_MANT_DIG-1) );

    __m128i expoOffset_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoPackedHi16);
    __m128i expoOffset_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoPackedHi16);

    __m128 toShuffFlt_lo = _convertLo_u16x8_f32x4(toShift);
    __m128 toShuffFlt_hi = _convertHi_u16x8_f32x4(toShift);

    // When toShift = 0, value is too small to change things
    __m128i shiftedFlt_lo = _mm_add_epi32(_mm_castps_si128(toShuffFlt_lo), expoOffset_lo);
    __m128i shiftedFlt_hi = _mm_add_epi32(_mm_castps_si128(toShuffFlt_hi), expoOffset_hi);

    __m128i shiftedInt_lo = _mm_cvttps_epi32(_mm_castsi128_ps(shiftedFlt_lo));
    __m128i shiftedInt_hi = _mm_cvttps_epi32(_mm_castsi128_ps(shiftedFlt_hi));
    
    // Otherwise saturates when powOf2 = 1 << 15
    return _trunc_u32x4_u16x8(shiftedInt_lo, shiftedInt_hi);
}


// ======================
// ===== 8-bit left =====
// ======================


NOINLINE __m128i scalar_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    uint8_t toShift_s[16], amount_s[16], result[16];
    _mm_storeu_si128((__m128i*)toShift_s, toShift);
    _mm_storeu_si128((__m128i*)amount_s, amount);

    for (size_t i=0; i < 16; i++) {
        result[i] = toShift_s[i] << (amount_s[i] % 8);
    }
    
    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i repeatedDouble_8(__m128i toShift, __m128i amount) {

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    __m128i partialShift = toShift;
    for (size_t i=0; i < 7; i++) {
        __m128i noMoreShifts = _mm_cmpeq_epi8(amount, _mm_setzero_si128());
        partialShift = _mm_add_epi8(partialShift, _mm_andnot_si128(noMoreShifts, partialShift));

        amount = _mm_subs_epu8(amount, _mm_set1_epi8(1));
    }

    return partialShift;
}


NOINLINE __m128i powOf2BitByBit_8(__m128i toShift, __m128i amount) {

    __m128i powOf2 = _mm_set1_epi8(1);

    // Shift size doesn't matter as no set bits cross the lane
    #define ISBITSET_8(x, i) _mm_cmplt_epi8( _mm_slli_epi16(x, 7 - (i)), _mm_setzero_si128() )

    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<0), powOf2, ISBITSET_8(amount, 0));
    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<1), powOf2, ISBITSET_8(amount, 1));
    powOf2 = _either_i128(_mm_slli_epi16(powOf2,1<<2), powOf2, ISBITSET_8(amount, 2));

    return _mulLo_u8x16(toShift, powOf2);
}

NOINLINE __m128i directBitByBit_8(__m128i toShift, __m128i amount) {


    #define MSB_SET8(x) _mm_cmplt_epi8(x, _mm_setzero_si128() )

    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);

    toShift = _either_i128(_shiftL_u8x16(toShift,1<<2), toShift, MSB_SET8(amount));
    amount = _mm_add_epi8(amount,amount);

    toShift = _either_i128(_shiftL_u8x16(toShift,1<<1), toShift, MSB_SET8(amount));
    amount = _mm_add_epi8(amount,amount);

    // Doubling avoids having to call `_either`
    toShift = _mm_add_epi8(toShift, _mm_and_si128(toShift, MSB_SET8(amount)));
    

    return toShift;
}

NOINLINE __m128i directBitByBitA_8(__m128i toShift, __m128i amount) {

    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);

    toShift = _either_i128(_shiftL_u8x16(toShift,1<<2), toShift, MSB_SET8(amount));
    amount = _mm_add_epi8(amount,amount);

    __m128i shiftBy2 = MSB_SET8(amount);
    toShift = _mm_add_epi8(toShift, _mm_and_si128(toShift, shiftBy2));
    toShift = _mm_add_epi8(toShift, _mm_and_si128(toShift, shiftBy2));
    amount = _mm_add_epi8(amount,amount);

    // Doubling avoids having to call `_either`
    toShift = _mm_add_epi8(toShift, _mm_and_si128(toShift, MSB_SET8(amount)));
    

    return toShift;
}



NOINLINE __m128i powOf2Float_8(__m128i toShift, __m128i amount) {
    const __m128 EXP2_0_FLT = _mm_set1_ps(1.0f);

    amount = _mm_and_si128(amount, _mm_set1_epi8(0b111));

    __m128i expoOffset_lo = _mm_unpacklo_epi8(_mm_setzero_si128(), amount);
    __m128i expoOffset_hi = _mm_unpackhi_epi8(_mm_setzero_si128(), amount);

    // Zero padding acts like a shift
    expoOffset_lo = _mm_srli_epi16(expoOffset_lo, 8+16 - (FLT_MANT_DIG-1));
    expoOffset_hi = _mm_srli_epi16(expoOffset_hi, 8+16 - (FLT_MANT_DIG-1));

    __m128i expoOffset_lolo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoOffset_lo);
    __m128i expoOffset_lohi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoOffset_lo);
    __m128i expoOffset_hilo = _mm_unpacklo_epi16(_mm_setzero_si128(), expoOffset_hi);
    __m128i expoOffset_hihi = _mm_unpackhi_epi16(_mm_setzero_si128(), expoOffset_hi);

    
    __m128i powOf2Flt_lolo = _mm_add_epi32(expoOffset_lolo, _mm_castps_si128(EXP2_0_FLT));
    __m128i powOf2Flt_lohi = _mm_add_epi32(expoOffset_lohi, _mm_castps_si128(EXP2_0_FLT));
    __m128i powOf2Flt_hilo = _mm_add_epi32(expoOffset_hilo, _mm_castps_si128(EXP2_0_FLT));
    __m128i powOf2Flt_hihi = _mm_add_epi32(expoOffset_hihi, _mm_castps_si128(EXP2_0_FLT));

    __m128i powOf2_lolo = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_lolo));
    __m128i powOf2_lohi = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_lohi));
    __m128i powOf2_hilo = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_hilo));
    __m128i powOf2_hihi = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt_hihi));

    __m128i powOf2_lo = _mm_packs_epi32(powOf2_lolo, powOf2_lohi);
    __m128i powOf2_hi = _mm_packs_epi32(powOf2_hilo, powOf2_hihi);

    __m128i powOf2 = _mm_packus_epi16(powOf2_lo, powOf2_hi);

    return _mulLo_u8x16(toShift, powOf2);
}
