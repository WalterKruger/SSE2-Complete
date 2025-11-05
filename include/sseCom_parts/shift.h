#pragma once

#include <emmintrin.h>  // SSE2
#include <stdint.h>
#include <float.h>      // Place integer into float exponent

#include "_common.h"
#include "negation.h"
#include "shuffle.h"
#include "conversion.h"
#include "multiply.h"



// ====== 8-bit ======

// Shift 8-bit integers left by immediate while shifting in zeros
#define _shiftL_u8x16(toShift, amount) (_mm_and_si128(\
    _mm_slli_epi32(toShift, amount),\
    _mm_set1_epi8( (uint8_t)(UINT8_MAX << (amount)) )\
))

// Shift 8-bit integers right by immediate while shifting in zeros
#define _shiftR_u8x16(toShift, amount) (_mm_and_si128(\
    _mm_srli_epi32(toShift, amount),\
    _mm_set1_epi8( (uint8_t)(UINT8_MAX >> (amount)) )\
))


// ((x + (1ULL << 7)) >> amount) - ((1ULL << 7) >> amount);

// Shift 8-bit integers right by immediate while shifting in sign bits
#define _signShiftR_i8x16(toShift, amount) (\
    _mm_sub_epi8(\
        _shiftR_u8x16( _mm_add_epi8(toShift, _mm_set1_epi8(1<<7)), amount ), \
        _shiftR_u8x16( _mm_set1_epi8(1<<7), amount)\
    ) )


// Shifts each 8-bit element left by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftLvar_u8x16(__m128i u8ToShift, __m128i amount) {
    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);

    u8ToShift = _either_i128(_shiftL_u8x16(u8ToShift,1<<2), u8ToShift, _fillWithMSB_i8x16(amount));
    amount = _mm_add_epi8(amount,amount);

    // Cheaper to shift left via add twice
    __m128i shiftsBy2 = _fillWithMSB_i8x16(amount);
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, shiftsBy2));
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, shiftsBy2));
    amount = _mm_add_epi8(amount,amount);

    // Doubling avoids having to call `_either`
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, _fillWithMSB_i8x16(amount)));
    
    return u8ToShift;
}

// Shifts each 8-bit element right by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftRvar_u8x16(__m128i toShift, __m128i amount) {

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



// ====== 16-bit ======

// Shifts each 16-bit element left by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftLvar_u16x8(__m128i u16ToShift, __m128i amount) {
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
    return _mm_mullo_epi16(_negate_i16x8(u16ToShift), negPow2Int);
}

// Shifts each 16-bit element right by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftRvar_u16x8(__m128i toShift, __m128i amount) {
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




// ====== 32-bit ======

// Shifts each 32-bit element left by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftLvar_u32x4(__m128i u32ToShift, __m128i amount) {
    const __m128 ONE_FLT = _mm_set1_ps(1.0f);

    __m128i powOf2Offset = _mm_srli_epi32( _mm_slli_epi32(amount, 32-5), 32-5 - (FLT_MANT_DIG-1) );

    __m128i powOf2Flt = _mm_add_epi32(powOf2Offset, _mm_castps_si128(ONE_FLT));
    // Can overflow, but that sets it to the correct value anyway
    __m128i powOf2Int = _mm_cvttps_epi32(_mm_castsi128_ps(powOf2Flt));

    return _mulLo_u32x4(u32ToShift, powOf2Int);
}

// Shifts each 32-bit element right by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftRvar_u32x4(__m128i u32ToShift, __m128i amount) {
    amount = _mm_and_si128(amount, _mm_set1_epi32(31));

    // Since we masked out the lower bits, the odd 16-bit lanes are all zeros
    // (The last `shuffelo_16` index is the one that matters)
    enum {
        ISOLATE_LO16 = _MM_SHUFFLE(1,1,1,0),
        ISOLATE_HI16 = _MM_SHUFFLE(3,3,3,2)
    };

    __m128i index_loHi = _mm_shufflelo_epi16(amount, ISOLATE_HI16);
    __m128i shifted_loHi = _mm_srl_epi32(u32ToShift, index_loHi);

    __m128i index_loLo = _mm_shufflelo_epi16(amount, ISOLATE_LO16);
    __m128i shifted_loLo = _mm_srl_epi32(u32ToShift, index_loLo);


    __m128i amountHiHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));

    __m128i index_hiHi = _mm_shufflelo_epi16(amountHiHi, ISOLATE_HI16);
    __m128i shifted_hiHi = _mm_srl_epi32(u32ToShift, index_hiHi);

    __m128i index_hiLo = _mm_shufflelo_epi16(amountHiHi, ISOLATE_LO16);
    __m128i shifted_hiLo = _mm_srl_epi32(u32ToShift, index_hiLo);

    // ##Lo: In first 32 lane, ##Hi: In second 32-bit lane
    __m128i shifted_lo = _mm_unpacklo_epi32(shifted_loLo, shifted_loHi);
    __m128i shifted_hi = _mm_unpacklo_epi32(shifted_hiLo, shifted_hiHi);

    // [##Hi, ???, ???, ##Lo]
    return _shuffleLoHi_i32x4(shifted_lo, _MM_SHUFHALF(3,0), shifted_hi, _MM_SHUFHALF(3,0));
}


// ====== 64-bit ======

// ((x + (1ULL << 63)) >> amount) - ((1ULL << 63) >> amount);

// Shift 64-bit integers right by immediate while shifting in sign bits
#define _signShiftR_i64x2(toShift, amount) (\
    _mm_sub_epi64(\
        _mm_srli_epi64( _mm_add_epi64(toShift, _mm_set1_epi64x(1Ull<<63)), amount ), \
        _mm_srli_epi64( _mm_set1_epi64x(1Ull<<63), amount)\
    ) )

// Shifts each 64-bit element left by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftLvar_u64x2(__m128i toShift, __m128i amount) {
    const __m128i AMOUNT_MASK = _mm_set1_epi64x(64-1);
    
    amount = _mm_and_si128(amount, AMOUNT_MASK);

    __m128i shiftedLo = _mm_sll_epi64(toShift, amount);

    __m128i amountHi = _shuffle_i64x2(amount, _MM_SHUFFLE2(1,1));
    __m128i shiftedHi = _mm_sll_epi64(toShift, amountHi);

    //return _shuffleLoHi_i32x4(shiftedLo, _MM_SHUFHALF(1,0), shiftedHi, _MM_SHUFHALF(3,2));
    return _mm_castpd_si128(_mm_move_sd(
        _mm_castsi128_pd(shiftedHi), _mm_castsi128_pd(shiftedLo)
    ));
}

// Shifts each 64-bit element right by the amount in the corresponding vector, well shifting in zeros (modulo the element size)
__m128i _shiftRvar_u64x2(__m128i toShift, __m128i amount) {
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
