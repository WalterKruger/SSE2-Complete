#pragma once
#include "../_perfCommon.h"
#include <float.h>

#include "../../include/sseCom_parts/conversion.h"
#include "../../include/sseCom_parts/multiply.h"
#include "../../include/sseCom_parts/shuffle.h"

#define dAsI128(x) _mm_castpd_si128(x)
#define iAsPd(x) _mm_castsi128_pd(x)
#define DUPLICATE_5 CASE_Ni(0) CASE_Ni(1) CASE_Ni(2) CASE_Ni(3) CASE_Ni(4)



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
