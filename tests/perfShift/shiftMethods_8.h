#pragma once
#include "../_perfCommon.h"
#include <float.h>

#include "../../include/sseCom_parts/shuffle.h"
#include "../../include/sseCom_parts/multiply.h"
#include "../../include/sseCom_parts/shift.h"   // Constant amount macros



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

    __asm__ volatile("" : "+x" (shiftBy2_2ndMSB));

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
