#pragma once

#include <limits.h>
#include "_common.h"
#include "conversion.h"
#include "compare.h"


// Add all 16 unsigned 8-bit elements together
uint16_t _sum_u8x16(__m128i u8ToSum) {
    // [..., sum(a16:a9), 0, 0, 0, sum(a8:a1)]
    __m128i sum16x2 = _mm_sad_epu8(u8ToSum, _mm_setzero_si128());

    return _mm_extract_epi16(sum16x2, 4) + _mm_cvtsi128_si32(sum16x2);
}


// ======================
//      Square root
// ======================

// Calculates `⌊√n⌋` of every unsigned 8-bit integer
__m128i _sqrt_u8x16(__m128i u8_radicand) {
    __m128i root = _mm_setzero_si128();
    __m128i bit = _mm_set1_epi8(1 << (8-2));

    for (int i = 8/2; i > 0; i--) {
        __m128i rootBitSum = _mm_add_epi8(root, bit);
        __m128i radGrtSum_mask = _cmpGrtEq_u8x16(u8_radicand, rootBitSum);

        root = _mm_srli_epi32(root, 1); // Element size doesn't matter
        root = _mm_add_epi8(root, _mm_and_si128(bit, radGrtSum_mask));
        u8_radicand = _mm_sub_epi8(u8_radicand, _mm_and_si128(rootBitSum, radGrtSum_mask));

        bit = _mm_srli_epi32(bit, 2);   // Element size doesn't matter
    }

    return root;
}

// Calculates `⌊√n⌋` of every unsigned 16-bit integer
__m128i _sqrt_u16x8(__m128i u16_radicand) {
    // Single can hold all uint32 and has packed square root instruction
    __m128 num_lo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(u16_radicand) );
    __m128 num_hi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(u16_radicand) );

    __m128i sqrt_lo = _mm_cvttps_epi32( _mm_sqrt_ps(num_lo) );
    __m128i sqrt_hi = _mm_cvttps_epi32( _mm_sqrt_ps(num_hi) );
    
    // Saturates if >15-bits, but sqrt(u16) produces a, at most, 8-bit result
    return _mm_packs_epi32(sqrt_lo, sqrt_hi);
}

// Calculates `⌊√n⌋` of every unsigned 32-bit integer
__m128i _sqrt_u32x4(__m128i u32_radicand) {
    // Double can hold all uint32 and has packed square root instruction
    __m128d num_lo = _convert_i64x2_f64x2( _zeroExtendLo_u32x4_i64x2(u32_radicand) );
    __m128d num_hi = _convert_i64x2_f64x2( _zeroExtendHi_u32x4_i64x2(u32_radicand) );

    // Signed conversion doesn't matter as sqrt(u32) => u16
    __m128i sqrt_lo = _mm_cvttpd_epi32( _mm_sqrt_pd(num_lo) );
    __m128i sqrt_hi = _mm_cvttpd_epi32( _mm_sqrt_pd(num_hi) );
    return _mm_unpacklo_epi32(sqrt_lo, sqrt_hi);
}

// Need inline asm because we want to use the x87's extended-precision square root.
// However even with 80-bit support via long double, compilers tend to produce very inefficient code 
// (like changing the round mode back and forth, or unnecessary isNegative checks)
// For compilers that doesn't support GGC style inline asm, you could try using an external .asm file
#if defined(__GNUC__) && (LDBL_MANT_DIG == 64) // Mantissa check ensures FPU in extended precision mode

// Calculates `⌊√n⌋` of every unsigned 64-bit integer
__m128i _sqrt_u64x2(__m128i u64_radicand) {
    // Use the FPU as it uses a 80-bit float capable of representing all 
    // 64-bit integers exactly and has hardware support for square root
    static const double NEG_ONE = -1.0;

    uint64_t u64_array[2], sqrts[2];
    _mm_storeu_si128((__m128i*)u64_array, u64_radicand);

    for (size_t i=0; i < 2; i++) {
        __asm__(".att_syntax\n\t"
        // Calculate double extended-precision square root
            "fsqrt                  ;"
        // Round to int to check rounding mode
            "fst     %%st(1)        ;"
            "frndint                ;"
        // if (rounded up) {  result -= 1.0  }  [Round down instead]
            "fcomi   %%st(1), %%st  ;"
            "jbe     .keepRounding_sqrtu64_%=;"
            "faddl   %[NEG_ONE]     ;"
        ".keepRounding_sqrtu64_%=:  ;"
        // Store as int and clear stack
            "fistpll %[result]  ;"
            "fstp    %%st       ;"
        : [result] "=m" (sqrts[i])
        : "t" ((long double)u64_array[i]), [NEG_ONE] "m" (NEG_ONE)
        : "st", "st(1)"
        );
    }

    return _mm_loadu_si128((__m128i*)sqrts);
}

#endif


// ======================
//      Saturation
// ======================

// Add the corresponding unsigned 32-bit elements together 
// If that addition would have resulted in an overflow, it is clamped to UINT32_MAX instead
__m128i _addSat_u32x4(__m128i a, __m128i b) {
    __m128i sum = _mm_add_epi32(a,b);
    __m128i hasOverflowed = _cmpGrt_u32x4(a, sum);

    // Overwrite sum with int max when overflowed
    return _mm_or_si128(sum, hasOverflowed);
}

// Subtract the corresponding unsigned 32-bit elements
// If that subtraction would have resulted in an underflow, it is clamped to 0 instead
__m128i _subSat_u32x4(__m128i minuend, __m128i subtrahend) {
    __m128i differance = _mm_sub_epi32(minuend, subtrahend);
    __m128i hasUnderflowed = _cmpLss_u32x4(minuend, differance);

    // Keep only when no underflow (0 & x = 0)
    return _mm_andnot_si128(hasUnderflowed, differance);
}

// Add the corresponding signed 32-bit elements together 
// If the sum is too great/small to be represented, it is clamped to its largest/smallest value
__m128i _addSat_i32x4(__m128i a, __m128i b) {
    __m128i sum = _mm_add_epi32(a, b);

    __m128i inpSignDiffer_MSB = _mm_xor_si128(a, b);
    __m128i resSignDiffer_MSB = _mm_xor_si128(sum, a);

    // Input elements have the same sign, but the result doesn't
    __m128i hasWrapped = _mm_andnot_si128(inpSignDiffer_MSB, resSignDiffer_MSB);
    hasWrapped = _mm_srai_epi32(hasWrapped, 31);

    // INT_MAX: 0b01..11, INT_MIN: 0b10..00
    __m128i clampedValue = _mm_add_epi32(
        _mm_set1_epi32(INT32_MAX), _mm_srli_epi32(a, 31)
    );

    // For SSE4.1, use `blendv_ps` and remove second line for hasWrapped
    return _either_i128(clampedValue, sum, hasWrapped);
}

// Subtract the corresponding signed 32-bit elements together 
// If the difference is too great/small to be represented, it is clamped to its largest/smallest value
__m128i _subSat_i32x4(__m128i minuend, __m128i subtrahend) {
    __m128i difference = _mm_sub_epi32(minuend, subtrahend);

    // Magnitude increases when signs differ (pos - neg = pos + pos)
    __m128i shouldIncrMagnitude_MSB = _mm_xor_si128(minuend, subtrahend);
    __m128i subChangedSign_MSB = _mm_xor_si128(difference, minuend);

    // Must have wrapped when a increase in magnitude caused a sign change
    __m128i hasWrapped = _mm_and_si128(shouldIncrMagnitude_MSB, subChangedSign_MSB);
    hasWrapped = _mm_srai_epi32(hasWrapped, 31);

    // INT_MAX: 0b01..11, INT_MIN: 0b10..00
    __m128i clampedValue = _mm_add_epi32(
        _mm_set1_epi32(INT32_MAX), _mm_srli_epi32(minuend, 31)
    );

    // For SSE4.1, use `blendv_ps` and remove second line for hasWrapped
    return _either_i128(clampedValue, difference, hasWrapped);
}

// Add the corresponding signed 64-bit elements together 
// If the sum is too great/small to be represented, it is clamped to its largest/smallest value
__m128i _addSat_i64x2(__m128i a, __m128i b) {
    __m128i sum = _mm_add_epi64(a, b);

    __m128i resSignDiffer_MSB = _mm_xor_si128(sum, a);
    __m128i inpSignDiffer_MSB = _mm_xor_si128(a, b);

    // Input elements have the same sign, but the result doesn't
    __m128i hasWrapped = _mm_andnot_si128(inpSignDiffer_MSB, resSignDiffer_MSB);
    hasWrapped = _fillWithMSB_i64x2(hasWrapped);

    // INT_MAX: 0b01..11, INT_MIN: 0b10..00
    __m128i clampedValue = _mm_add_epi64(
        _mm_set1_epi64x(INT64_MAX), _mm_srli_epi64(a, 31)
    );

    // For SSE4.1, use `blendv_pd` and remove second line for hasWrapped
    return _either_i128(clampedValue, sum, hasWrapped);
}

// Subtract the corresponding signed 64-bit elements together 
// If the difference is too great/small to be represented, it is clamped to its largest/smallest value
__m128i _subSat_i64x2(__m128i minuend, __m128i subtrahend) {
    __m128i differance = _mm_sub_epi64(minuend, subtrahend);

    // Magnitude increases when signs differ (pos - neg = pos + pos)
    __m128i shouldIncrMagnitude_MSB = _mm_xor_si128(minuend, subtrahend);
    __m128i subChangedSign_MSB = _mm_xor_si128(minuend, differance);

    // Must have wrapped when a increase in magnitude caused a sign change
    __m128i hasWrapped = _mm_and_si128(shouldIncrMagnitude_MSB, subChangedSign_MSB);
    hasWrapped = _fillWithMSB_i64x2(hasWrapped);

    // INT_MAX: 0b01..11, INT_MIN: 0b10..00
    __m128i clampedValue = _mm_add_epi64(
        _mm_set1_epi64x(INT64_MAX), _mm_srli_epi64(minuend, 63)
    );

    // For SSE4.1, use `blendv_pd` and remove second line for hasWrapped
    return _either_i128(clampedValue, differance, hasWrapped);
}


// 64-bit compares are very expensive!
/*
__m128i _addSat_u64x2(__m128i a, __m128i b) {
    __m128i sum = _mm_add_epi64(a,b);
    __m128i hasOverflowed = _cmpGrt_u64x2(a, sum);

    return _mm_or_si128(sum, hasOverflowed);
}
*/

// Add the corresponding unsigned 64-bit elements together 
// If that addition would have resulted in an overflow, it is clamped to UINT64_MAX instead
__m128i _addSat_u64x2(__m128i a, __m128i b) {
    #if 0
    __m128i sum = _mm_add_epi64(a,b);

    // A carry can only occur if either of the inputs have atleast one bit set
    // Positions that create carries will xor to be zero (1 ^ Cin = 0; 1 ^ 1 = 0)
    // ...except for positions that generated a carry and had a carry in 
    __m128i generateCarry = _mm_and_si128(a, b);
    __m128i propergateOrGenWithoutCin = _mm_andnot_si128(sum, _mm_or_si128(a, b));
    __m128i bitsThatCarried = _mm_or_si128(generateCarry, propergateOrGenWithoutCin);

    // If the MSB carried, an overflow must have occurred
    __m128i hasOverflowedMask = _fillWithMSB_i64x2(bitsThatCarried);

    // Overwrite sum with int max when overflowed
    return _mm_or_si128(sum, hasOverflowedMask);
    #else

    __m128i sum = _mm_add_epi64(a,b);
    __m128i hasOverflowed = _cmpGrt_u64x2(a, sum);

    return _mm_or_si128(sum, hasOverflowed);

    #endif
}

// Subtract the corresponding unsigned 64-bit elements
// If that subtraction would have resulted in an underflow, it is clamped to 0 instead
__m128i _subSat_u64x2(__m128i minuend, __m128i subtrahend) {
    // See `cmpLss_u64x2` for the logic
    
    __m128i difference = _mm_sub_epi64(minuend, subtrahend);

   // The cases where we know that the difference's MSB wasn't left over from `a`
    __m128i diffMSBNotFromA = _mm_andnot_si128(minuend, difference);
    // When MSB(b): a>b iff the `a-b` removed the MSB of `a` and a borrow didn't "restore" it
    __m128i bMSBExcludeDiffShowsAGrt = _mm_andnot_si128(_mm_andnot_si128(difference, minuend), subtrahend);

    __m128i hasUnderflowed = _fillWithMSB_i64x2(_mm_or_si128(diffMSBNotFromA, bMSBExcludeDiffShowsAGrt));

    return _mm_andnot_si128(hasUnderflowed, difference);
}