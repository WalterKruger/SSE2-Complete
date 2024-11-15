#pragma once

#include "_common.h"
#include "conversion.h"
#ifdef _MSC_VER
#include <intrin.h> // __umulh, __mulh
#endif

// ======================
//       Multiply
// ======================

// ====== 8-bit ======

// Multiply the lower 8 unsigned 8-bit elements and return 8, full 16-bit products
SSECOM_INLINE __m128i _mulFull_u8x16Lo(__m128i termA, __m128i termB) {
    __m128i lo8_A = _zeroExtendLo_u8x16_i16x8(termA);
    __m128i lo8_B = _zeroExtendLo_u8x16_i16x8(termB);

    return _mm_mullo_epi16(lo8_A, lo8_B);
}

// Multiply the upper 8 unsigned 8-bit elements and return 8, full 16-bit products
SSECOM_INLINE __m128i _mulFull_u8x16Hi(__m128i termA, __m128i termB) {
    __m128i lo8_A = _zeroExtendHi_u8x16_i16x8(termA);
    __m128i lo8_B = _zeroExtendHi_u8x16_i16x8(termB);

    return _mm_mullo_epi16(lo8_A, lo8_B);
}

// Multiply the lower 8 signed 8-bit elements and return 8, full 16-bit products
SSECOM_INLINE __m128i _mulFull_i8x16Lo(__m128i termA, __m128i termB) {
    // Same as zero extending and shifting left 8 bits
    // (Since each element is in the MSB of the padded version's, `mulhi_epi16` will sign extend)
    __m128i aZeroPad_lo = _mm_unpacklo_epi8(_mm_setzero_si128(), termA);
    __m128i bZeroPad_lo = _mm_unpacklo_epi8(_mm_setzero_si128(), termB);

    // Since each term is multiplied by 2^8, their product is shifted entirely into the high part
    return _mm_mulhi_epi16(aZeroPad_lo, bZeroPad_lo);
}

// Multiply the upper 8 signed 8-bit elements and return 8, full 16-bit products
SSECOM_INLINE __m128i _mulFull_i8x16Hi(__m128i termA, __m128i termB) {
    // Same as zero extending and shifting left 8 bits
    // (Ensures each 8-bit element would be sign extended to 32-bits during `mulhi_epi16`)
    __m128i aZeroPad_lo = _mm_unpackhi_epi8(_mm_setzero_si128(), termA);
    __m128i bZeroPad_lo = _mm_unpackhi_epi8(_mm_setzero_si128(), termB);

    // Since each term is multiplied by 2^8, their product is shifted entirely into the high part
    return _mm_mulhi_epi16(aZeroPad_lo, bZeroPad_lo);
}

// Multiply all unsigned 8-bit elements and return the lower 8-bit of the intermediate 16-bit result 
__m128i _mulLo_u8x16(__m128i termA, __m128i termB) {
    const __m128i HIGH8_MASK = _mm_set1_epi16(0xFF00);

    // mullo_u8(a_16, b_16) = (u8)(a_16 * b_16) [See the long multiplication algorithm]
    // [..., ###, a3 * b3 , ###, a1 * b1]
    __m128i res_even = _mm_andnot_si128(HIGH8_MASK, _mm_mullo_epi16(termA, termB));

    // ...0    a6 | 0    a4 | 0    a2
    // ...b6    0 | b4    0 | b2    0
    __m128i res_odd = _mm_mullo_epi16(_mm_srli_epi16(termA, 8), _mm_and_si128(HIGH8_MASK, termB));

    // [...    0    | a3 * b3 |    0    | a1 * b1]
    // [... a4 * b4 |    0    | a2 * b2 |    0   ]
    __m128i result_full = _mm_or_si128(res_even, res_odd);

    return result_full;
}

// Multiply all signed 8-bit elements and return the lower 8-bit of the intermediate 16-bit result 
__m128i _mulLo_i8x16(__m128i termA, __m128i termB) {
    // Low product is the same, regardless of signess
    return _mulLo_u8x16(termA, termB);
}

/*__m128i _mulLo_u8x16(__m128i termA, __m128i termB) {
    __m128i lo8_A = _zeroExtend_u8Lo_i16x8(termA);
    __m128i lo8_B = _zeroExtend_u8Lo_i16x8(termB);
    __m128i result16_lo = _mm_mullo_epi16(lo8_A, lo8_B);

    __m128i hi8_A = _zeroExtend_u8Hi_i16x8(termA);
    __m128i hi8_B = _zeroExtend_u8Hi_i16x8(termB);
    __m128i result16_hi = _mm_mullo_epi16(hi8_A, hi8_B);

    return _trunc_u16HiLo_u8x16(result16_lo, result16_hi);
}*/

// Multiply all unsigned 8-bit elements and return the upper 8-bit of the intermediate 16-bit result 
__m128i _mulHi_u8x16(__m128i termA, __m128i termB) {
    const __m128i LOW8_MASK = _mm_set1_epi16(UINT8_MAX);

    // mulhi_u16((a << 8), b) = (a * b) >> 8
    // [..., 0, HI(a3 * b3), 0, HI(a1 * b1)]
    __m128i res_even = _mm_mulhi_epu16(_mm_slli_epi16(termA, 8), _mm_and_si128(termB, LOW8_MASK));

    // [..., HI(a4 * b4), LO(a4 * b4), HI(a2 * b2), LO(a2 * b2)]
    __m128i res_odd = _mm_mullo_epi16(_mm_and_si128(termA, LOW8_MASK), _mm_and_si128(termB, LOW8_MASK));

    // [...     0      | HI(a3, b3) |     0      | HI(a1, b1) ]
    // [... HI(a4, b4) | LO(a4, b4) | HI(a1, b2) | LO(a2, b2) ]
    __m128i result_full = _mm_or_si128(res_even, _mm_andnot_si128(LOW8_MASK, res_odd));

    return result_full;
}

// Multiply all signed 8-bit elements and return the upper 8-bit of the intermediate 16-bit result 
__m128i _mulHi_i8x16(__m128i termA, __m128i termB) {
    // mulFull_i8 is fairly cheap

    __m128i mulFull_lo = _mulFull_i8x16Lo(termA, termB);
    __m128i mulFull_hi = _mulFull_i8x16Hi(termA, termB);

    __m128i mulHi_lo = _mm_srli_epi16(mulFull_lo, 8);
    __m128i mulHi_hi = _mm_srli_epi16(mulFull_hi, 8);

    // Can't saturate as shift zeroed upper 8 bits
    return _mm_packus_epi16(mulHi_lo, mulHi_hi);
}

// ====== 16-bit ======

// Multiply all signed 16-bit elements and return the lower 16-bit of the intermediate 32-bit product 
SSECOM_INLINE __m128i _mulLo_i16x2(__m128i termA, __m128i termB) {
    return _mm_mullo_epi16(termA, termB);
}

// Multiply all unsigned 16-bit elements and return the lower 16-bit of the intermediate 32-bit product 
SSECOM_INLINE __m128i _mulLo_u16x2(__m128i termA, __m128i termB) {
    // The low result is the same regardless of signess
    return _mm_mullo_epi16(termA, termB);
}

// Multiply all signed 16-bit elements and return the upper 16 bits of the intermediate 32-bit product 
SSECOM_INLINE __m128i _mulHi_i16x2(__m128i termA, __m128i termB) {
    return _mm_mulhi_epi16(termA, termB);
}

// Multiply all unsigned 16-bit elements and return the upper 16 bits of the intermediate 32-bit product 
SSECOM_INLINE __m128i _mulHi_u16x2(__m128i termA, __m128i termB) {
    return _mm_mulhi_epu16(termA, termB);
}


// Multiply the lower 4 unsigned 16-bit elements and return 4, full 32-bit products
SSECOM_INLINE __m128i _mulFull_u16x8Lo(__m128i termA, __m128i termB) {
    // Low sign/unsigned mul acts the same
    __m128i resultLo = _mm_mullo_epi16(termA, termB);
    __m128i resultHi = _mm_mulhi_epu16(termA, termB);

    return _mm_unpacklo_epi16(resultLo, resultHi);
}

// Multiply the upper 4 unsigned 16-bit elements and return 4, full 32-bit products
SSECOM_INLINE __m128i _mulFull_u16x8Hi(__m128i termA, __m128i termB) {
    // Low sign/unsigned mul act the same
    __m128i resultLo = _mm_mullo_epi16(termA, termB);
    __m128i resultHi = _mm_mulhi_epu16(termA, termB);

    return _mm_unpackhi_epi16(resultLo, resultHi);
}

// Multiply the even (0,2,4,6) 4 signed 16-bit elements and return 4, full 32-bit products
SSECOM_INLINE __m128i _mulFull_i16x8Even(__m128i termA, __m128i termB) {
    __m128i bEvenOnly = _mm_and_si128(termB, _mm_set1_epi32(UINT16_MAX));

    // [... a4*0 + a3*b3, a2*0 + a1*b1]
    return _mm_madd_epi16(termA, bEvenOnly);
}

// Multiply the odd (1,3,5,7) 4 signed 16-bit elements and return 4, full 32-bit products
SSECOM_INLINE __m128i _mulFull_i16x8Odd(__m128i termA, __m128i termB) {
    __m128i bOddOnly = _mm_and_si128(termB, _mm_set1_epi32(UINT16_MAX << 16));

    // [... a4*b4 + a3*0, a2*b2 + a1*0]
    return _mm_madd_epi16(termA, bOddOnly);
}

// ====== 32-bit ======

// Multiply all unsigned 32-bit elements and return the lower 32-bit of the intermediate 64-bit product
SSECOM_INLINE __m128i _mulLo_u32x4(__m128i termA, __m128i termB) {
    // [HI(a2*b2), LO(a2*b2), HI(a0*b0), LO(a0*b0)]
    __m128i mulFull_even = _mm_mul_epu32(termA, termB);

    __m128i aShifted = _mm_srli_epi64(termA, 32);
    __m128i bShifted = _mm_srli_epi64(termB, 32);

    // [HI(a3*b3), LO(a3*b3), HI(a1*b1), LO(a1*b1)]
    __m128i mulFull_odd = _mm_mul_epu32(aShifted, bShifted);
    
    // [LO(a3*b3), LO(a1*b1), LO(a2*b2), LO(a0*b0)]
    __m128i result_evenOdd = _shuffleLoHi_i32x4(mulFull_even, _MM_SHUFHALF(2,0), mulFull_odd, _MM_SHUFHALF(2,0));
    return _mm_shuffle_epi32(result_evenOdd, _MM_SHUFFLE(3, 1, 2, 0));
}

// Multiply all signed 32-bit elements and return the lower 32-bit of the intermediate 64-bit result
__m128i _mulLo_i32x4(__m128i termA, __m128i termB) {
    // Low product is the same, regardless of signess
    return _mulLo_u32x4(termA, termB);
}

// Multiply all unsigned 32-bit elements and return the higher 32-bit of the intermediate 64-bit result 
SSECOM_INLINE __m128i _mulHi_u32x4(__m128i termA, __m128i termB) {
    // [HI(a2*b2), LO(a2*b2), HI(a0*b0), LO(a0*b0)]
    __m128i mulFull_even = _mm_mul_epu32(termA, termB);

    __m128i aShifted = _mm_srli_epi64(termA, 32);
    __m128i bShifted = _mm_srli_epi64(termB, 32);

    // [HI(a3*b3), LO(a3*b3), HI(a1*b1), LO(a1*b1)]
    __m128i mulFull_odd = _mm_mul_epu32(aShifted, bShifted);

    // [HI(a3*b3), HI(a1*b1), HI(a2*b2), HI(a0*b0)]
    __m128i mulHi_evenOdd = _shuffleLoHi_i32x4(mulFull_even, _MM_SHUFHALF(3,1), mulFull_odd, _MM_SHUFHALF(3,1));
    return _mm_shuffle_epi32(mulHi_evenOdd, _MM_SHUFFLE(3, 1, 2, 0));
}

// Multiply all signed 32-bit elements and return the higher 32-bit of the intermediate 64-bit result 
SSECOM_INLINE __m128i _mulHi_i32x4(__m128i termA, __m128i termB) {
    // A full signed multiplication sign extends rather than zero extends
    __m128i unsignedHi = _mulHi_u32x4(termA, termB);

    // signExtend(a).hi = (a < 0)? -1 : 0   [b.lo * a.hi = -b.lo or 0]
    __m128i b_ifANeg = _mm_and_si128(termB, _mm_srai_epi32(termA, 31));
    __m128i a_ifBNeg = _mm_and_si128(termA, _mm_srai_epi32(termB, 31));

    // Add the partial products (subtract since the values are suppose to be negative)
    return _mm_sub_epi32(_mm_sub_epi32(unsignedHi, b_ifANeg), a_ifBNeg);
}

// Multiply the even (0,2) 2 unsigned 32-bit elements and return 2, full 64-bit products
SSECOM_INLINE __m128i _mulFull_u32x4Even(__m128i termA, __m128i termB) {
    return _mm_mul_epu32(termA, termB);
}

// Multiply the even (0,2) 2 signed 32-bit elements and return 2, full 64-bit products
__m128i _mulFull_i32x4Even(__m128i termA, __m128i termB) {
    // A full signed multiplication sign extends rather than zero extends
    __m128i unsignedResult = _mm_mul_epu32(termA, termB);

    __m128i aSign = _mm_srai_epi32(termA, 31);
    __m128i bSign = _mm_srai_epi32(termB, 31);

    // signExtend(a).hi * b = (a < 0)? -b : 0
    // So only keep low part when negative, to subtract later
    __m128i aLo_bHi_neg = _mm_slli_epi64( _mm_and_si128(termA, bSign), 32 );
    __m128i bLo_aHi_neg = _mm_slli_epi64( _mm_and_si128(termB, aSign), 32 );

    // resultHi = HI(a*b) - (bIsNeg? a) - (aIsNeg? b)
    return _mm_sub_epi64(_mm_sub_epi64(unsignedResult, aLo_bHi_neg), bLo_aHi_neg );
}


// ====== 64-bit ======

// Multiply all signed 64-bit elements and return the lower 64-bit of the intermediate 128-bit result 
__m128i _mulLo_i64x2(__m128i termA, __m128i termB) {
    __m128i aHi = _mm_srli_epi64(termA, 32);
    __m128i bHi = _mm_srli_epi64(termB, 32);

    // 'mul_epu32' multiples the lower 32 bits of each 64-bit lane
    __m128i aLo_bLo = _mm_mul_epu32(termA, termB);
    __m128i aLo_bHi = _mm_mul_epu32(termA, bHi);
    __m128i aHi_bLo = _mm_mul_epu32(aHi, termB);

    __m128i result = aLo_bLo;
    result = _mm_add_epi64(result, _mm_slli_epi64(aLo_bHi, 32));
    result = _mm_add_epi64(result, _mm_slli_epi64(aHi_bLo, 32));
    return result;
}

// Multiply all unsigned 64-bit elements and return the lower 64-bit of the intermediate 128-bit product
__m128i _mulLo_u64x2(__m128i termA, __m128i termB) {
    // The low result is the same regardless of signess
    return _mulLo_i64x2(termA, termB);
}

#if defined(__SIZEOF_INT128__) || defined(_MSC_VER)
// Multiply both unsigned 64-bit elements and return the higher 64-bit of the intermediate 128-bit product 
__m128i _mulHi_u64x2(__m128i termA, __m128i termB) {
    uint64_t A_array[2], B_array[2], product[2];
    
    _mm_store_si128((__m128i*)A_array, termA);
    _mm_store_si128((__m128i*)B_array, termB);

    for (size_t i=0; i < 2; i++)
        #ifdef __SIZEOF_INT128__
            product[i] = (__uint128_t)A_array[i] * B_array[i] >> 64;
        #elif defined(_MSC_VER)
            product[i] = __umulh(A_array[i], B_array[i]);
        #endif
    
    return _mm_loadu_si128((__m128i*)product);
}

// Multiply both signed 64-bit elements and return the higher 64-bit of the intermediate 128-bit product 
__m128i _mulHi_i64x2(__m128i termA, __m128i termB) {
    int64_t A_array[2], B_array[2], product[2];
    
    _mm_store_si128((__m128i*)A_array, termA);
    _mm_store_si128((__m128i*)B_array, termB);

    for (size_t i=0; i < 2; i++)
        #ifdef __SIZEOF_INT128__
            product[i] = (__int128_t)A_array[i] * B_array[i] >> 64;
        #elif defined(_MSC_VER)
            product[i] = __mulh(A_array[i], B_array[i]);
        #endif
    
    return _mm_loadu_si128((__m128i*)product);
}
#endif
