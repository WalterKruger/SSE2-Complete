#pragma once

#include "_common.h"
#include "shiftShuffle.h"
#include <float.h>

// ======================
//       Int-int
// ======================

// ====== 8-bit ======

// Zero extend the lower 8, 8-bit elements to 16-bit
SSECOM_INLINE __m128i _zeroExtendLo_u8x16_i16x8(__m128i u8ToConvert) {
    return _mm_unpacklo_epi8(u8ToConvert, _mm_setzero_si128());
}

// Zero extend the upper 8, 8-bit elements to 16-bit
SSECOM_INLINE __m128i _zeroExtendHi_u8x16_i16x8(__m128i u8ToConvert) {
    return _mm_unpackhi_epi8(u8ToConvert, _mm_setzero_si128());
}

// Sign extend the lower 8, 8-bit elements to 16-bit
SSECOM_INLINE __m128i _signExtendLo_i8x16_i16x8(__m128i i8ToConvert) {
    return _mm_srai_epi16( _mm_unpacklo_epi8(i8ToConvert, i8ToConvert), 8 );
}

// Sign extend the upper 8, 8-bit elements to 16-bit
SSECOM_INLINE __m128i _signExtendHi_i8x16_i16x8(__m128i i8ToConvert) {
    return _mm_srai_epi16( _mm_unpackhi_epi8(i8ToConvert, i8ToConvert), 8 );
}


// ====== 16-bit ======

// Zero extend the lower 4, 16-bit elements to 32-bit
SSECOM_INLINE __m128i _zeroExtendLo_u16x8_i32x4(__m128i u16ToConvert) {
    return _mm_unpacklo_epi16(u16ToConvert, _mm_setzero_si128());
}

// Zero extend the upper 4, 16-bit elements to 32-bit
SSECOM_INLINE __m128i _zeroExtendHi_u16x8_i32x4(__m128i u16ToConvert) {
    return _mm_unpackhi_epi16(u16ToConvert, _mm_setzero_si128());
}

// Sign extend the lower 4, 16-bit elements to 32-bit
SSECOM_INLINE __m128i _signExtendLo_i16x8_i32x4(__m128i i16ToConvert) {
    return _mm_srai_epi32( _mm_unpacklo_epi16(i16ToConvert, i16ToConvert), 16 );
}

// Sign extend the upper 4, 16-bit elements to 32-bit
SSECOM_INLINE __m128i _signExtendHi_i16x8_i32x4(__m128i i16ToConvert) {
    return _mm_srai_epi32( _mm_unpackhi_epi16(i16ToConvert, i16ToConvert), 16 );
}

// Truncade 16-bit elements to 8-bit, return first argument's elements in low half, the second in high
SSECOM_INLINE __m128i _trunc_u16x8_u8x16(__m128i u16LoHalf, __m128i u16HiHalf) {
    __m128i loHalf_lo8 = _mm_and_si128(u16LoHalf, _mm_set1_epi16(UINT8_MAX));
    __m128i hiHalf_lo8 = _mm_and_si128(u16HiHalf, _mm_set1_epi16(UINT8_MAX));

    return _mm_packus_epi16(loHalf_lo8, hiHalf_lo8);
}


// ====== 32-bit ======

// Zero extend the lower 2, 32-bit elements to 64-bit
SSECOM_INLINE __m128i _zeroExtendLo_u32x4_i64x2(__m128i u32ToConvert) {
    return _mm_unpacklo_epi32(u32ToConvert, _mm_setzero_si128());
}

// Zero extend the upper 2, 32-bit elements to 64-bit
SSECOM_INLINE __m128i _zeroExtendHi_u32x4_i64x2(__m128i u32ToConvert) {
    return _mm_unpackhi_epi32(u32ToConvert, _mm_setzero_si128());
}

// Sign extend the lower 2, 32-bit elements to 64-bit
SSECOM_INLINE __m128i _signExtendLo_i32x4_i64x2(__m128i i32ToConvert) {
    return _mm_unpacklo_epi32(i32ToConvert, _mm_cmplt_epi32(i32ToConvert, _mm_setzero_si128()));
}

// Sign extend the upper 2, 32-bit elements to 64-bit
SSECOM_INLINE __m128i _signExtendHi_i32x4_i64x2(__m128i i32ToConvert) {
    return _mm_unpackhi_epi32(i32ToConvert, _mm_cmplt_epi32(i32ToConvert, _mm_setzero_si128()));
}

// Truncade 32-bit elements to 16-bit, return first argument's elements in low half, the second in high
SSECOM_INLINE __m128i _trunc_u32x4_u16x8(__m128i u32LoHalf, __m128i u32HiHalf) {
    // [##, ##, 3Lo, 2Lo, ##, ##, 1Lo, 0Lo]
    __m128i resultLo_inLo64Part = _mm_shufflehi_epi16( _mm_shufflelo_epi16(u32LoHalf, _MM_SHUFFLE(0,0,2,0)), _MM_SHUFFLE(0,0,2,0) );
    __m128i resultHi_inLo64Part = _mm_shufflehi_epi16( _mm_shufflelo_epi16(u32HiHalf, _MM_SHUFFLE(0,0,2,0)), _MM_SHUFFLE(0,0,2,0) );
    
    return _shuffleLoHi_i32x4(resultLo_inLo64Part, _MM_SHUFHALF(2,0), resultHi_inLo64Part, _MM_SHUFHALF(2,0));
}

// Convert unsigned 32-bit integers into 16-bit via unsigned saturation
// (When value is too large to fit, it is clamped to UINT32_MAX)
__m128i _satConvert_u32x4_u16x8(__m128i u32LoHalf, __m128i u32HiHalf) {
    // TODO: Maybe there is a cheaper way...
    __m128i loWontSat = _mm_cmpeq_epi16( _mm_srli_epi32(u32LoHalf, 16), _mm_setzero_si128());
    __m128i loWillSat = _mm_cmpeq_epi16(loWontSat, _mm_setzero_si128());

    __m128i hiWontSat = _mm_cmpeq_epi16(_mm_srli_epi32(u32HiHalf, 16), _mm_setzero_si128());
    __m128i hiWillSat = _mm_cmpeq_epi16(hiWontSat, _mm_setzero_si128());
    
    // Cause a saturation before truncating
    u32LoHalf = _mm_or_si128(u32LoHalf, loWillSat);
    u32HiHalf = _mm_or_si128(u32HiHalf, hiWillSat);

    return _trunc_u32x4_u16x8(u32LoHalf, u32HiHalf);
}

// ====== 64-bit ======

// Truncade 64-bit elements to 32-bit, return first argument's elements in low half, the second in high
SSECOM_INLINE __m128i _trunc_u64x4_u32x4(__m128i u64LoHalf, __m128i u64HiHalf) {
    return _shuffleLoHi_i32x4(u64LoHalf, _MM_SHUFHALF(2,0), u64HiHalf, _MM_SHUFHALF(2,0));
}

// Convert unsigned 64-bit integers into 32-bit via unsigned saturation
// (When value is too large to fit, it is clamped to UINT32_MAX)
SSECOM_INLINE __m128i _satConvert_u64x4_u32x4(__m128i u64LoHalf, __m128i u64HiHalf) {
    __m128i truncNoSat = _trunc_u64x4_u32x4(u64LoHalf, u64HiHalf);

    __m128i partsThatCauseSat = _shuffleLoHi_i32x4(
        u64LoHalf, _MM_SHUFHALF(3,1), u64HiHalf, _MM_SHUFHALF(3,1)
    );

    __m128i satIfNotZero = _mm_cmpeq_epi32(_mm_cmpeq_epi32(partsThatCauseSat, _mm_setzero_si128()), _mm_setzero_si128());
    return _mm_or_si128(truncNoSat, satIfNotZero);
}









// ======================
//       Float-int
// ======================


// ===== 16-bit int =====

// Converts the lower unsigned 16-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convertLo_u16x8_f32x4(__m128i u16_toCvt) {
    return _mm_cvtepi32_ps(_zeroExtendLo_u16x8_i32x4(u16_toCvt));
}

// Converts the upper unsigned 16-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convertHi_u16x8_f32x4(__m128i u16_toCvt) {
    return _mm_cvtepi32_ps(_zeroExtendHi_u16x8_i32x4(u16_toCvt));
}

// Converts the lower signed 16-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convertLo_i16x8_f32x4(__m128i u16_toCvt) {
    return _mm_cvtepi32_ps(_signExtendLo_i16x8_i32x4(u16_toCvt));
}

// Converts the upper signed 16-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convertHi_i16x8_f32x4(__m128i u16_toCvt) {
    return _mm_cvtepi32_ps(_signExtendHi_i16x8_i32x4(u16_toCvt));
}


// ===== 32-bit int =====


// Converts unsigned 32-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convert_u32x4_f32x4(__m128i u32_toCvt) {
    // Convert the high and low 16-bit seperatly, then recombind
    __m128i low16Bits = _mm_and_si128(u32_toCvt, _mm_set1_epi32(UINT16_MAX));
    __m128i hi16Bits = _mm_srli_epi32(u32_toCvt, 16);

    // Signed 32 => float32
    __m128 hiCvted =  _mm_cvtepi32_ps(hi16Bits);
    __m128 lowCvted = _mm_cvtepi32_ps(low16Bits);

    // lowCvt + (hiCvt << 16)
    return _mm_add_ps(lowCvted, _mm_mul_ps(hiCvted, _mm_set1_ps(1<<16)));
}

// Converts signed 32-bit integers into 32-bit floats
SSECOM_INLINE __m128 _convert_i32x4_f32x4(__m128i i32_toCvt) {
    return _mm_cvtepi32_ps(i32_toCvt);
}

// Alternative method: signed32ToDbl(x) + ( ((int32_t)x < 0)? (1ull<<32) : 0 )

// Converts the lower unsigned 32-bit integers into 64-bit floats
SSECOM_INLINE __m128d _convertLo_u32x4_f64x2(__m128i u32_toCvt) {
    // Setting the exponent to 2**52 "shifts" the entire mantissa into the integer part
    // However, this also shifts over the implict leading one
    const __m128d MANT_AS_INT_EXPO = _mm_set1_pd(1ull << (DBL_MANT_DIG-1));

    __m128d toCvt_u64 = _mm_castsi128_pd( _zeroExtendLo_u32x4_i64x2(u32_toCvt) );
    return _mm_sub_pd(_mm_or_pd(toCvt_u64, MANT_AS_INT_EXPO), MANT_AS_INT_EXPO);
}

// Converts the upper unsigned 32-bit integers into 64-bit floats
SSECOM_INLINE __m128d _convertHi_u32x4_f64x2(__m128i u32_toCvt) {
    // Setting the exponent to 2**52 "shifts" the entire mantissa into the integer part
    // However, this also shifts over the implict leading one
    const __m128d MANT_AS_INT_EXPO = _mm_set1_pd(1ull << (DBL_MANT_DIG-1));

    __m128d toCvt_u64 = _mm_castsi128_pd( _zeroExtendHi_u32x4_i64x2(u32_toCvt) );
    return _mm_sub_pd(_mm_or_pd(toCvt_u64, MANT_AS_INT_EXPO), MANT_AS_INT_EXPO);
}

// Converts the lower signed 32-bit integers into 64-bit floats
SSECOM_INLINE __m128d _convertLo_i32x4_f64x2(__m128i i32_toCvt) {
    return _mm_cvtepi32_pd(i32_toCvt);
}

// ===== 64-bit int =====

// Converts signed 64-bit integers into 32-bit floats, store in lower half and zero upper
SSECOM_INLINE __m128 _convert_i64x2_f32x4(__m128i i64_toCvt) {
    #ifdef __clang__
    // Clang produces multiple redundent zero vectors with the other code...
    int64_t i64_array[2];
    _mm_store_si128((__m128i*)i64_array, i64_toCvt);

    return _mm_setr_ps((float)i64_array[0], (float)i64_array[1], 0, 0);

    #else
    // Statement ordering used to manipulate asm output
    __m128i hiInLow = _shuffle_i64x2(i64_toCvt, _MM_SHUFFLE2(0,1));
    
    int64_t i64_lo = _mm_cvtsi128_si64(i64_toCvt);
    __m128 dbl_lo = _mm_cvtsi64_ss(_mm_setzero_ps(), i64_lo);
    
    int64_t i64_hi = _mm_cvtsi128_si64(hiInLow);
    __m128 dbl_hi = _mm_cvtsi64_ss(_mm_setzero_ps(), i64_hi);

    return _mm_unpacklo_ps(dbl_lo, dbl_hi);
    #endif
}


// Converts signed 64-bit integers into 64-bit floats
SSECOM_INLINE __m128d _convert_i64x2_f64x2(__m128i i64_toCvt) {
    // Statement ordering used to manipulate asm output
    __m128i hiInLow = _shuffle_i64x2(i64_toCvt, _MM_SHUFFLE2(0,1));
    
    int64_t i64_lo = _mm_cvtsi128_si64(i64_toCvt);
    __m128d dbl_lo = _mm_cvtsi64_sd(_mm_setzero_pd(), i64_lo);
    
    int64_t i64_hi = _mm_cvtsi128_si64(hiInLow);
    __m128d dbl_hi = _mm_cvtsi64_sd(_mm_setzero_pd(), i64_hi);

    return _mm_unpacklo_pd(dbl_lo, dbl_hi);
}

// Converts unsigned 64-bit integers into 64-bit floats
SSECOM_INLINE __m128d _convert_u64x2_f64x2(__m128i u64_toCvt) {
    // Magic method based on clang (when converting a uint64 => double)
    const uint32_t MAGIC_EXP_LO = DBL_MAX_EXP + DBL_MANT_DIG - 2;
    const uint32_t MAGIC_EXP_HI = DBL_MAX_EXP + DBL_MANT_DIG+32 - 2;

    const __m128i MAGIC_EXP = _mm_setr_epi32(
        MAGIC_EXP_LO << (DBL_MANT_DIG-33), MAGIC_EXP_LO << (DBL_MANT_DIG-33),
        MAGIC_EXP_HI << (DBL_MANT_DIG-33), MAGIC_EXP_HI << (DBL_MANT_DIG-33)
    );

    const __m128d MAGIC_TERM_LO = _mm_set1_pd(1ull << (DBL_MANT_DIG-1));
    const __m128d MAGIC_TERM_HI = _mm_set1_pd(1.9342813113834067e+25); // 2**(DBL_MANT_DIG-1+32);

    // [inp1.hi, inp0.hi, inp1.lo, inp0.lo]
    __m128i inp_hihi_lolo = _mm_shuffle_epi32(u64_toCvt, _MM_SHUFFLE(3,1,2,0));

    // [MAGIC, inp1.lo, MAGIC, inp0.lo]
    __m128d magicFlt_lo = _mm_castsi128_pd(_mm_unpacklo_epi32(inp_hihi_lolo, MAGIC_EXP));
    __m128d magicFlt_hi = _mm_castsi128_pd(_mm_unpackhi_epi32(inp_hihi_lolo, MAGIC_EXP));

    // Promotes using a memory argument rather than a 128-bit to register
    magicFlt_lo = _mm_sub_pd(magicFlt_lo, *(__m128d*)&MAGIC_TERM_LO);
    magicFlt_hi = _mm_sub_pd(magicFlt_hi, *(__m128d*)&MAGIC_TERM_HI);

    return _mm_add_pd(magicFlt_hi, magicFlt_lo);
}

// Converts unsigned 64-bit integers into 32-bit floats, store in lower half and zero upper
__m128 _convert_u64x2_f32x4(__m128i u64_toCvt) {
    //
    //if ((int64_t)toCvt >= 0) {
    //    return signedIntToF32(toCvt);
    //} else {
    //    int64_t halfRoundUp = (toCvt >> 1) | (toCvt | 1);
    //    float halfCvted = signedIntToF32(halfRoundUp);
    //    return halfCvted + halfCvted;
    //}
    //
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


// ===== 32-bit float =====

// Converts 32-bit floats into unsigned 32-bit integers via truncation
// Valid for inputs [INT32_MIN, UINT32_MAX] (negative inputs act like converting to signed, then casting to unsigned)
SSECOM_INLINE __m128i _convert_f32x4_u32x4(__m128 f32_toCvt) {
    __m128i directCvt = _mm_cvttps_epi32(f32_toCvt);
    __m128 f32_scaled = _mm_sub_ps(f32_toCvt, _mm_set1_ps(1ull << 32));
    __m128i scaledCvt = _mm_cvttps_epi32(f32_scaled);

    __m128i isNegMask = _mm_srai_epi32(directCvt, 31);
    return _mm_or_si128(_mm_and_si128(scaledCvt, isNegMask), directCvt);
}

// Converts 32-bit floats into signed 64-bit integers via truncation
// Valid for inputs [INT64_MIN, INT64_MAX]
SSECOM_INLINE __m128i _convertLo_f32x4_i64x2(__m128 f32_toCvt) {
    uint64_t i64_lo = _mm_cvttss_si64(f32_toCvt);
    uint64_t i64_hi = _mm_cvttss_si64(_mm_shuffle_ps(f32_toCvt, f32_toCvt, _MM_SHUFFLE(0,0,0,1)));
    
    __m128i lo_vec = _mm_cvtsi64_si128(i64_lo);
    __m128i hi_vec = _mm_cvtsi64_si128(i64_hi);

    return _mm_unpacklo_epi64(lo_vec, hi_vec);
}

// Converts 64-bit floats into unsigned 64-bit integers via truncation
// Valid for inputs [INT64_MIN, UINT64_MAX] (negative inputs act like converting to signed, then casting to unsigned)
__m128i _convertLo_f32x4_u64x2(__m128 f32_toCvt) {
    const __m128 OVERFLOW_THRESHOLD = _mm_set1_ps(1ull << 63);

    // If input is too large to fit in signed int, scale so it does
    __m128 willOverflowMask = _mm_cmpge_ps(f32_toCvt, OVERFLOW_THRESHOLD);
    __m128 inputNoOverflow = _mm_sub_ps(f32_toCvt, _mm_and_ps(OVERFLOW_THRESHOLD, willOverflowMask));
    __m128i scaledCvt = _convertLo_f32x4_i64x2(inputNoOverflow);

    // Scaling "removed" the MSB, so re-add it
    // "Align" lower two masks with the converted upper [mask1, mask1, mask0, mask0]
    __m128i maskAligned = _mm_shuffle_epi32( _mm_castps_si128(willOverflowMask), _MM_SHUFFLE(1, 1, 0, 0));
    return _mm_xor_si128(scaledCvt, _mm_slli_epi64( maskAligned, 63 ));
}


// ===== 64-bit float =====

// Converts 64-bit floats into signed 32-bit integers via truncation, store in lower half and zero upper
SSECOM_INLINE __m128i _convert_f64x2_i32x4(__m128d f64_toCvt) {
    return _mm_cvttpd_epi32(f64_toCvt);
}

// Converts 64-bit floats into unsigned 32-bit integers via truncation, store in lower half and zero upper
// Valid for inputs [INT32_MIN, UINT32_MAX] (negative inputs act like converting to signed, then casting to unsigned)
SSECOM_INLINE __m128i _convert_f64x2_u32x4(__m128d f64_toCvt) {
    // Use f64 => signed 64 instruction [in range of (0, UINT32_MAX)]
    uint64_t i64_lo = _mm_cvttsd_si64(f64_toCvt);
    uint64_t i64_hi = _mm_cvttsd_si64(_mm_unpackhi_pd(f64_toCvt, f64_toCvt));
    
    // (Upper elements are zeroed out)
    __m128i lo_vec = _mm_cvtsi32_si128((uint32_t)i64_lo);
    __m128i hi_vec = _mm_cvtsi32_si128((uint32_t)i64_hi);

    return _mm_unpacklo_epi32(lo_vec, hi_vec);
}

// Converts 64-bit floats into signed 64-bit integers via truncation
// Valid for inputs [INT64_MIN, INT64_MAX]
SSECOM_INLINE __m128i _convert_f64x2_i64x2(__m128d f64_toCvt) {
    uint64_t i64_lo = _mm_cvtsd_si64(f64_toCvt);
    uint64_t i64_hi = _mm_cvtsd_si64(_mm_unpackhi_pd(f64_toCvt, f64_toCvt));
    
    __m128i lo_vec = _mm_cvtsi64_si128(i64_lo);
    __m128i hi_vec = _mm_cvtsi64_si128(i64_hi);

    return _mm_unpacklo_epi64(lo_vec, hi_vec);
}

// Converts 64-bit floats into unsigned 64-bit integers via truncation
// Valid for inputs [INT64_MIN, UINT64_MAX] (negative inputs act like converting to signed, then casting to unsigned)
__m128i _convert_f64x2_u64x2(__m128d f64_toCvt) {
    const __m128d OVERFLOW_THRESHOLD = _mm_set1_pd(1ull << 63);

    // If input is too large to fit in signed int, scale so it does
    __m128d willOverflowMask = _mm_cmpge_pd(f64_toCvt, OVERFLOW_THRESHOLD);
    __m128d inputNoOverflow = _mm_sub_pd(f64_toCvt, _mm_and_pd(OVERFLOW_THRESHOLD, willOverflowMask));
    __m128i scaledCvt = _convert_f64x2_i64x2(inputNoOverflow);

    // Scaling "removed" the MSB, so re-add it
    return _mm_xor_si128(scaledCvt, _mm_slli_epi64( _mm_castpd_si128(willOverflowMask), 63 ));
}






// Converts signed 32-bit integers stored as two's complement to sign and magnitude representation
SSECOM_INLINE __m128i _toSignAndMag_i32x4(__m128i i32_as2sComp) {
    // If negative: negate and set sign bit
    __m128i isNegMask = _mm_cmplt_epi32(i32_as2sComp, _mm_setzero_si128()); // _mm_srai_epi32(i32_as2sComp, 31);
    __m128i notIfNeg = _mm_xor_si128(i32_as2sComp, isNegMask);
    
    // x - 0b01..11 == x + 0b10..01   (Complete negation by adding one, and also set sign bit)
    __m128i addOneSign_ifNeg = _mm_srli_epi32(isNegMask, 1);
    return _mm_sub_epi32(notIfNeg, addOneSign_ifNeg);
}
