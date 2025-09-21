#pragma once

#include "_common.h"
#include "conversion.h"
#include "compare.h"
#include "multiply.h"


// `getDivMagic` needs double word division
__m128i _div_u16x8(__m128i, __m128i);
__m128i _div_u32x4(__m128i, __m128i);
__m128i _div_u64x2(__m128i, __m128i);

// ====== 8-bit ======

// Divides each unsigned 8-bit integers by the corresponding unsigned 8-bit divisor
// NOTE: `divP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _div_u8x16(__m128i numerator, __m128i denominator) {
    __m128i num_lo = _zeroExtendLo_u8x16_i16x8(numerator);
    __m128i num_hi = _zeroExtendHi_u8x16_i16x8(numerator);

    __m128 num_lolo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(num_lo) ); 
    __m128 num_lohi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(num_lo) );
    __m128 num_hilo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(num_hi) );
    __m128 num_hihi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(num_hi) );


    __m128i deno_lo = _zeroExtendLo_u8x16_i16x8(denominator);
    __m128i deno_hi = _zeroExtendHi_u8x16_i16x8(denominator);

    __m128 deno_lolo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(deno_lo) ); 
    __m128 deno_lohi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(deno_lo) );
    __m128 deno_hilo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(deno_hi) ); 
    __m128 deno_hihi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(deno_hi) );

    __m128i quot_lolo = _mm_cvttps_epi32( _mm_div_ps(num_lolo, deno_lolo) );
    __m128i quot_lohi = _mm_cvttps_epi32( _mm_div_ps(num_lohi, deno_lohi) );
    __m128i quot_hilo = _mm_cvttps_epi32( _mm_div_ps(num_hilo, deno_hilo) );
    __m128i quot_hihi = _mm_cvttps_epi32( _mm_div_ps(num_hihi, deno_hihi) );

    // Can't saturate as `u8 / u8 => u8`
    __m128i res_lo = _mm_packs_epi32(quot_lolo, quot_lohi);
    __m128i res_hi = _mm_packs_epi32(quot_hilo, quot_hihi);

    return _mm_packus_epi16(res_lo, res_hi);
}

// Divides each signed 8-bit integers by the corresponding signed 8-bit divisor
__m128i _div_i8x16(__m128i numerator, __m128i denominator) {
    __m128i nume_lo_padded = _mm_unpacklo_epi8(_mm_setzero_si128(), numerator);
    __m128i nume_hi_padded = _mm_unpackhi_epi8(_mm_setzero_si128(), numerator);

    // (signed)num_0_0_0 >> 24 = signExtend_i32(num)
    __m128i nume_i32_lolo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), nume_lo_padded), 24);
    __m128i nume_i32_lohi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), nume_lo_padded), 24);
    __m128i nume_i32_hilo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), nume_hi_padded), 24);
    __m128i nume_i32_hihi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), nume_hi_padded), 24);


    __m128i denom_lo_padded = _mm_unpacklo_epi8(_mm_setzero_si128(), denominator);
    __m128i denom_hi_padded = _mm_unpackhi_epi8(_mm_setzero_si128(), denominator);

    __m128i denom_i32_lolo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), denom_lo_padded), 24);
    __m128i denom_i32_lohi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), denom_lo_padded), 24);
    __m128i denom_i32_hilo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), denom_hi_padded), 24);
    __m128i denom_i32_hihi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), denom_hi_padded), 24);


    __m128 quot_lolo = _mm_div_ps( _mm_cvtepi32_ps(nume_i32_lolo), _mm_cvtepi32_ps(denom_i32_lolo) );
    __m128 quot_lohi = _mm_div_ps( _mm_cvtepi32_ps(nume_i32_lohi), _mm_cvtepi32_ps(denom_i32_lohi) );
    __m128 quot_hilo = _mm_div_ps( _mm_cvtepi32_ps(nume_i32_hilo), _mm_cvtepi32_ps(denom_i32_hilo) );
    __m128 quot_hihi = _mm_div_ps( _mm_cvtepi32_ps(nume_i32_hihi), _mm_cvtepi32_ps(denom_i32_hihi) );

    // Can't saturate as `i8 / i8 => i8`
    __m128i quot_lo = _mm_packs_epi32( _mm_cvttps_epi32(quot_lolo), _mm_cvttps_epi32(quot_lohi) );
    __m128i quot_hi = _mm_packs_epi32( _mm_cvttps_epi32(quot_hilo), _mm_cvttps_epi32(quot_hihi) );

    // This can saturate when (INT8_MAX / -1), but that is undefined behavior anyway
    return _mm_packs_epi16(quot_lo, quot_hi);
}

// Calculate the remainder (n%d) after dividing each unsigned 8-bit integers by the corresponding unsigned 8-bit divisor
// NOTE: `modP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _mod_u8x16(__m128i numerator, __m128i denominator) {
    // Convert numerators to floats (zero extend 4x, convert i32 f32)
    __m128i num_lo = _zeroExtendLo_u8x16_i16x8(numerator);
    __m128i num_hi = _zeroExtendHi_u8x16_i16x8(numerator);

    __m128 num_lolo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(num_lo) ); 
    __m128 num_lohi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(num_lo) );
    __m128 num_hilo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(num_hi) );
    __m128 num_hihi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(num_hi) );

    // Convert denominators to floats (zero extend 4x, convert i32 f32)
    __m128i deno_lo = _zeroExtendLo_u8x16_i16x8(denominator);
    __m128i deno_hi = _zeroExtendHi_u8x16_i16x8(denominator);

    __m128 deno_lolo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(deno_lo) ); 
    __m128 deno_lohi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(deno_lo) );
    __m128 deno_hilo = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(deno_hi) ); 
    __m128 deno_hihi = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(deno_hi) );

    // Use float division, then convert back to i16
    __m128i quot_lolo = _mm_cvttps_epi32( _mm_div_ps(num_lolo, deno_lolo) );
    __m128i quot_lohi = _mm_cvttps_epi32( _mm_div_ps(num_lohi, deno_lohi) );
    __m128i quot_hilo = _mm_cvttps_epi32( _mm_div_ps(num_hilo, deno_hilo) );
    __m128i quot_hihi = _mm_cvttps_epi32( _mm_div_ps(num_hihi, deno_hihi) );

    __m128i quot_lo = _mm_packs_epi32(quot_lolo, quot_lohi);
    __m128i quot_hi = _mm_packs_epi32(quot_hilo, quot_hihi);

    // n % d = n - (n / d) * d      [no u8 multiply]
    __m128i mod_lo = _mm_sub_epi16(num_lo, _mm_mullo_epi16(quot_lo, deno_lo));
    __m128i mod_hi = _mm_sub_epi16(num_hi, _mm_mullo_epi16(quot_hi, deno_hi));

    // Can't saturate as `u8 % u8 => u8`
    return _mm_packus_epi16(mod_lo, mod_hi);
}

struct sseCom_divMagic_u8 {__m128i LO_EL; __m128i HI_EL;};

// Creates half of the magic numbers used in the 8-bit division p function
// NOTE: The divisor has to be greater than 1
struct sseCom_divMagic_u8 _getDivMagic_u8x16(__m128i divisor) {
    __m128i div_lo = _zeroExtendLo_u8x16_i16x8(divisor);
    __m128i div_hi = _zeroExtendHi_u8x16_i16x8(divisor);

    struct sseCom_divMagic_u8 magicNums;
    magicNums.LO_EL = _mm_add_epi16(_div_u16x8(_mm_set1_epi16(UINT16_MAX), div_lo), _mm_set1_epi16(1));
    magicNums.HI_EL = _mm_add_epi16(_div_u16x8(_mm_set1_epi16(UINT16_MAX), div_hi), _mm_set1_epi16(1));

    return magicNums;
}

// Creates the magic numbers used in the 8-bit division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u8 _getDivMagic_set_u8x16(
    uint8_t divisor16, uint8_t divisor15, uint8_t divisor14, uint8_t divisor13,
    uint8_t divisor12, uint8_t divisor11, uint8_t divisor10, uint8_t divisor9,
    uint8_t divisor8,  uint8_t divisor7,  uint8_t divisor6,  uint8_t divisor5,
    uint8_t divisor4,  uint8_t divisor3,  uint8_t divisor2,  uint8_t divisor1
    ) {

    struct sseCom_divMagic_u8 magicNums;

    magicNums.LO_EL = _mm_set_epi16(
        (uint16_t)UINT16_MAX / divisor8 + 1,
        (uint16_t)UINT16_MAX / divisor7 + 1,
        (uint16_t)UINT16_MAX / divisor6 + 1,
        (uint16_t)UINT16_MAX / divisor5 + 1,
        (uint16_t)UINT16_MAX / divisor4 + 1,
        (uint16_t)UINT16_MAX / divisor3 + 1,
        (uint16_t)UINT16_MAX / divisor2 + 1,
        (uint16_t)UINT16_MAX / divisor1 + 1
    );

    magicNums.HI_EL = _mm_set_epi16(
        (uint16_t)UINT16_MAX / divisor16 + 1,
        (uint16_t)UINT16_MAX / divisor15 + 1,
        (uint16_t)UINT16_MAX / divisor14 + 1,
        (uint16_t)UINT16_MAX / divisor13 + 1,
        (uint16_t)UINT16_MAX / divisor12 + 1,
        (uint16_t)UINT16_MAX / divisor11 + 1,
        (uint16_t)UINT16_MAX / divisor10 + 1,
        (uint16_t)UINT16_MAX / divisor9  + 1
    );

    return magicNums;
}

// Creates the magic numbers used in the 8-bit division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u8 _getDivMagic_setr_u8x16(
    uint8_t divisor1,  uint8_t divisor2,  uint8_t divisor3,  uint8_t divisor4,
    uint8_t divisor5,  uint8_t divisor6,  uint8_t divisor7,  uint8_t divisor8,
    uint8_t divisor9,  uint8_t divisor10, uint8_t divisor11, uint8_t divisor12,
    uint8_t divisor13, uint8_t divisor14, uint8_t divisor15,  uint8_t divisor16
) {
    return _getDivMagic_set_u8x16(
        divisor1, divisor2,  divisor3,  divisor4,  divisor5,  divisor6,  divisor7,  divisor8,
        divisor9, divisor10, divisor11, divisor12, divisor13, divisor14, divisor15, divisor16
    );
}

// Creates the magic numbers used in the division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u8 _getDivMagic_set1_u8x16(uint8_t divisor) {
    uint16_t magicSingle = UINT16_MAX / divisor + 1;

    struct sseCom_divMagic_u8 magicNums = 
        {.LO_EL = _mm_set1_epi16(magicSingle), .HI_EL = _mm_set1_epi16(magicSingle)};
    
    return magicNums;
}

// Divides 16 unsigned 8-bit integers using a precomputed magic number
__m128i _divP_u8x16(__m128i dividend, struct sseCom_divMagic_u8 *magic) {
    // Technique from: arXiv:1902.01961 (Lemire et al, 2019) 
    // (u8)(n / d) = (u16)(UINT16_MAX / d + 1) * (u32)n >> 16

    __m128i toDiv_lo = _zeroExtendLo_u8x16_i16x8(dividend); 
    __m128i toDiv_hi = _zeroExtendHi_u8x16_i16x8(dividend);

    // Since a 8x16 mul results in a 24-bit product, the odd elements must be zero
    // [0, res7, ..., 0, res1, 0, res0]
    __m128i quotentLo = _mm_mulhi_epu16(toDiv_lo, magic->LO_EL);
    __m128i quotentHi = _mm_mulhi_epu16(toDiv_hi, magic->HI_EL);

    // `_trunc` masks out the upper 8-bits, so this is faster
    return _mm_packus_epi16(quotentLo, quotentHi);
}

// Calculates the modulo (a % d) of 16 unsigned 8-bit intergers using a precomputed magic number
__m128i _modP_u8x16(__m128i u8_toModulo, struct sseCom_divMagic_u8 *magic, __m128i u8_divisors) {
    // Technique from: arXiv:1902.01961 (Lemire et al, 2019) 
    // n % d = HI16( LO16(n * M) * d )
    // Faster than "n - DIV(n, M) * d"

    __m128i toMod_lo = _zeroExtendLo_u8x16_i16x8(u8_toModulo);
    __m128i toMod_hi = _zeroExtendHi_u8x16_i16x8(u8_toModulo);

    __m128i divisor_lo = _zeroExtendLo_u8x16_i16x8(u8_divisors);
    __m128i divisor_hi = _zeroExtendHi_u8x16_i16x8(u8_divisors);

    __m128i loBits_lo = _mm_mullo_epi16(toMod_lo, magic->LO_EL);
    __m128i loBits_hi = _mm_mullo_epi16(toMod_hi, magic->HI_EL);

    // Since divisor was zero extended, the mul product is 24-bits (8x16)
    // [0, res7, ..., 0, res1, 0, res0]
    __m128i resultLo_inEven = _mm_mulhi_epu16(loBits_lo, divisor_lo);
    __m128i resultHi_inEven = _mm_mulhi_epu16(loBits_hi, divisor_hi);

    // `_trunc` masks out the upper 8-bits, so this is faster
    return _mm_packus_epi16(resultLo_inEven, resultHi_inEven);
}


// ====== 16-bit ======

// Divides each unsigned 16-bit integers by the corresponding unsigned 16-bit divisor
// NOTE: `divP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _div_u16x8(__m128i numerator, __m128i denominator) {
    __m128 nume_lo_flt = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(numerator) );
    __m128 denom_lo_flt = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(denominator) );

    __m128 nume_hi_flt = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(numerator) );
    __m128 denom_hi_flt = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(denominator) );

    __m128 quot_lo_flt = _mm_div_ps(nume_lo_flt, denom_lo_flt);
    __m128 quot_hi_flt = _mm_div_ps(nume_hi_flt, denom_hi_flt);

    __m128i quot_lo_int = _mm_cvttps_epi32(quot_lo_flt);
    __m128i quot_hi_int = _mm_cvttps_epi32(quot_hi_flt);

    return _trunc_u32x4_u16x8(quot_lo_int, quot_hi_int);
}

// Divides each signed 16-bit integers by the corresponding signed 16-bit divisor
__m128i _div_i16x8(__m128i numerator, __m128i denominator) {
    __m128 nume_lo_flt = _mm_cvtepi32_ps( _signExtendLo_i16x8_i32x4(numerator) );
    __m128 denom_lo_flt = _mm_cvtepi32_ps( _signExtendLo_i16x8_i32x4(denominator) );

    __m128 nume_hi_flt = _mm_cvtepi32_ps( _signExtendHi_i16x8_i32x4(numerator) );
    __m128 denom_hi_flt = _mm_cvtepi32_ps( _signExtendHi_i16x8_i32x4(denominator) );

    __m128 quot_lo_flt = _mm_div_ps(nume_lo_flt, denom_lo_flt);
    __m128 quot_hi_flt = _mm_div_ps(nume_hi_flt, denom_hi_flt);

    __m128i quot_lo_int = _mm_cvttps_epi32(quot_lo_flt);
    __m128i quot_hi_int = _mm_cvttps_epi32(quot_hi_flt);

    // This can saturate when (INT16_MAX / -1), but that is undefined behavior anyway
    return _mm_packs_epi32(quot_lo_int, quot_hi_int);
}

// Calculate the remainder (n%d) after dividing each unsigned 16-bit integers by the corresponding unsigned 16-bit divisor
// NOTE: `modP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _mod_u16x8(__m128i numerator, __m128i denominator) {
    __m128i quotient = _div_u16x8(numerator, denominator);

    return _mm_sub_epi16(numerator, _mm_mullo_epi16(quotient, denominator));
}


/*
    Normally a 16-bit reciprocal isn't accurate enough to calculate a quotient
    ...but `UMAX_16 / d` can only be off by -1 (when "remainder" >= d) [verified u8 & u16]

    ModP/divP performs similar to double-width but precomp only needs single width div (>2x faster!)
*/
struct sseCom_divMagic_u16 {__m128i RECIP; __m128i DENOM;};


// Creates the magic numbers used in the `divP` function
struct sseCom_divMagic_u16 _getDivMagic_u16x8(__m128i divisors) {
    struct sseCom_divMagic_u16 MAGIC_NUMS = {
        .RECIP = _div_u16x8(_setone_i128(), divisors),
        .DENOM = divisors
    };

    return MAGIC_NUMS;
}

// Creates the magic numbers used in the division p function
SSECOM_INLINE struct sseCom_divMagic_u16 _getDivMagic_set_u16x8(
    uint16_t divisor8, uint16_t divisor7, uint16_t divisor6, uint16_t divisor5,
    uint16_t divisor4, uint16_t divisor3, uint16_t divisor2, uint16_t divisor1
    ) {
    
    uint16_t rcp1 = UINT16_MAX / divisor1;
    uint16_t rcp2 = UINT16_MAX / divisor2;
    uint16_t rcp3 = UINT16_MAX / divisor3;
    uint16_t rcp4 = UINT16_MAX / divisor4;
    uint16_t rcp5 = UINT16_MAX / divisor5;
    uint16_t rcp6 = UINT16_MAX / divisor6;
    uint16_t rcp7 = UINT16_MAX / divisor7;
    uint16_t rcp8 = UINT16_MAX / divisor8;
    
    struct sseCom_divMagic_u16 MAGIC_NUMS;

    MAGIC_NUMS.RECIP = _mm_set_epi16(
        rcp8, rcp7, rcp6, rcp5, rcp4, rcp3, rcp2, rcp1
    );

    MAGIC_NUMS.DENOM = _mm_set_epi16(
        divisor8, divisor7, divisor6, divisor5,
        divisor4, divisor3, divisor2, divisor1
    );

    return MAGIC_NUMS;
}

// Creates the magic numbers used in the division p function
SSECOM_INLINE struct sseCom_divMagic_u16 _getDivMagic_setr_u16x8(
    uint16_t divisor1, uint16_t divisor2, uint16_t divisor3, uint16_t divisor4,
    uint16_t divisor5, uint16_t divisor6, uint16_t divisor7, uint16_t divisor8
) {
    return _getDivMagic_set_u16x8(
        divisor1, divisor2,  divisor3,  divisor4,  divisor5,  divisor6,  divisor7,  divisor8
    );
}

// Creates the magic numbers used in the division p function
SSECOM_INLINE struct sseCom_divMagic_u16 _getDivMagic_set1_u16x8(uint16_t divisor) {
    struct sseCom_divMagic_u16 MAGIC_NUM = {
        .RECIP = _mm_set1_epi16(UINT16_MAX / divisor), .DENOM =  _mm_set1_epi16(divisor)
    };
    
    return MAGIC_NUM;
}

// Divides 8 unsigned 16-bit integers using a precomputed magic number
__m128i _divP_u16x8(__m128i numerator, struct sseCom_divMagic_u16 *MAGIC) {
    // u16(n / d) = n * (MAX_16 / d) + (remainder > d)

    __m128i almostQuot = _mm_mulhi_epu16(numerator, MAGIC->RECIP);
    __m128i almostRem = _mm_sub_epi16(numerator, _mm_mullo_epi16(almostQuot, MAGIC->DENOM));

    // If `rem >= d`, the quotient was one off
    return _mm_sub_epi16(almostQuot, _cmpGrtEq_u16x8(almostRem, MAGIC->DENOM));
}

// Calculates the modulo (n % d) of 8 unsigned 16-bit intergers using a precomputed magic number
__m128i _modP_u16x8(__m128i numerator, struct sseCom_divMagic_u16 *MAGIC) {
    __m128i almostQuot = _mm_mulhi_epu16(numerator, MAGIC->RECIP);
    __m128i almostRem = _mm_sub_epi16(numerator, _mm_mullo_epi16(almostQuot, MAGIC->DENOM));
    
    // Unsigned underflow is always larger
    __m128i dIfQuotTooSmall = _mm_subs_epu16(almostRem, _mm_sub_epi16(almostRem, MAGIC->DENOM));

    // If `rem >= d`, the quotient was one off
    return _mm_sub_epi16(almostRem, dIfQuotTooSmall);
}


// ====== 32-bit ======

// This version is needed as inlining inside `mod_u32` is much faster than a function call
// well still prevents `div_u32` from always inlining
SSECOM_INLINE __m128i _sseComInternal_div_u32x4(__m128i numerator, __m128i denominator) {
    __m128d nume_lo_flt =  _convertLo_u32x4_f64x2(numerator);
    __m128d nume_hi_flt =  _convertHi_u32x4_f64x2(numerator);

    __m128d denom_lo_flt = _convertLo_u32x4_f64x2(denominator);
    __m128d denom_hi_flt = _convertHi_u32x4_f64x2(denominator);

    __m128d quot_lo_flt = _mm_div_pd(nume_lo_flt, denom_lo_flt);
    __m128d quot_hi_flt = _mm_div_pd(nume_hi_flt, denom_hi_flt);


    #if 0
    // Full range method (slower, but no overflow)
    __m128i quot_lo_int = _convert_f64x2_u32x4(quot_lo_flt);
    __m128i quot_hi_int = _convert_f64x2_u32x4(quot_hi_flt);

    return _mm_unpacklo_epi64(quot_lo_int, quot_hi_int);
    #else

    // This conversion may overflow
    __m128i quot_lo_int = _mm_cvttpd_epi32(quot_lo_flt);
    __m128i quot_hi_int = _mm_cvttpd_epi32(quot_hi_flt);
    __m128i quotentMayOverflow = _mm_unpacklo_epi64(quot_lo_int, quot_hi_int);

    // Range: [0, INT32_MAX] (MSB=0), all larger values overflow to "80000000H" (MSB=1)
    __m128i numeIfOverflow = _mm_and_si128(numerator, _mm_srai_epi32(quotentMayOverflow, 31));

    // Overflow when: [n/1, n>INT32_MAX], so `n` is unchanged and has its MSB set
    return _mm_or_si128(quotentMayOverflow, numeIfOverflow);
    
    #endif
}

// Divides each unsigned 32-bit integers by the corresponding unsigned 32-bit divisor
// NOTE: `divP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _div_u32x4(__m128i numerator, __m128i denominator) {
    return _sseComInternal_div_u32x4(numerator, denominator);
}

// Divides each signed 32-bit integers by the corresponding signed 32-bit divisor
__m128i _div_i32x4(__m128i numerator, __m128i denominator) {
    __m128d nume_lo_dbl = _mm_cvtepi32_pd(numerator);
    __m128d nume_hi_dbl = _mm_cvtepi32_pd(_mm_bsrli_si128(numerator, 8));

    __m128d denom_lo_dbl = _mm_cvtepi32_pd(denominator);
    __m128d denom_hi_dbl = _mm_cvtepi32_pd(_mm_bsrli_si128(denominator, 8));

    __m128i quot_lo_int = _mm_cvttpd_epi32(_mm_div_pd(nume_lo_dbl, denom_lo_dbl));
    __m128i quot_hi_int = _mm_cvttpd_epi32(_mm_div_pd(nume_hi_dbl, denom_hi_dbl));

    // Converted int32 is placed in lower two elements
    return _mm_unpackhi_epi64(quot_lo_int, quot_hi_int);
}

// Calculate the remainder (n%d) after dividing each unsigned 32-bit integers by the corresponding unsigned 32-bit divisor
// NOTE: `modP` is much faster if you repeatedly reuse the same divisor vector or it is known at compile time
__m128i _mod_u32x4(__m128i numerator, __m128i denominator) {
    __m128i quotent = _sseComInternal_div_u32x4(numerator, denominator);

    // n % d = n - (n / d) * d
    return _mm_sub_epi32(numerator, _mulLo_u32x4(quotent, denominator));
}



/*  The only 32-bit support we have is `mul_epu32` which is a gives a 64-bit result
     and acts the low 32-bit element in each 64-bit lane (the "even" elements).

    We calculate: mulhi(a32, b64) = [mul_epu32(a32, b.hi) + (mul_epu32(a32, b.lo) >> 32)] >> 32
    Storing the hi32 bits seperatly saves on shifting the magic number
*/
struct sseCom_divMagic_u32 {__m128i lo32_even; __m128i hi32_even; __m128i lo32_odd; __m128i hi32_odd;};


// Creates the magic numbers used in the `divP` function
// NOTE: The divisor has to be greater than 1
struct sseCom_divMagic_u32 _getDivMagic_u32x4(__m128i divisor) {
    __m128i div_even = _mm_and_si128(divisor, _mm_set1_epi64x(UINT32_MAX));
    __m128i div_odd = _mm_srli_epi64(divisor, 32);

    // M = (DWORD_MAX / d) + 1
    __m128i magic_even = _mm_add_epi64( _div_u64x2(_mm_set1_epi64x(UINT64_MAX), div_even), _mm_set1_epi64x(1) );
    __m128i magic_odd =  _mm_add_epi64( _div_u64x2(_mm_set1_epi64x(UINT64_MAX), div_odd), _mm_set1_epi64x(1) );

    struct sseCom_divMagic_u32 MAGIC_NUMS = {
        .lo32_even = magic_even,
        .hi32_even = _mm_srli_epi64(magic_even, 32),

        .lo32_odd  = magic_odd,
        .hi32_odd  = _mm_srli_epi64(magic_odd, 32)
    };
    return MAGIC_NUMS;
}


// Creates the magic numbers used in the division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u32 _getDivMagic_set_u32x4(uint32_t divisor4, uint32_t divisor3, uint32_t divisor2, uint32_t divisor1) {
    uint64_t magic1 = UINT64_MAX / divisor1 + 1;
    uint64_t magic2 = UINT64_MAX / divisor2 + 1;
    uint64_t magic3 = UINT64_MAX / divisor3 + 1;
    uint64_t magic4 = UINT64_MAX / divisor4 + 1;

    struct sseCom_divMagic_u32 magicVec = {
        .lo32_even = _mm_set_epi64x(magic4,       magic1),
        .hi32_even = _mm_set_epi64x(magic4 >> 32, magic1 >> 32),

        .lo32_odd  = _mm_set_epi64x(magic3,       magic2),
        .hi32_odd  = _mm_set_epi64x(magic3 >> 32, magic2 >> 32)
    };
    return magicVec;
}

// Creates the magic numbers used in the division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u32 _getDivMagic_setr_u32x4(uint32_t divisor1, uint32_t divisor2, uint32_t divisor3, uint32_t divisor4) {
    return _getDivMagic_set_u32x4(divisor1, divisor2, divisor3, divisor4);
}

// Creates the magic numbers used in the division p function
// NOTE: The divisor has to be greater than 1
SSECOM_INLINE struct sseCom_divMagic_u32 _getDivMagic_set1_u32x4(uint32_t divisor) {
    uint64_t magic = UINT64_MAX / divisor + 1;

    __m128i loBits = _mm_set1_epi64x(magic);
    __m128i hiBits = _mm_set1_epi64x(magic >> 32);

    struct sseCom_divMagic_u32 MAGIC_NUMS = {
        .lo32_even = loBits, .hi32_even = hiBits,
        .lo32_odd  = loBits, .hi32_odd  = hiBits
    };
    return MAGIC_NUMS;
}


// Divides 4 unsigned 32-bit integers using a precomputed magic number
__m128i _divP_u32x4(__m128i numerator, struct sseCom_divMagic_u32 *magic) {
    // Technique from: arXiv:1902.01961 (Lemire et al, 2019)
    // (u32)(n / d) = (UINT64_MAX / d + 1) * n >> 64

    // mulhi(a_u32, b_u64) = ( mulfull(a_u32, b.lo) >> 32) + mulfull(a_u32, b.hi) ) >> 32
    __m128i even_lo = _mm_mul_epu32(numerator, magic->lo32_even);
    __m128i even_hi = _mm_mul_epu32(numerator, magic->hi32_even);
    __m128i even_res_hi64 = _mm_add_epi64(_mm_srli_epi64(even_lo, 32), even_hi);

    __m128i odd_numerator = _mm_srli_epi64(numerator, 32);

    __m128i odd_lo = _mm_mul_epu32(odd_numerator, magic->lo32_odd);
    __m128i odd_hi = _mm_mul_epu32(odd_numerator, magic->hi32_odd);
    __m128i odd_res_hi64 = _mm_add_epi64(_mm_srli_epi64(odd_lo, 32), odd_hi);

    // We only need the high 32-bits of each result
    __m128i quotent_3_1_2_0 = _shuffleLoHi_i32x4(even_res_hi64, _MM_SHUFHALF(3, 1), odd_res_hi64, _MM_SHUFHALF(3, 1));
    return _mm_shuffle_epi32(quotent_3_1_2_0, _MM_SHUFFLE(3, 1, 2, 0));
}

// Calculates the modulo (n % d) of 4 unsigned 32-bit intergers using a precomputed magic number
__m128i _modP_u32x4(__m128i numerator, struct sseCom_divMagic_u32 *magic, __m128i divisor) {
    // Faster to do: n % d = [n - (n / d) * d], than [mulhi( mullo(M, n), d)]
    // As low 32x64 mul needs: 4 `mul_epu32`, 2 adds, 3 shifts
    // ...and the high mul need two additional shifts (maigc precomputes them)
    __m128i quotent = _divP_u32x4(numerator, magic);

    return _mm_sub_epi32(numerator, _mulLo_u32x4(quotent, divisor));
}


// ====== 64-bit ======

// !!EXPENSIVE AVOID USING!! Divides each unsigned 64-bit integers by the corresponding unsigned 64-bit divisor
__m128i _div_u64x2(__m128i numerator, __m128i denominator) {
    // Statement order use to produce better asm output
    uint64_t nume_lo = _mm_cvtsi128_si64(numerator);
    uint64_t denom_lo = _mm_cvtsi128_si64(denominator);

    __m128i nume_hi_vec = _mm_unpackhi_epi64(numerator, numerator);
    __m128i denom_hi_vec = _mm_unpackhi_epi64(denominator, denominator);

    uint64_t quot_lo = nume_lo / denom_lo;

    uint64_t nume_hi = _mm_cvtsi128_si64(nume_hi_vec);
    uint64_t denom_hi = _mm_cvtsi128_si64(denom_hi_vec);

    __m128i quot_lo_vec = _mm_cvtsi64_si128(quot_lo);

    uint64_t quot_hi = nume_hi / denom_hi;
    __m128i quot_hi_vec = _mm_cvtsi64_si128(quot_hi);

    return _mm_unpacklo_epi64(quot_lo_vec, quot_hi_vec);
}

// !!EXPENSIVE AVOID USING!! Divides each signed 64-bit integers by the corresponding signed 64-bit divisor
__m128i _div_i64x2(__m128i numerator, __m128i denominator) {
    // Statement order use to produce better asm output
    int64_t nume_lo = _mm_cvtsi128_si64(numerator);
    int64_t denom_lo = _mm_cvtsi128_si64(denominator);

    __m128i nume_hi_vec = _mm_unpackhi_epi64(numerator, numerator);
    __m128i denom_hi_vec = _mm_unpackhi_epi64(denominator, denominator);

    int64_t quot_lo = nume_lo / denom_lo;

    int64_t nume_hi = _mm_cvtsi128_si64(nume_hi_vec);
    int64_t denom_hi = _mm_cvtsi128_si64(denom_hi_vec);

    __m128i quot_lo_vec = _mm_cvtsi64_si128(quot_lo);

    int64_t quot_hi = nume_hi / denom_hi;
    __m128i quot_hi_vec = _mm_cvtsi64_si128(quot_hi);

    return _mm_unpacklo_epi64(quot_lo_vec, quot_hi_vec);
}

// !!EXPENSIVE!! Calculate the remainder (n%d) after dividing each unsigned 64-bit integers by the corresponding unsigned 64-bit divisor
__m128i _mod_u64x2(__m128i numerator, __m128i denominator) {
    // Statement order use to produce better asm output
    uint64_t nume_lo = _mm_cvtsi128_si64(numerator);
    uint64_t denom_lo = _mm_cvtsi128_si64(denominator);

    __m128i nume_hi_vec = _mm_unpackhi_epi64(numerator, numerator);
    __m128i denom_hi_vec = _mm_unpackhi_epi64(denominator, denominator);

    uint64_t mod_lo = nume_lo % denom_lo;

    uint64_t nume_hi = _mm_cvtsi128_si64(nume_hi_vec);
    uint64_t denom_hi = _mm_cvtsi128_si64(denom_hi_vec);

    __m128i mod_lo_vec = _mm_cvtsi64_si128(mod_lo);

    uint64_t mod_hi = nume_hi % denom_hi;
    __m128i mod_hi_vec = _mm_cvtsi64_si128(mod_hi);

    return _mm_unpacklo_epi64(mod_lo_vec, mod_hi_vec);
}
