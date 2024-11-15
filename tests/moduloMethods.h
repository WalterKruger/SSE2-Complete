#pragma once

#include <stdint.h>
#include <emmintrin.h> // SSE2

#include "../include/sseComplete.h"
#include "_perfCommon.h"


// ====== 8-bit ======

// Deposit values into array, modulo one-by-one
NOINLINE __m128i linMod_u8(__m128i numerator, __m128i denominator) {
    uint8_t num_array[16], denom_array[16], quotients[16];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 16; i++)
        quotients[i] = num_array[i] % denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}


NOINLINE __m128i vecMod_u8(__m128i numerator, __m128i denominator) {
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

    return _mm_packus_epi16(mod_lo, mod_hi);
}



// Branchless long division
NOINLINE __m128i longMod_u8(__m128i numerator, __m128i denominator) {
    const __m128i LSB_MASK = _mm_set1_epi8(0x01);

    //__m128i quotient = _mm_setzero_si128();
    __m128i remainder = _mm_setzero_si128();
    
    for (int i = 8-1; i >= 0; i--) {
        remainder = _mm_add_epi8(remainder, remainder); // << 1 (no 8-bit shift)
        // remainder =| (numerator >> i) & 0x01
        // [And sized shift works as we only keep the LSB]
        __m128i ithBit = _mm_and_si128( _mm_srl_epi32(numerator, _mm_cvtsi32_si128(i)), LSB_MASK );
        remainder = _mm_or_si128(remainder, ithBit);

        // If (remainder >= denominator) then subtract denom.
        __m128i isGrtEq_mask = _cmpGrtEq_u8x16(remainder, denominator);
        remainder = _mm_sub_epi8(remainder, _mm_and_si128(denominator, isGrtEq_mask));

        // quotient << 1 | (mask & 1) 
        // [ num - 0xFF == num + 1]
        //quotient = _mm_sub_epi8( _mm_add_epi8(quotient, quotient), isGrtEq_mask);
    }

    return remainder;
}



// ====== 32-bit ======


NOINLINE __m128i linMod_u32(__m128i numerator, __m128i denominator) {
    uint32_t num_array[4], denom_array[4], quotients[4];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    for (size_t i=0; i < 4; i++)
        quotients[i] = num_array[i] % denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

NOINLINE __m128i linUnrollMod_u32(__m128i numerator, __m128i denominator) {
    uint32_t num_array[4], denom_array[4], quotients[4];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    // Compilers don't always unroll, despite being much faster
    quotients[0] = num_array[0] % denom_array[0];
    quotients[1] = num_array[1] % denom_array[1];
    quotients[2] = num_array[2] % denom_array[2];
    quotients[3] = num_array[3] % denom_array[3];

    return _mm_loadu_si128((__m128i*)quotients);
}

NOINLINE __m128i vecMod_u32(__m128i numerator, __m128i denominator) {
    #if 0
    __m128d nume_lo_flt =  _convertLo_u32x4_f64x2(numerator);
    __m128d nume_hi_flt =  _convertHi_u32x4_f64x2(numerator);
    __m128d denom_lo_flt = _convertLo_u32x4_f64x2(denominator);
    __m128d denom_hi_flt = _convertHi_u32x4_f64x2(denominator);

    __m128d quot_lo_flt = _mm_div_pd(nume_lo_flt, denom_lo_flt);
    __m128d quot_hi_flt = _mm_div_pd(nume_hi_flt, denom_hi_flt);

    // TODO: Since only need unsigned when (numerator > INT32_MAX) and (divisor == 1)
    //       It might be possible to simplify...
    __m128i quot_lo_int = _convert_f64x2_u32x4(quot_lo_flt);
    __m128i quot_hi_int = _convert_f64x2_u32x4(quot_hi_flt);

    // Each quotent vec is: [0, 0, quot1, quot0]
    __m128i quotent = _mm_unpacklo_epi64(quot_lo_int, quot_hi_int);

    // n % d = n - (n / d) * d
    return _mm_sub_epi32(numerator, _mulLo_u32x4(quotent, denominator));
    #else
    return _mod_u32x4(numerator, denominator);
    #endif
}





#ifdef __INTEL_LLVM_COMPILER
    NOINLINE __m128i _svmlMod_u8( __m128i nume, __m128i denom) { return _mm_rem_epu8(nume, denom); }
    NOINLINE __m128i _svmlMod_u32(__m128i nume, __m128i denom) { return _mm_rem_epu32(nume, denom); }
#endif