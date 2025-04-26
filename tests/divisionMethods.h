#pragma once

#include <stdint.h>
#include <emmintrin.h> // SSE2

#include "../include/sseComplete.h"
#include "_perfCommon.h"


// ====== 8-bit ======

static uint16_t MUL_U8_MAGIC_LUT[] = {
    // Special cases for div by zero and one
    0x0000        , 0xFFFF        , 0xFFFF/  2 + 1, 0xFFFF/  3 + 1, 0xFFFF/  4 + 1, 0xFFFF/  5 + 1, 0xFFFF/  6 + 1, 0xFFFF/  7 + 1, 
    0xFFFF/  8 + 1, 0xFFFF/  9 + 1, 0xFFFF/ 10 + 1, 0xFFFF/ 11 + 1, 0xFFFF/ 12 + 1, 0xFFFF/ 13 + 1, 0xFFFF/ 14 + 1, 0xFFFF/ 15 + 1, 
    0xFFFF/ 16 + 1, 0xFFFF/ 17 + 1, 0xFFFF/ 18 + 1, 0xFFFF/ 19 + 1, 0xFFFF/ 20 + 1, 0xFFFF/ 21 + 1, 0xFFFF/ 22 + 1, 0xFFFF/ 23 + 1,
    0xFFFF/ 24 + 1, 0xFFFF/ 25 + 1, 0xFFFF/ 26 + 1, 0xFFFF/ 27 + 1, 0xFFFF/ 28 + 1, 0xFFFF/ 29 + 1, 0xFFFF/ 30 + 1, 0xFFFF/ 31 + 1,
    0xFFFF/ 32 + 1, 0xFFFF/ 33 + 1, 0xFFFF/ 34 + 1, 0xFFFF/ 35 + 1, 0xFFFF/ 36 + 1, 0xFFFF/ 37 + 1, 0xFFFF/ 38 + 1, 0xFFFF/ 39 + 1,
    0xFFFF/ 40 + 1, 0xFFFF/ 41 + 1, 0xFFFF/ 42 + 1, 0xFFFF/ 43 + 1, 0xFFFF/ 44 + 1, 0xFFFF/ 45 + 1, 0xFFFF/ 46 + 1, 0xFFFF/ 47 + 1,
    0xFFFF/ 48 + 1, 0xFFFF/ 49 + 1, 0xFFFF/ 50 + 1, 0xFFFF/ 51 + 1, 0xFFFF/ 52 + 1, 0xFFFF/ 53 + 1, 0xFFFF/ 54 + 1, 0xFFFF/ 55 + 1,
    0xFFFF/ 56 + 1, 0xFFFF/ 57 + 1, 0xFFFF/ 58 + 1, 0xFFFF/ 59 + 1, 0xFFFF/ 60 + 1, 0xFFFF/ 61 + 1, 0xFFFF/ 62 + 1, 0xFFFF/ 63 + 1,
    0xFFFF/ 64 + 1, 0xFFFF/ 65 + 1, 0xFFFF/ 66 + 1, 0xFFFF/ 67 + 1, 0xFFFF/ 68 + 1, 0xFFFF/ 69 + 1, 0xFFFF/ 70 + 1, 0xFFFF/ 71 + 1,
    0xFFFF/ 72 + 1, 0xFFFF/ 73 + 1, 0xFFFF/ 74 + 1, 0xFFFF/ 75 + 1, 0xFFFF/ 76 + 1, 0xFFFF/ 77 + 1, 0xFFFF/ 78 + 1, 0xFFFF/ 79 + 1,
    0xFFFF/ 80 + 1, 0xFFFF/ 81 + 1, 0xFFFF/ 82 + 1, 0xFFFF/ 83 + 1, 0xFFFF/ 84 + 1, 0xFFFF/ 85 + 1, 0xFFFF/ 86 + 1, 0xFFFF/ 87 + 1,
    0xFFFF/ 88 + 1, 0xFFFF/ 89 + 1, 0xFFFF/ 90 + 1, 0xFFFF/ 91 + 1, 0xFFFF/ 92 + 1, 0xFFFF/ 93 + 1, 0xFFFF/ 94 + 1, 0xFFFF/ 95 + 1,
    0xFFFF/ 96 + 1, 0xFFFF/ 97 + 1, 0xFFFF/ 98 + 1, 0xFFFF/ 99 + 1, 0xFFFF/100 + 1, 0xFFFF/101 + 1, 0xFFFF/102 + 1, 0xFFFF/103 + 1,
    0xFFFF/104 + 1, 0xFFFF/105 + 1, 0xFFFF/106 + 1, 0xFFFF/107 + 1, 0xFFFF/108 + 1, 0xFFFF/109 + 1, 0xFFFF/110 + 1, 0xFFFF/111 + 1,
    0xFFFF/112 + 1, 0xFFFF/113 + 1, 0xFFFF/114 + 1, 0xFFFF/115 + 1, 0xFFFF/116 + 1, 0xFFFF/117 + 1, 0xFFFF/118 + 1, 0xFFFF/119 + 1,
    0xFFFF/120 + 1, 0xFFFF/121 + 1, 0xFFFF/122 + 1, 0xFFFF/123 + 1, 0xFFFF/124 + 1, 0xFFFF/125 + 1, 0xFFFF/126 + 1, 0xFFFF/127 + 1,
    0xFFFF/128 + 1, 0xFFFF/129 + 1, 0xFFFF/130 + 1, 0xFFFF/131 + 1, 0xFFFF/132 + 1, 0xFFFF/133 + 1, 0xFFFF/134 + 1, 0xFFFF/135 + 1,
    0xFFFF/136 + 1, 0xFFFF/137 + 1, 0xFFFF/138 + 1, 0xFFFF/139 + 1, 0xFFFF/140 + 1, 0xFFFF/141 + 1, 0xFFFF/142 + 1, 0xFFFF/143 + 1,
    0xFFFF/144 + 1, 0xFFFF/145 + 1, 0xFFFF/146 + 1, 0xFFFF/147 + 1, 0xFFFF/148 + 1, 0xFFFF/149 + 1, 0xFFFF/150 + 1, 0xFFFF/151 + 1,
    0xFFFF/152 + 1, 0xFFFF/153 + 1, 0xFFFF/154 + 1, 0xFFFF/155 + 1, 0xFFFF/156 + 1, 0xFFFF/157 + 1, 0xFFFF/158 + 1, 0xFFFF/159 + 1,
    0xFFFF/160 + 1, 0xFFFF/161 + 1, 0xFFFF/162 + 1, 0xFFFF/163 + 1, 0xFFFF/164 + 1, 0xFFFF/165 + 1, 0xFFFF/166 + 1, 0xFFFF/167 + 1,
    0xFFFF/168 + 1, 0xFFFF/169 + 1, 0xFFFF/170 + 1, 0xFFFF/171 + 1, 0xFFFF/172 + 1, 0xFFFF/173 + 1, 0xFFFF/174 + 1, 0xFFFF/175 + 1,
    0xFFFF/176 + 1, 0xFFFF/177 + 1, 0xFFFF/178 + 1, 0xFFFF/179 + 1, 0xFFFF/180 + 1, 0xFFFF/181 + 1, 0xFFFF/182 + 1, 0xFFFF/183 + 1,
    0xFFFF/184 + 1, 0xFFFF/185 + 1, 0xFFFF/186 + 1, 0xFFFF/187 + 1, 0xFFFF/188 + 1, 0xFFFF/189 + 1, 0xFFFF/190 + 1, 0xFFFF/191 + 1,
    0xFFFF/192 + 1, 0xFFFF/193 + 1, 0xFFFF/194 + 1, 0xFFFF/195 + 1, 0xFFFF/196 + 1, 0xFFFF/197 + 1, 0xFFFF/198 + 1, 0xFFFF/199 + 1,
    0xFFFF/200 + 1, 0xFFFF/201 + 1, 0xFFFF/202 + 1, 0xFFFF/203 + 1, 0xFFFF/204 + 1, 0xFFFF/205 + 1, 0xFFFF/206 + 1, 0xFFFF/207 + 1,
    0xFFFF/208 + 1, 0xFFFF/209 + 1, 0xFFFF/210 + 1, 0xFFFF/211 + 1, 0xFFFF/212 + 1, 0xFFFF/213 + 1, 0xFFFF/214 + 1, 0xFFFF/215 + 1,
    0xFFFF/216 + 1, 0xFFFF/217 + 1, 0xFFFF/218 + 1, 0xFFFF/219 + 1, 0xFFFF/220 + 1, 0xFFFF/221 + 1, 0xFFFF/222 + 1, 0xFFFF/223 + 1,
    0xFFFF/224 + 1, 0xFFFF/225 + 1, 0xFFFF/226 + 1, 0xFFFF/227 + 1, 0xFFFF/228 + 1, 0xFFFF/229 + 1, 0xFFFF/230 + 1, 0xFFFF/231 + 1,
    0xFFFF/232 + 1, 0xFFFF/233 + 1, 0xFFFF/234 + 1, 0xFFFF/235 + 1, 0xFFFF/236 + 1, 0xFFFF/237 + 1, 0xFFFF/238 + 1, 0xFFFF/239 + 1,
    0xFFFF/240 + 1, 0xFFFF/241 + 1, 0xFFFF/242 + 1, 0xFFFF/243 + 1, 0xFFFF/244 + 1, 0xFFFF/245 + 1, 0xFFFF/246 + 1, 0xFFFF/247 + 1,
    0xFFFF/248 + 1, 0xFFFF/249 + 1, 0xFFFF/250 + 1, 0xFFFF/251 + 1, 0xFFFF/252 + 1, 0xFFFF/253 + 1, 0xFFFF/254 + 1, 0xFFFF/255 + 1
};

// Uses a precomputed magic number
// (Also doesn't work when div by one)
NOINLINE __m128i magicDiv_u8(__m128i numerator, __m128i denominator) {
    uint8_t denom_array[16];
    uint16_t magic_nums[16];

    _mm_store_si128((__m128i*)denom_array, denominator);
    
    for (size_t i=0; i < 16; i++)
        magic_nums[i] = MUL_U8_MAGIC_LUT[denom_array[i]];

    __m128i magic_lo = _mm_loadu_si128((__m128i*)magic_nums + 0);
    __m128i magic_hi = _mm_loadu_si128((__m128i*)magic_nums + 1);

    // Correct issue where division will be incorrect when div by one
    numerator = _mm_sub_epi8(numerator, _mm_cmpeq_epi8(denominator, _mm_set1_epi8(1)));

    __m128i num_lo = _zeroExtendLo_u8x16_i16x8(numerator);
    __m128i num_hi = _zeroExtendHi_u8x16_i16x8(numerator);

    __m128i quot_lo = _mm_mulhi_epu16(num_lo, magic_lo);
    __m128i quot_hi = _mm_mulhi_epu16(num_hi, magic_hi);

    
    return _trunc_u16x8_u8x16(quot_lo, quot_hi);
}

// Deposit values into array, divide one-by-one
NOINLINE __m128i linDiv_u8(__m128i numerator, __m128i denominator) {
    uint8_t num_array[16], denom_array[16], quotients[16];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 16; i++)
        quotients[i] = num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

NOINLINE __m128i linfDiv_u8(__m128i numerator, __m128i denominator) {
    uint8_t num_array[16], denom_array[16], quotients[16];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 16; i++)
        quotients[i] = (float)num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}


/*NOINLINE __m128i recipDiv_u8(__m128i numerator, __m128i denominator) {
    uint8_t num_array[16], denom_array[16], quotients[16];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 16; i++) {
        __m128 num_lin = _mm_set_ss(num_array[i]), denom_lin = _mm_set_ss(denom_array[i]);
        
        __m128 quotent_lin = _mm_mul_ps(num_lin, _mm_rcp_ps(denom_lin));
        quotients[i] = (uint8_t)_mm_cvtss_f32(quotent_lin);
    }

    return _mm_loadu_si128((__m128i*)quotients);
}*/

NOINLINE __m128i vecDiv_u8(__m128i numerator, __m128i denominator) {
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

    __m128i res_lo = _mm_packs_epi32(quot_lolo, quot_lohi);
    __m128i res_hi = _mm_packs_epi32(quot_hilo, quot_hihi);

    return _mm_packus_epi16(res_lo, res_hi);
}



// Branchless long division
NOINLINE __m128i longDiv_u8(__m128i numerator, __m128i denominator) {
    const __m128i LSB_MASK = _mm_set1_epi8(0x01);

    __m128i quotient = _mm_setzero_si128();
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
        quotient = _mm_sub_epi8( _mm_add_epi8(quotient, quotient), isGrtEq_mask);
    }

    return quotient;
}



// ====== 16-bit ======


// Deposit values into array, divide one-by-one
NOINLINE __m128i linDiv_u16(__m128i numerator, __m128i denominator) {
    uint16_t num_array[8], denom_array[8], quotients[8];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 8; i++)
        quotients[i] = num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

// Deposit values into array, divide one-by-one UNROLLED
NOINLINE __m128i linUnrolledDiv_u16(__m128i numerator, __m128i denominator) {
    uint16_t num_array[8], denom_array[8], quotients[8];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    quotients[0] = num_array[0] / denom_array[0];
    quotients[1] = num_array[1] / denom_array[1];
    quotients[2] = num_array[2] / denom_array[2];
    quotients[3] = num_array[3] / denom_array[3];
    quotients[4] = num_array[4] / denom_array[4];
    quotients[5] = num_array[5] / denom_array[5];
    quotients[6] = num_array[6] / denom_array[6];
    quotients[7] = num_array[7] / denom_array[7];

    return _mm_loadu_si128((__m128i*)quotients);
}


// Encourages autovectorization
NOINLINE __m128i linDivf_u16(__m128i numerator, __m128i denominator) {
    uint16_t num_array[8], denom_array[8], quotients[8];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 8; i++)
        // Casting to float is much faster than doing interger div (4x faster!)
        // Due to autovectorization
        quotients[i] = (float)num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}


// Branchless long division
NOINLINE __m128i longDiv_u16(__m128i numerator, __m128i denominator) {
    const __m128i LSB_MASK = _mm_set1_epi16(0x0001);

    __m128i quotient = _mm_setzero_si128();
    __m128i remainder = _mm_setzero_si128();
    
    for (int i = 16-1; i >= 0; i--) {
        remainder = _mm_add_epi16(remainder, remainder); // << 1 (no 8-bit shift)
        // remainder =| (numerator >> i) & 0x01
        __m128i ithBit = _mm_and_si128( _mm_srl_epi16(numerator, _mm_cvtsi32_si128(i)), LSB_MASK );
        remainder = _mm_or_si128(remainder, ithBit);

        // If (remainder >= denominator) then subtract
        __m128i isGrtEq_mask = _cmpGrtEq_u16x8(remainder, denominator);
        remainder = _mm_sub_epi16(remainder, _mm_and_si128(denominator, isGrtEq_mask));

        // quotient << 1 | (mask & 1) 
        // [ num - 0xFFFF == num + 1]
        quotient = _mm_sub_epi16( _mm_add_epi16(quotient, quotient), isGrtEq_mask);
    }

    return quotient;
}




NOINLINE __m128i vecDiv_u16(__m128i numerator, __m128i denominator) {
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


NOINLINE __m128i vecRCPDiv_u16(__m128i numerator, __m128i denominator) {
    __m128 nume_lo_flt = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(numerator) );
    __m128 denom_lo_flt = _mm_cvtepi32_ps( _zeroExtendLo_u16x8_i32x4(denominator) );

    __m128 nume_hi_flt = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(numerator) );
    __m128 denom_hi_flt = _mm_cvtepi32_ps( _zeroExtendHi_u16x8_i32x4(denominator) );

    __m128 quot_lo_flt = _mm_mul_ps(nume_lo_flt, _mm_rcp_ps(denom_lo_flt));
    __m128 quot_hi_flt = _mm_mul_ps(nume_hi_flt, _mm_rcp_ps(denom_hi_flt));

    __m128i quot_lo_int = _mm_cvttps_epi32(quot_lo_flt);
    __m128i quot_hi_int = _mm_cvttps_epi32(quot_hi_flt);

    return _trunc_u32x4_u16x8(quot_lo_int, quot_hi_int);
}





// ====== 32-bit ======


// Deposit values into array, divide one-by-one
NOINLINE __m128i linDiv_u32(__m128i numerator, __m128i denominator) {
    uint32_t num_array[4], denom_array[4], quotients[4];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 4; i++)
        quotients[i] = num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

// Deposit values into array, divide one-by-one UNROLLED
NOINLINE __m128i linUnrolledDiv_u32(__m128i numerator, __m128i denominator) {
    uint32_t num_array[4], denom_array[4], quotients[4];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    quotients[0] = num_array[0] / denom_array[0];
    quotients[1] = num_array[1] / denom_array[1];
    quotients[2] = num_array[2] / denom_array[2];
    quotients[3] = num_array[3] / denom_array[3];

    return _mm_loadu_si128((__m128i*)quotients);
}

// Encourage compiler autovectorization
NOINLINE __m128i linDivf_u32(__m128i numerator, __m128i denominator) {
    uint32_t num_array[4], denom_array[4], quotients[4];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 4; i++)
        quotients[i] = (double)num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

NOINLINE __m128i vecDiv_u32(__m128i numerator, __m128i denominator) {
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




// Branchless long division
NOINLINE __m128i longDiv_u32(__m128i numerator, __m128i denominator) {
    const __m128i LSB_MASK = _mm_set1_epi32(0x0001);

    __m128i quotient = _mm_setzero_si128();
    __m128i remainder = _mm_setzero_si128();
    
    for (int i = 32-1; i >= 0; i--) {
        remainder = _mm_add_epi32(remainder, remainder); // << 1 (no 8-bit shift)
        // remainder =| (numerator >> i) & 0x01
        __m128i ithBit = _mm_and_si128( _mm_srl_epi32(numerator, _mm_cvtsi32_si128(i)), LSB_MASK );
        remainder = _mm_or_si128(remainder, ithBit);

        // If (remainder >= denominator) then subtract
        __m128i isGrtEq_mask = _cmpGrt_u32x4(remainder, denominator);
        remainder = _mm_sub_epi32(remainder, _mm_and_si128(denominator, isGrtEq_mask));

        // quotient << 1 | (mask & 1) 
        // [ num - 0xFFFF == num + 1]
        quotient = _mm_sub_epi32( _mm_add_epi32(quotient, quotient), isGrtEq_mask);
    }

    return quotient;
}





// ====== 64-bit ======

// Deposit values into array, divide one-by-one
NOINLINE __m128i linDiv_u64(__m128i numerator, __m128i denominator) {
    uint64_t num_array[2], denom_array[2], quotients[2];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 2; i++)
        quotients[i] = num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

// Is x87 div faster?
NOINLINE __m128i linDivf_u64(__m128i numerator, __m128i denominator) {
    uint64_t num_array[2], denom_array[2], quotients[2];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 2; i++)
        quotients[i] = (long double)num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}

NOINLINE __m128i vecLin_u64(__m128i numerator, __m128i denominator) {
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

// Branchless long division
NOINLINE __m128i longDiv_u64(__m128i numerator, __m128i denominator) {
    const __m128i LSB_MASK = _mm_set1_epi64x(1);

    __m128i quotient = _mm_setzero_si128();
    __m128i remainder = _mm_setzero_si128();
    
    for (int i = 64-1; i >= 0; i--) {
        remainder = _mm_add_epi64(remainder, remainder); // << 1 (no 8-bit shift)
        // remainder =| (numerator >> i) & 0x01
        __m128i ithBit = _mm_and_si128( _mm_srl_epi64(numerator, _mm_cvtsi32_si128(i)), LSB_MASK );
        remainder = _mm_or_si128(remainder, ithBit);

        // If (remainder >= denominator) then subtract
        __m128i isGrtEq_mask = _cmpGrt_u64x2(remainder, denominator);
        remainder = _mm_sub_epi64(remainder, _mm_and_si128(denominator, isGrtEq_mask));

        // quotient << 1 | (mask & 1) 
        // [ num - 0xFFFF == num + 1]
        quotient = _mm_sub_epi64( _mm_add_epi64(quotient, quotient), isGrtEq_mask);
    }

    return quotient;
}

#ifdef __INTEL_LLVM_COMPILER
    NOINLINE __m128i _svmlDiv_u8( __m128i nume, __m128i denom) { return _mm_div_epu8(nume, denom); }
    NOINLINE __m128i _svmlDiv_u16(__m128i nume, __m128i denom) { return _mm_div_epu16(nume, denom); }
    NOINLINE __m128i _svmlDiv_u32(__m128i nume, __m128i denom) { return _mm_div_epu32(nume, denom); }
    NOINLINE __m128i _svmlDiv_u64(__m128i nume, __m128i denom) { return _mm_div_epu64(nume, denom); }

    NOINLINE __m128i _svmlDiv_i8( __m128i nume, __m128i denom) { return _mm_div_epi8(nume, denom); }
    NOINLINE __m128i _svmlDiv_i16(__m128i nume, __m128i denom) { return _mm_div_epi16(nume, denom); }
    NOINLINE __m128i _svmlDiv_i32(__m128i nume, __m128i denom) { return _mm_div_epi32(nume, denom); }
    NOINLINE __m128i _svmlDiv_i64(__m128i nume, __m128i denom) { return _mm_div_epi64(nume, denom); }
#endif



// ======= Fastest signed division =======

NOINLINE __m128i vecDiv_i8(__m128i numerator, __m128i denominator) {
    __m128i nume_lo_padded = _mm_unpacklo_epi8(_mm_setzero_si128(), numerator);
    __m128i nume_hi_padded = _mm_unpackhi_epi8(_mm_setzero_si128(), numerator);
    __m128i denom_lo_padded = _mm_unpacklo_epi8(_mm_setzero_si128(), denominator);
    __m128i denom_hi_padded = _mm_unpackhi_epi8(_mm_setzero_si128(), denominator);

    // (signed)num_0_0_0 >> 24 = signExtend_i32(num)
    __m128i nume_i32_lolo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), nume_lo_padded), 24);
    __m128i nume_i32_lohi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), nume_lo_padded), 24);
    __m128i nume_i32_hilo = _mm_srai_epi32( _mm_unpacklo_epi16(_mm_setzero_si128(), nume_hi_padded), 24);
    __m128i nume_i32_hihi = _mm_srai_epi32( _mm_unpackhi_epi16(_mm_setzero_si128(), nume_hi_padded), 24);

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

    // This can saturate when (INT8_MAX / -1), but this is undefined behavior anyway
    return _mm_packs_epi16(quot_lo, quot_hi);
}

NOINLINE __m128i vecDiv_i16(__m128i numerator, __m128i denominator) {
    __m128 nume_lo_flt = _mm_cvtepi32_ps( _signExtendLo_i16x8_i32x4(numerator) );
    __m128 denom_lo_flt = _mm_cvtepi32_ps( _signExtendLo_i16x8_i32x4(denominator) );

    __m128 nume_hi_flt = _mm_cvtepi32_ps( _signExtendHi_i16x8_i32x4(numerator) );
    __m128 denom_hi_flt = _mm_cvtepi32_ps( _signExtendHi_i16x8_i32x4(denominator) );

    __m128 quot_lo_flt = _mm_div_ps(nume_lo_flt, denom_lo_flt);
    __m128 quot_hi_flt = _mm_div_ps(nume_hi_flt, denom_hi_flt);

    __m128i quot_lo_int = _mm_cvttps_epi32(quot_lo_flt);
    __m128i quot_hi_int = _mm_cvttps_epi32(quot_hi_flt);

    // This can saturate when (INT16_MAX / -1), but this is undefined behavior anyway
    return _mm_packs_epi32(quot_lo_int, quot_hi_int);
}

NOINLINE __m128i vecDiv_i32(__m128i numerator, __m128i denominator) {
    __m128d nume_lo_dbl = _mm_cvtepi32_pd(numerator);
    __m128d nume_hi_dbl = _mm_cvtepi32_pd(_mm_bsrli_si128(numerator, 8));

    __m128d denom_lo_dbl = _mm_cvtepi32_pd(denominator);
    __m128d denom_hi_dbl = _mm_cvtepi32_pd(_mm_bsrli_si128(denominator, 8));

    __m128i quot_lo_int = _mm_cvttpd_epi32(_mm_div_pd(nume_lo_dbl, denom_lo_dbl));
    __m128i quot_hi_int = _mm_cvttpd_epi32(_mm_div_pd(nume_hi_dbl, denom_hi_dbl));

    // Converted int32 is placed in lower two elements
    return _mm_unpackhi_epi64(quot_lo_int, quot_hi_int);
}

NOINLINE __m128i linDiv_i64(__m128i numerator, __m128i denominator) {
    int64_t num_array[2], denom_array[2], quotients[2];
    
    _mm_store_si128((__m128i*)num_array, numerator);
    _mm_store_si128((__m128i*)denom_array, denominator);

    
    for (size_t i=0; i < 2; i++)
        quotients[i] = num_array[i] / denom_array[i];

    return _mm_loadu_si128((__m128i*)quotients);
}