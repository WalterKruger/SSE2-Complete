#pragma once

#include <stdint.h>
#include <emmintrin.h> // SSE2

#include "../include/sseComplete.h"
#include "_perfCommon.h"

// ===== Float32 to unsigned int 32 =====

NOINLINE __m128i scaleBitCombind_f32ToU32(__m128 f32_toCvt) {
    __m128i directCvt = _mm_cvttps_epi32(f32_toCvt);
    __m128 f32_scaled = _mm_sub_ps(f32_toCvt, _mm_set1_ps(1ull << 32));
    __m128i scaledCvt = _mm_cvttps_epi32(f32_scaled);

    __m128i isNegMask = _mm_srai_epi32(directCvt, 31);
    return _mm_or_si128(_mm_and_si128(scaledCvt, isNegMask), directCvt);
}

NOINLINE __m128i compiler_f32ToU32(__m128 f32_toCvt) {
    float f32_array[4];
    _mm_store_ps(f32_array, f32_toCvt);

    return _mm_setr_epi32((uint32_t)f32_array[0], (uint32_t)f32_array[1], (uint32_t)f32_array[2], (uint32_t)f32_array[3]);
}



// ===== Float32 to signed int 64 =====


NOINLINE __m128i scalarInstruction_f32ToI64(__m128 f32_toCvt) {
    uint64_t i64_lo = _mm_cvttss_si64(f32_toCvt);
    uint64_t i64_hi = _mm_cvttss_si64(_mm_shuffle_ps(f32_toCvt, f32_toCvt, 0b01));
    
    __m128i lo_vec = _mm_cvtsi64_si128(i64_lo);
    __m128i hi_vec = _mm_cvtsi64_si128(i64_hi);

    return _mm_unpacklo_epi64(lo_vec, hi_vec);
}

NOINLINE __m128i compiler_f32ToI64(__m128 f32_toCvt) {
    float f32_array[4];
    _mm_store_ps(f32_array, f32_toCvt);

    return _mm_set_epi64x((int64_t)f32_array[1], (int64_t)f32_array[0]);
}


// ===== Float32 to unsigned int 64 =====

NOINLINE __m128i scale_f32ToU64(__m128 f32_toCvt) {
    const __m128 OVERFLOW_THRESHOLD = _mm_set1_ps(1ull << 63);

    // If input is too large to fit in signed int, scale so it does
    __m128 willOverflowMask = _mm_cmpge_ps(f32_toCvt, OVERFLOW_THRESHOLD);
    __m128 inputNoOverflow = _mm_sub_ps(f32_toCvt, _mm_and_ps(OVERFLOW_THRESHOLD, willOverflowMask));
    __m128i scaledCvt = _convertLo_f32x4_i64x2(inputNoOverflow);

    // Scaling "removed" the MSB, so re-add it
    // "Align" lower two masks with the converted upper [mask1, mask1, mask0, mask0]
    __m128i maskAligned = _mm_shuffle_epi32( _mm_castps_si128(willOverflowMask), 0b10100000);
    return _mm_xor_si128(scaledCvt, _mm_slli_epi64( maskAligned, 63 ));
}

NOINLINE __m128i compiler_f32ToU64(__m128 f32_toCvt) {
    float f32_array[4];
    _mm_store_ps(f32_array, f32_toCvt);

    return _mm_set_epi64x((uint64_t)f32_array[1], (uint64_t)f32_array[0]);
}






// ===== Float64 to unsigned int 32 =====

NOINLINE __m128i scaleBitCombind_f64ToU32(__m128d f64_toCvt) {
    __m128i directCvt = _mm_cvttpd_epi32(f64_toCvt);
    __m128d f64_scaled = _mm_sub_pd(f64_toCvt, _mm_set1_pd(1ull << 32));
    __m128i scaledCvt = _mm_cvttpd_epi32(f64_scaled);

    __m128i isNegMask = _mm_srai_epi32(directCvt, 31);
    return _mm_or_si128(_mm_and_si128(scaledCvt, isNegMask), directCvt);
}

NOINLINE __m128i scale_f64ToU32(__m128d f64_toCvt) {
    const __m128d OVERFLOW_THRESHOLD = _mm_set1_pd(1ull << 31);

    // If input is too large to fit in signed int, scale so it does
    __m128d willOverflowMask = _mm_cmpge_pd(f64_toCvt, OVERFLOW_THRESHOLD);
    __m128d inputNoOverflow = _mm_sub_pd(f64_toCvt, _mm_and_pd(OVERFLOW_THRESHOLD, willOverflowMask));
    __m128i scaledCvt = _convert_f64x2_u32x4(inputNoOverflow);

    // Scaling "removed" the MSB, so re-add it
    // "Align" lower two masks with the converted upper [0, mask1, mask1, mask0]
    __m128i maskAligned = _mm_bsrli_si128( _mm_castpd_si128(willOverflowMask) , 4);
    return _mm_xor_si128(scaledCvt, _mm_slli_epi32( maskAligned, 31 ));
}

NOINLINE __m128i scalarInstruction_f64ToU32(__m128d f64_toCvt) {
    // Use f64 => signed 64 instruction [in range of (0, UINT32_MAX)]
    uint64_t i64_lo = _mm_cvttsd_si64(f64_toCvt);
    uint64_t i64_hi = _mm_cvttsd_si64(_mm_unpackhi_pd(f64_toCvt, f64_toCvt));
    
    __m128i lo_vec = _mm_cvtsi32_si128((uint32_t)i64_lo);
    __m128i hi_vec = _mm_cvtsi32_si128((uint32_t)i64_hi);

    return _mm_unpacklo_epi32(lo_vec, hi_vec);
}

NOINLINE __m128i compiler_f64ToU32(__m128d f64_toCvt) {
    double f64_array[2];
    _mm_store_pd(f64_array, f64_toCvt);

    return _mm_setr_epi32((uint32_t)f64_array[0], (uint32_t)f64_array[1], 0, 0);
}

// ===== Float64 to signed int 64 =====

NOINLINE __m128i scalarInstruction_f64ToI64(__m128d f64_toCvt) {
    uint64_t i64_lo = _mm_cvttsd_si64(f64_toCvt);
    uint64_t i64_hi = _mm_cvttsd_si64(_mm_unpackhi_pd(f64_toCvt, f64_toCvt));
    
    __m128i lo_vec = _mm_cvtsi64_si128(i64_lo);
    __m128i hi_vec = _mm_cvtsi64_si128(i64_hi);

    return _mm_unpacklo_epi64(lo_vec, hi_vec);
}

NOINLINE __m128i compiler_f64ToI64(__m128d f64_toCvt) {
    double f64_array[2];
    _mm_store_pd(f64_array, f64_toCvt);

    return _mm_set_epi64x((int64_t)f64_array[1], (int64_t)f64_array[0]);
}

// ===== Float64 to unsigned int 64 =====

NOINLINE __m128i scaleBitCombind_f64ToU64(__m128d f64_toCvt) {
    __m128i directCvt = _convert_f64x2_i64x2(f64_toCvt);
    __m128d f64_scaled = _mm_sub_pd(f64_toCvt, _mm_set1_pd(1ull << 31));
    __m128i scaledCvt = _convert_f64x2_i64x2(f64_scaled);

    __m128i isNegMask = _fillWithMSB_i64x2(directCvt);
    return _mm_or_si128(_mm_and_si128(scaledCvt, isNegMask), directCvt);
}

// Converts 64-bit floats into unsigned 64-bit integers
// Valid for inputs [INT64_MIN, UINT64_MAX] (negative inputs act like converting to signed, then casting to unsigned)
NOINLINE __m128i scale_f64ToU64(__m128d f64_toCvt) {
    const __m128d OVERFLOW_THRESHOLD = _mm_set1_pd(1ull << 63);

    // If input is too large to fit in signed int, scale so it does
    __m128d willOverflowMask = _mm_cmpge_pd(f64_toCvt, OVERFLOW_THRESHOLD);
    __m128d inputNoOverflow = _mm_sub_pd(f64_toCvt, _mm_and_pd(OVERFLOW_THRESHOLD, willOverflowMask));
    __m128i scaledCvt = _convert_f64x2_i64x2(inputNoOverflow);

    // Scaling "removed" the MSB, so re-add it
    return _mm_xor_si128(scaledCvt, _mm_slli_epi64( _mm_castpd_si128(willOverflowMask), 63 ));
}

NOINLINE __m128i compiler_f64ToU64(__m128d f64_toCvt) {
    double f64_array[2];
    _mm_store_pd(f64_array, f64_toCvt);

    return _mm_set_epi64x((uint64_t)f64_array[1], (uint64_t)f64_array[0]);
}
