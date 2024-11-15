#pragma once

#include <emmintrin.h> // SSE2
#include <stdint.h>
#include <limits.h>
#include "_common.h"

// Create an immediate for use with the `_shuffleLoHi` function
#define _MM_SHUFHALF(index1, index0) (((index1) << 2) | (index0))


// ====== 8-bit ======

// Shift 8-bit integers left by variable while shifting in zeros
__m128i _shiftL_var_u8x16(__m128i u8ToShift, int amount) {
    __m128i shiftWCarry = _mm_sll_epi16(u8ToShift, _mm_cvtsi32_si128(amount));

    __m128i keepMask = _mm_set1_epi8((uint8_t)(UINT8_MAX << amount));
    return _mm_and_si128(shiftWCarry, keepMask);
}

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
#define _signShiftR_i8x2(toShift, amount) (\
    _mm_sub_epi8(\
        _shiftR_u8x16( _mm_add_epi8(toShift, _mm_set1_epi8(1<<7)), amount ), \
        _shiftR_u8x16( _mm_set1_epi8(1<<7), amount)\
    ) )


// ====== 64-bit ======

// ((x + (1ULL << 63)) >> amount) - ((1ULL << 63) >> amount);

// Shift 64-bit integers right by immediate while shifting in sign bits
#define _signShiftR_i64x2(toShift, amount) (\
    _mm_sub_epi64(\
        _mm_srli_epi64( _mm_add_epi64(toShift, _mm_set1_epi64x(1Ull<<63)), amount ), \
        _mm_srli_epi64( _mm_set1_epi64x(1Ull<<63), amount)\
    ) )



// ====== Shuffle ======

// Set the lower-half of the result to 2 32-bit elements from loPart, and the high to 2 elements from hiPart
#define _shuffleLoHi_i32x4(loPart, loIndexes, hiPart, hiIndexes) (_mm_castps_si128(\
    _mm_shuffle_ps(_mm_castsi128_ps(loPart), _mm_castsi128_ps(hiPart), ((hiIndexes) << 4) | (loIndexes))\
))

// Shuffle 64-bit integers using an immediate control mask
// Output: { input[imm8.bit[0]], input[imm8.bit[1]] }
#define _shuffle_i64x2(toShuffle, mask) ( _mm_shuffle_epi32(toShuffle,\
    ((mask & 1)? _MM_SHUFHALF(3,2) : _MM_SHUFHALF(1,0)) | ((mask & 2)? _MM_SHUFHALF(3,2) << 4 : _MM_SHUFHALF(1,0) << 4)\
) )
