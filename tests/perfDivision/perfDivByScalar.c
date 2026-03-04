#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include <time.h>      // Messuring clock cycles

#include "../../include/sseCom_parts/division.h"
#include "../_perfCommon.h"


#define SAMPLES (1<<13) // ~8k

#define _divPerfBase(divFunc, denomFunc, denomType, maybeAddr, denomBroad) do {\
    volatile __m128i result;\
    clock_t start_time = clock();\
    \
    for (size_t i = 0; i < iterations; i++) {\
    __m128i numerators = _setone_i128();\
    denomType denoms = denomFunc(rand_ints[i % SAMPLES]);\
    \
    result = divFunc(numerators, maybeAddr denoms denomBroad(rand_ints[i % SAMPLES]));\
    }\
    \
    clock_t end_time = clock();\
    printf("%-18llu\t %-25s: %.2f seconds\n", _mm_cvtsi128_si64(result), #denomFunc, (float)(end_time-start_time) / CLOCKS_PER_SEC);\
} while (0)

#define COMA(x) , x
#define NON(x)

#define divPerfNormal(divFunc, numeFunc) _divPerfBase(divFunc, numeFunc, __m128i, , NON)
#define divPerfMagic(divFunc, numeFunc, denomType) _divPerfBase(divFunc, numeFunc, struct denomType, &, NON)

#define modPerfNormal(divFunc, numeFunc) _divPerfBase(divFunc, numeFunc, __m128i, , NON)
#define modPerfMagic(divFunc, numeFunc, denomType, deomBroad) _divPerfBase(divFunc, numeFunc, struct denomType, &, COMA(deomBroad))




__m128i _divA_u64x2(__m128i nume, struct sseCom_divMagic_u64 *MAGIC) {
    uint64_t nume_lo = _mm_cvtsi128_si64(nume);
    uint64_t nume_hi = _mm_cvtsi128_si64(_mm_shuffle_epi32(nume, _MM_SHUFFLE(3,2,3,2)));

    uint64_t almostQuot_lo = ((__uint128_t)nume_lo * MAGIC->rcpLo) >> 64;
    uint64_t almostQuot_hi = ((__uint128_t)nume_hi * MAGIC->rcpHi) >> 64;

    uint64_t diff_lo, diff_hi;

    // Quotent is one too small when: `almostQuot * d <= n - d`
    // ...however `n - d` can underflow, but only when `almostQuot` is exact ("0")
    uint64_t quot_lo = almostQuot_lo - __builtin_sub_overflow(nume_lo, MAGIC->denomLo, &diff_lo);
    quot_lo += (almostQuot_lo * MAGIC->denomLo <= diff_lo);

    uint64_t quot_hi = almostQuot_hi - __builtin_sub_overflow(nume_hi, MAGIC->denomHi, &diff_hi);
    quot_hi += (almostQuot_hi * MAGIC->denomHi <= diff_hi);

    return _mm_unpacklo_epi64(_mm_cvtsi64_si128(quot_lo), _mm_cvtsi64_si128(quot_hi));
}


int main() {

    #define PERF_DIV
    //#define PERF_MOD

    const uint64_t PREVENT_DIV0_MASK = UINT64_C(0x0101010101010101);

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i) | PREVENT_DIV0_MASK;

    const size_t iterations = 500000000ull;


    #ifdef PERF_DIV

    printf("\nDivision unsigned 8-bit: Time taken to calculate %zu results...\n", iterations);
    divPerfNormal(_div_u8x16, _mm_set1_epi8);
    divPerfMagic(_divP_u8x16, _getDivMagic_set1_u8x16, sseCom_divMagic_u8);

    printf("\nDivision unsigned 16-bit: Time taken to calculate %zu results...\n", iterations);
    divPerfNormal(_div_u16x8, _mm_set1_epi16);
    divPerfMagic(_divP_u16x8, _getDivMagic_set1_u16x8, sseCom_divMagic_u16);

    printf("\nDivision unsigned 32-bit: Time taken to calculate %zu results...\n", iterations);
    divPerfNormal(_div_u32x4, _mm_set1_epi32);
    divPerfMagic(_divP_u32x4, _getDivMagic_set1_u32x4, sseCom_divMagic_u32);

    printf("\nDivision unsigned 64-bit: Time taken to calculate %zu results...\n", iterations);
    divPerfNormal(_div_u64x2, _mm_set1_epi64x);
    divPerfMagic(_divP_u64x2, _getDivMagic_set1_u64x2, sseCom_divMagic_u64);
    divPerfMagic(_divA_u64x2, _getDivMagic_set1_u64x2, sseCom_divMagic_u64);

    #endif

    printf("\n\n");

    #ifdef PERF_MOD

    printf("\nModulo unsigned 8-bit: Time taken to calculate %zu results...\n", iterations);
    modPerfNormal(_mod_u8x16, _mm_set1_epi8);
    modPerfMagic(_modP_u8x16, _getDivMagic_set1_u8x16, sseCom_divMagic_u8, _mm_set1_epi8);

    printf("\nModulo unsigned 16-bit: Time taken to calculate %zu results...\n", iterations);
    modPerfNormal(_mod_u16x8, _mm_set1_epi16);
    divPerfMagic(_modP_u16x8, _getDivMagic_set1_u16x8, sseCom_divMagic_u16); // Div as no denom arg

    printf("\nModulo unsigned 32-bit: Time taken to calculate %zu results...\n", iterations);
    modPerfNormal(_mod_u32x4, _mm_set1_epi32);
    modPerfMagic(_modP_u32x4, _getDivMagic_set1_u32x4, sseCom_divMagic_u32, _mm_set1_epi32);

    printf("\nModulo unsigned 64-bit: Time taken to calculate %zu results...\n", iterations);
    modPerfNormal(_mod_u64x2, _mm_set1_epi64x);
    divPerfMagic(_modP_u64x2, _getDivMagic_set1_u64x2, sseCom_divMagic_u64); // Div as no denom arg

    #endif

    char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}