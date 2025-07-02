#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"
#include <time.h>      // Messuring clock cycles

#include "shuffle_Methods.h"

#define SAMPLES (1<<13) // ~8k

NOINLINE void _perfMessure_fun(__m128i (*funToMessure)(__m128i, __m128i), size_t iterations, char* funAsStr, uint64_t* rand_ints) {
    MAYBE_VOLATILE uint64_t result = 0;
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128i numerators = _mm_loadu_si128( (__m128i*)&rand_ints[i % SAMPLES]);
        __m128i denominators = _mm_loadu_si128( (__m128i*)&rand_ints[(i+1) % SAMPLES]);

        result = _mm_cvtsi128_si64(funToMessure(numerators, denominators));
    }

    clock_t end_time = clock();
    printf("%llu\t %-20s: %.2f seconds\n", result, funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}

// Don't need to manually enter the function's name
#define perfMessure(funToMessure, iterations, rand_ints) _perfMessure_fun(funToMessure, iterations, #funToMessure, rand_ints)


int main() {

    // Comment the ones out that you don't want to messure
    #define PERF_u8
    #define PERF_u16
    #define PERF_u32
    #define PERF_u64

    _GNUC_ONLY(puts("Compiler can use the GNUC extention..."));

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i);


    size_t iterations = 200000000ull; // 1000000000ull

    
    #ifdef PERF_u8

    printf("\nShuffle 8-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(SSSE3_u8, iterations, rand_ints);
    perfMessure(scalarForce_u8, iterations, rand_ints);
    perfMessure(shiftUnroll_u8, iterations, rand_ints);
    perfMessure(shiftUnNoMem_u8, iterations, rand_ints);
    perfMessure(indexUnroll_u8, iterations, rand_ints);
    perfMessure(loopMemIndex_u8, iterations, rand_ints);
    perfMessure(msvc_u8, iterations, rand_ints);
    #ifdef __GNUC__
    perfMessure(gccAutoVec_u8, iterations, rand_ints);
    #endif
    perfMessure(viaXor_u8, iterations, rand_ints);

    #endif
    #ifdef PERF_u16

    iterations = 400000000ull; // 400000000ull

    printf("\nShuffle 16-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(memLoop_u16, iterations, rand_ints);
    //perfMessure(switch_u16, iterations, rand_ints);
    perfMessure(insertExtractMem_u16, iterations, rand_ints);
    perfMessure(clang_u16, iterations, rand_ints);
    perfMessure(viaXor_u16, iterations, rand_ints);

    #endif
    #ifdef PERF_u32

    iterations = 800000000ull; // 800000000ull

    printf("\nShuffle 32-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_u32, iterations, rand_ints);
    perfMessure(shiftSIMD_u32, iterations, rand_ints);
    perfMessure(gcc_u32, iterations, rand_ints);
    perfMessure(shift_u32, iterations, rand_ints);

    perfMessure(clang_u32, iterations, rand_ints);

    //perfMessure(AVX_u32, iterations, rand_ints);
    perfMessure(viaXor_u32, iterations, rand_ints);
    perfMessure(viaXorA_u32, iterations, rand_ints);
    perfMessure(viaXorB_u32, iterations, rand_ints);
    _GNUC_ONLY(perfMessure(instructionJmp_u32, iterations, rand_ints));
    perfMessure(switch_u32, iterations, rand_ints);

    #endif
    #ifdef PERF_u64

    iterations = 800000000ull; // 800000000ull

    printf("\nShuffle 64-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_u64, iterations, rand_ints);
    perfMessure(cmov_u64, iterations, rand_ints);
    perfMessure(selectHiLo_u64, iterations, rand_ints);
    perfMessure(selecRev_u64, iterations, rand_ints);
    perfMessure(viaXor_u64, iterations, rand_ints);

    #endif

    char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}