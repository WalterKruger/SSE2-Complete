#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"
#include <time.h>      // Messuring clock cycles

#include "divisionMethods.h"
#include "moduloMethods.h"


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
    #define PERF_mod_u8
    #define PERF_u16
    #define PERF_u32
    #define PERF_mod_u32
    #define PERF_u64



    #if 0
        printf("Verifying correctness...\n");
        for (size_t i=1; i <= UINT32_MAX; i++) {
            uint32_t testDenom = i;
            uint32_t test_nume = UINT32_MAX;

            uint32_t result = _mm_cvtsi128_si32( vecDiv_u32( _mm_cvtsi32_si128(test_nume), _mm_cvtsi32_si128(testDenom) ) );
            uint32_t truth = test_nume / testDenom;
            
            if (result != truth) {
                printf("Failed with: `%u / %u` (expected: %u, got: %u)\n", test_nume, testDenom, truth, result);
            }
        }
        printf("Endof verification!\n");
    #endif


    const uint64_t PREVENT_DIV0_MASK = UINT64_C(0x0101010101010101);

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i) | PREVENT_DIV0_MASK;


    const size_t iterations = 500000000ull;

    #ifdef PERF_u8

    printf("\nDivision unsigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(longDiv_u8, iterations, rand_ints);
    perfMessure(linDiv_u8, iterations, rand_ints);
    perfMessure(linfDiv_u8, iterations, rand_ints);
    perfMessure(vecDiv_u8, iterations, rand_ints);
    perfMessure(magicDiv_u8, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_u8, iterations, rand_ints));

    printf("\nSigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(vecDiv_i8, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_i8, iterations, rand_ints));

    #endif
    #ifdef PERF_mod_u8

    printf("\nModulo unsigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(vecMod_u8, iterations, rand_ints);
    perfMessure(longMod_u8, iterations, rand_ints);
    perfMessure(linMod_u8, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlMod_u8, iterations, rand_ints));

    #endif
    #ifdef PERF_u16

    printf("\nUnsigned 16-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(longDiv_u16, iterations, rand_ints);
    perfMessure(linDiv_u16, iterations, rand_ints);
    perfMessure(linUnrolledDiv_u16, iterations, rand_ints);
    perfMessure(linDivf_u16, iterations, rand_ints);
    perfMessure(vecDiv_u16, iterations, rand_ints);
    perfMessure(vecRCPDiv_u16, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_u16, iterations, rand_ints));
    
    printf("\nSigned 16-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(vecDiv_i16, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_i16, iterations, rand_ints));



    #endif
    #ifdef PERF_u32

    printf("\nDivision unsigned 32-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(linDiv_u32, iterations, rand_ints);
    perfMessure(linUnrolledDiv_u32, iterations, rand_ints);
    perfMessure(linDivf_u32, iterations, rand_ints);
    //perfMessure(longDiv_u32, iterations, rand_ints);
    perfMessure(vecDiv_u32, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_u32, iterations, rand_ints));

    printf("\nDivision signed 32-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(vecDiv_i32, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_i32, iterations, rand_ints));

    #endif
    #ifdef PERF_mod_u32

    printf("\nModulo unsigned 32-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(linMod_u32, iterations, rand_ints);
    perfMessure(linUnrollMod_u32, iterations, rand_ints);
    perfMessure(vecMod_u32, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlMod_u32, iterations, rand_ints));

    #endif

    #ifdef PERF_u64

    printf("\nDivision unsigned 64-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(linDiv_u64, iterations, rand_ints);
    //perfMessure(longDiv_u64, iterations, rand_ints);
    perfMessure(linDivf_u64, iterations, rand_ints);
    perfMessure(vecLin_u64, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_u64, iterations, rand_ints));

    printf("\nDivision signed 64-bit: Time taken to calculate %llu results...\n", iterations);
    perfMessure(linDiv_i64, iterations, rand_ints);
    _ICC_ONLY(perfMessure(_svmlDiv_i64, iterations, rand_ints));

    #endif
}
