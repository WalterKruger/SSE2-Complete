#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"
#include <time.h>      // Messuring clock cycles

#include "sqrt64_Methods.h"

#ifdef __GNUC__
uint64_t scalarSqrt_u64(uint64_t input) {
    uint64_t result;
    __asm__(".att_syntax;"
        "fsqrt      ;"
        "fisttpll   %0"
        : "=m" (result) : "t" ((long double)input) : "st"
    );
    return result;
}
#endif

#define SAMPLES (1<<13) // ~8k

NOINLINE void _perfMessure_fun(__m128i (*funToMessure)(__m128i), size_t iterations, char* funAsStr, uint64_t* rand_ints) {
    MAYBE_VOLATILE uint64_t result = 0;
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128i radicands = _mm_loadu_si128( (__m128i*)&rand_ints[i % SAMPLES]);

        result = _mm_cvtsi128_si64(funToMessure(radicands));
    }

    clock_t end_time = clock();
    printf("%llu\t %-20s: %.2f seconds\n", result, funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}

// Don't need to manually enter the function's name
#define perfMessure(funToMessure, iterations, rand_ints) _perfMessure_fun(funToMessure, iterations, #funToMessure, rand_ints)


int main() {

    #define INCLUDE_SLOW

    #if 0
    printf("Verifying correctness...\n");
    for (size_t i=0; i <= (1ull << 34); i++) {
        uint64_t testValue = rrmxmx_64(i);

        volatile uint64_t result = _mm_cvtsi128_si64( newton_u64( _mm_cvtsi64_si128(testValue) ) );
        volatile uint64_t truth = scalarSqrt_u64(testValue);
        
        if (result != truth) {
            printf("Failed with int: %llu (expected: %llu, got: %llu) diff: %lld\n", testValue, truth, result, result-truth);
            //abort();
        }
    }
    printf("Endof verification!\n");
    #endif


    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i);


    const size_t iterations = 500000000ull;


    printf("\n64-bit Integer square root: Time taken to calculate %llu results...\n", iterations);
    perfMessure(fpu_ssse3_u64, iterations, rand_ints);
    perfMessure(fpu_rndInt_u64, iterations, rand_ints);
    perfMessure(fpu_intViaStore_u64, iterations, rand_ints);
    perfMessure(fpu_intViaStoreA_u64, iterations, rand_ints);
    perfMessure(fpu_ctrlReg_u64, iterations, rand_ints);
    perfMessure(fpu_sqCheck_u64, iterations, rand_ints);
    perfMessure(fpu_sqCheckA_u64, iterations, rand_ints);
    
    #ifdef INCLUDE_SLOW
    
    //perfMessure(fpu_minAsm_u64, iterations, rand_ints);
    perfMessure(fpu_remTrunc_u64, iterations, rand_ints);

    perfMessure(newton_u64, iterations, rand_ints);
    perfMessure(newtonA_u64, iterations, rand_ints);
    perfMessure(newtonB_u64, iterations, rand_ints);
    
    #endif


}
