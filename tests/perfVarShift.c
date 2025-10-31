#include <stdio.h>
#include <stdint.h>
#include <time.h>      // Messuring clock cycles

#include "_perfCommon.h"
#include "varShift_Methods.h"


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




uint32_t scalar_bitByBit_16(uint32_t x) {
    uint32_t result = 1;
    for (size_t i=0; i < 5; i++) {
        result = (x & (1<<i))? result << (1<<i) : result;
    }
    return result;
}

int main() {

    // Comment the ones out that you don't want to messure
    #define PERF_u8
    #define PERF_u16
    #define PERF_u32
    #define PERF_u64



    #if 0
        printf("Verifying correctness...\n");
        for (size_t i=0; i <= 1000; i++) {
            const uint32_t VAL_TO_SHIFT = 0x123;

            uint32_t tesVal = VAL_TO_SHIFT * scalar_bitByBit_16(i);
            uint32_t truth = VAL_TO_SHIFT << i;
            
            if (tesVal != truth) {
                printf("Failed with: `%u << %zu` (expected: %u, got: %u)\n", VAL_TO_SHIFT, i, truth, tesVal);
            }
        }
        printf("End of verification!\n");
    #endif

    #if 0

    __m128i shiftAmount = _mm_cvtsi32_si128(1);
    __m128i toShift = _mm_cvtsi32_si128(0);

    uint32_t resTrue = _mm_cvtsi128_si32( scalarR_32(toShift, shiftAmount) );
    uint32_t resTest = _mm_cvtsi128_si32( floatScaleR_32(toShift, shiftAmount) );

    printf("True: %u, Test: %u (%u)\n", resTrue, resTest, resTest == resTrue);
        
    #endif




    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i);


    
    size_t iterations = 300000000ull; // 1000000000ull

    #ifdef PERF_u8
    
    printf("\n\nVariable 8-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_8, iterations, rand_ints);
    perfMessure(powOf2BitByBit_8, iterations, rand_ints);
    perfMessure(directBitByBit_8, iterations, rand_ints);
    perfMessure(directBitByBitA_8, iterations, rand_ints);
    perfMessure(repeatedDouble_8, iterations, rand_ints);
    perfMessure(powOf2Float_8, iterations, rand_ints);
    
    printf("\n8-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalarR_8, iterations, rand_ints);
    perfMessure(directBitByBitR_8, iterations, rand_ints);
    perfMessure(directBitByBitAR_8, iterations, rand_ints);
    perfMessure(repeatedHalfR_8, iterations, rand_ints);
    perfMessure(repeatedHalfAR_8, iterations, rand_ints);
    
    #endif

    #ifdef PERF_u16
    printf("\n\nVariable 16-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_16, iterations, rand_ints);
    perfMessure(insertExtractMem_16, iterations, rand_ints);
    perfMessure(directBitByBit_16, iterations, rand_ints);
    perfMessure(powOf2BitByBit_16, iterations, rand_ints);
    perfMessure(powOf2Float_16, iterations, rand_ints);
    perfMessure(powOf2FloatA_16, iterations, rand_ints);
    perfMessure(floatScale_16, iterations, rand_ints);

    printf("\n16-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalarR_16, iterations, rand_ints);
    perfMessure(floatScaleR_16, iterations, rand_ints);
    perfMessure(directBitByBitR_16, iterations, rand_ints);
    perfMessure(powOf2FloatR_16, iterations, rand_ints);
    #endif

    #ifdef PERF_u32
    printf("\n\nVariable 32-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_32, iterations, rand_ints);
    perfMessure(directBitByBit_32, iterations, rand_ints);
    perfMessure(powOf2BitByBit_32, iterations, rand_ints);
    perfMessure(powOf2Float_32, iterations, rand_ints);
    perfMessure(shiftSIMD_32, iterations, rand_ints);

    printf("\n32-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalarR_32, iterations, rand_ints);
    perfMessure(floatScaleR_32, iterations, rand_ints);
    perfMessure(shiftSIMDR_32, iterations, rand_ints);
    perfMessure(powOf2FloatR_32, iterations, rand_ints);
    #endif

    #ifdef PERF_u64
    printf("\n\nVariable 64-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalar_64, iterations, rand_ints);
    perfMessure(clang_64, iterations, rand_ints);
    perfMessure(directBitByBit_64, iterations, rand_ints);
    perfMessure(powOf2BitByBit_64, iterations, rand_ints);
    perfMessure(powOf2Float_64, iterations, rand_ints);

    printf("\n64-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    perfMessure(scalarR_64, iterations, rand_ints);
    perfMessure(clangR_64, iterations, rand_ints);
    #endif

    char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}