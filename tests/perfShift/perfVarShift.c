#include <stdio.h>
#include <stdint.h>

#include "shiftMethods_8.h"
#include "shiftMethods_16.h"
#include "shiftMethods_32.h"
#include "shiftMethods_64.h"



// Don't need to manually enter the function's name
#define perfMessure(funToMessure, iterations, rand_ints) perfMessure2int((genericFunc_t)funToMessure, iterations, #funToMessure, rand_ints)




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


    
    size_t iterations = 600000000ull; // 600000000ull

    #ifdef PERF_u8
    
    printf("\n\nVariable 8-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_8, powOf2BitByBit_8, directBitByBit_8, directBitByBitA_8, 
        repeatedDouble_8, powOf2Float_8
    );
    
    printf("\n8-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalarR_8, directBitByBitR_8, directBitByBitAR_8,
        repeatedHalfR_8, repeatedHalfAR_8
    );

    #endif
    #ifdef PERF_u16

    printf("\n\nVariable 16-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_16, insertExtractMem_16, directBitByBit_16,
        powOf2BitByBit_16, powOf2Float_16, powOf2FloatA_16,
        floatScale_16, ssse3LUT_16
    );

    printf("\n16-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalarR_16, floatScaleR_16, directBitByBitR_16, powOf2FloatR_16
    );

    #endif
    #ifdef PERF_u32

    printf("\n\nVariable 32-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_32, directBitByBit_32, powOf2BitByBit_32, 
        powOf2Float_32, shiftSIMD_32
    );

    printf("\n32-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalarR_32, floatScaleR_32, shiftSIMDR_32, powOf2FloatR_32
    );

    #endif
    #ifdef PERF_u64

    printf("\n\nVariable 64-bit LEFT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_64, clang_64, directBitByBit_64, 
        powOf2BitByBit_64, powOf2Float_64
    );

    printf("\n64-bit RIGHT shifts: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalarR_64, clangR_64
    );
    
    #endif

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}
