#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "../_perfCommon.h"

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



GEN_PERF_SINGLE_ARG(__m128i, 1int, __m128i, uint64_t, uint64_t, SAMPLES, "%llu")


// Don't need to manually enter the function's name
#define perfMessure(funToMessure, iterations, rand_ints) perfMessure_1int((genericFunc_t)funToMessure, iterations, #funToMessure, rand_ints)


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
    PERF_MESSURE_GROUP(perfMessure_1int, iterations, rand_ints,
        fpu_ssse3_u64,
        fpu_rndInt_u64,
        fpu_intViaStore_u64,
        fpu_intViaStoreA_u64,
        fpu_ctrlReg_u64,
        fpu_sqCheck_u64,
        fpu_sqCheckA_u64
    );
    
    #ifdef INCLUDE_SLOW
    
    PERF_MESSURE_GROUP(perfMessure_1int, iterations, rand_ints,
        //fpu_minAsm_u64,
        fpu_remTrunc_u64,

        newton_u64,
        newtonA_u64,
        newtonB_u64
    );

    #endif


}
