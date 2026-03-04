#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <emmintrin.h> // SSE2
#include "../_perfCommon.h"

#include "cvt_IntToFlt_Methods.h"
#include "cvt_FltToInt_Methods.h"


#define SAMPLES (1<<13) // ~8k


GEN_PERF_SINGLE_ARG(__m128, intFlt, __m128i, float, uint64_t, SAMPLES, "%.2f")
GEN_PERF_SINGLE_ARG(__m128d, intDbl, __m128i, double, uint64_t, SAMPLES, "%.2lf")
GEN_PERF_SINGLE_ARG(__m128i, fltInt, __m128, uint64_t, float, SAMPLES, "%"PRIu64)
GEN_PERF_SINGLE_ARG(__m128i, dblInt, __m128d, uint64_t, double, SAMPLES, "%"PRIu64)


int main() {

    // Comment the ones out that you don't want to messure
    #define PERF_INT32_TO_FLOAT
    #define PERF_SIGNED64_TO_FLOAT
    #define PERF_UNSIGNED64_TO_FLOAT
    #define PERF_FLOAT_TO_INT
    #define PERF_DOUBLE_TO_INT

    #if 0
    // NOTE: When converting INT_MAX to a float, it may cause it to be slightly larger (and thus out of range)...
    printf("Verifying correctness...\n");
    for (size_t i=0; i <= UINT32_MAX; i++) {
        uint32_t testValue = i;
        //double testValue = (uint32_t)rrmxmx_64(i);//i;

        volatile float result = _mm_cvtss_f32( gccCvtA_u32ToF32( _mm_cvtsi32_si128(testValue) ) );
        volatile float truth = (float)testValue;
        
        if (result != truth) {
            printf("Failed with int: %llu (expected: %+.2f, got: %+.2f)\n", (uint64_t)testValue, truth, result);
            abort();
        }
    }
    printf("Endof verification!\n");
    #endif


    UNUSED uint64_t rand_ints[SAMPLES];
    UNUSED float rand_flt_range32[SAMPLES], UNUSED rand_flt_range64[SAMPLES];;
    UNUSED double rand_dbl_range32[SAMPLES], UNUSED rand_dbl_range64[SAMPLES];

    // Assume floats are in range since converting values too large for a type
    // is undefined behavior
    for (size_t i=0; i < SAMPLES; i++) rand_ints[i] =    rrmxmx_64(i);
    for (size_t i=0; i < SAMPLES; i++) rand_flt_range64[i] = (float)(int64_t)rrmxmx_64(i);
    for (size_t i=0; i < SAMPLES; i++) rand_flt_range32[i] = (float)(int32_t)rrmxmx_64(i);
    for (size_t i=0; i < SAMPLES; i++) rand_dbl_range64[i] = (double)(int64_t)rrmxmx_64(i);
    for (size_t i=0; i < SAMPLES; i++) rand_dbl_range32[i] = (double)(int32_t)rrmxmx_64(i);


    const size_t iterations = 1000000000ull;

    #ifdef PERF_INT32_TO_FLOAT

    printf("\nUnsigned 32-bit to float32: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intFlt, iterations, rand_ints,
        magicExpo_u32ToF32, scaleBranchless_u32ToF32, halfElement_u32ToF32,
        halfElementA_u32ToF32, compiler_u32ToF32
    );

    printf("\nUnsigned 32-bit to float64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intDbl, iterations, rand_ints,
        correctAfterCvt_u32ToF64, mantissaDepo_u32ToF64, scalarInstruction_u32ToF64, compiler_u32ToF64
    );

    #endif
    #ifdef PERF_SIGNED64_TO_FLOAT

    printf("\nSigned 64-bit to float32: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intFlt, iterations, rand_ints,
        scalarInstruction_i64ToF32, compiler_i64ToF32
    );

    printf("\nSigned 64-bit to float64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intDbl, iterations, rand_ints,
        scalarInstruction_i64ToF64, compiler_i64ToF64
    );

    #endif
    #ifdef PERF_UNSIGNED64_TO_FLOAT

    printf("\nUnsigned 64-bit to float32: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intFlt, iterations, rand_ints,
        scaleBranchless_u64ToF32, scaleLin_u64ToF32, doubleRound_u64ToF32,
        thirdFusedSum_u64ToF32, thirdFusedSumA_u64ToF32, dblHalf2Sum_u64ToF32,
        dblHalf2SumA_u64ToF32, viaF80_u64ToF32, compiler_u64ToF32
    );
    
    printf("\nUnsigned 64-bit to float64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_intDbl, iterations, rand_ints,
        magicExpo_u64ToF64, compiler_u64ToF64
    );

    #endif
    #ifdef PERF_FLOAT_TO_INT

    printf("\nFloat-32 to unsigned int32: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_fltInt, iterations, rand_flt_range32,
        scaleBitCombind_f32ToU32, scale_f32ToU32, compiler_f32ToU32
    );

    printf("\nFloat-32 to signed int64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_fltInt, iterations, rand_flt_range64,
        scalarInstruction_f32ToI64, compiler_f32ToI64
    );

    printf("\nFloat-32 to unsigned int64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_fltInt, iterations, rand_flt_range64,
        scale_f32ToU64, scaleBranch_f32ToU64, /*scaleBranchA_f32ToU64,*/ compiler_f32ToU64
    );

    #endif
    #ifdef PERF_DOUBLE_TO_INT
    
    printf("\nFloat-64 to unsigned int32: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_dblInt, iterations, rand_dbl_range32,
        scaleBitCombind_f64ToU32, scale_f64ToU32, scalarInstruction_f64ToU32,
        scaleIntoMant_f64ToU32, compiler_f64ToU32
    );

    printf("\nFloat-64 to signed int64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_dblInt, iterations, rand_dbl_range64,
        scalarInstruction_f64ToI64, compiler_f64ToI64
    );

    printf("\nFloat-64 to unsigned int64: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure_dblInt, iterations, rand_dbl_range64,
        scaleBitCombind_f64ToU64, scale_f64ToU64, compiler_f64ToU64
    );
    
    #endif

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}