#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"

#include "cvt_IntToFlt_Methods.h"
#include "cvt_FltToInt_Methods.h"


#define SAMPLES (1<<13) // ~8k


GEN_PERF_SINGLE_ARG(__m128, intFlt, __m128i, float, uint64_t, SAMPLES, "%.2f")
GEN_PERF_SINGLE_ARG(__m128d, intDbl, __m128i, double, uint64_t, SAMPLES, "%.2lf")
GEN_PERF_SINGLE_ARG(__m128i, fltInt, __m128, uint64_t, float, SAMPLES, "%llu")
GEN_PERF_SINGLE_ARG(__m128i, dblInt, __m128d, uint64_t, double, SAMPLES, "%llu")

// Don't need to manually enter the function's name
#define perfMessure_intFlt(funToMessure, iterations, rand_ints)     _perfMessure_intFlt_func(funToMessure, iterations, #funToMessure, rand_ints)
#define perfMessure_intDbl(funToMessure, iterations, rand_ints)     _perfMessure_intDbl_func(funToMessure, iterations, #funToMessure, rand_ints)
#define perfMessure_fltInt(funToMessure, iterations, rand_floats)   _perfMessure_fltInt_func(funToMessure, iterations, #funToMessure, rand_floats)
#define perfMessure_dblInt(funToMessure, iterations, rand_doubles)  _perfMessure_dblInt_func(funToMessure, iterations, #funToMessure, rand_doubles)


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
    perfMessure_intFlt(magicExpo_u32ToF32, iterations, rand_ints);
    perfMessure_intFlt(scaleBranchless_u32ToF32, iterations, rand_ints);
    perfMessure_intFlt(halfElement_u32ToF32, iterations, rand_ints);
    perfMessure_intFlt(halfElementA_u32ToF32, iterations, rand_ints);
    perfMessure_intFlt(compiler_u32ToF32, iterations, rand_ints);

    printf("\nUnsigned 32-bit to float64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_intDbl(correctAfterCvt_u32ToF64, iterations, rand_ints);
    perfMessure_intDbl(mantissaDepo_u32ToF64, iterations, rand_ints);
    perfMessure_intDbl(scalarInstruction_u32ToF64, iterations, rand_ints);
    perfMessure_intDbl(compiler_u32ToF64, iterations, rand_ints);

    #endif
    #ifdef PERF_SIGNED64_TO_FLOAT

    printf("\nSigned 64-bit to float32: Time taken to calculate %llu results...\n", iterations);
    perfMessure_intFlt(scalarInstruction_i64ToF32, iterations, rand_ints);
    perfMessure_intFlt(compiler_i64ToF32, iterations, rand_ints);

    printf("\nSigned 64-bit to float64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_intDbl(scalarInstruction_i64ToF64, iterations, rand_ints);
    perfMessure_intDbl(compiler_i64ToF64, iterations, rand_ints);

    #endif
    #ifdef PERF_UNSIGNED64_TO_FLOAT

    printf("\nUnsigned 64-bit to float32: Time taken to calculate %llu results...\n", iterations);
    perfMessure_intFlt(scaleBranchless_u64ToF32, iterations, rand_ints);
    perfMessure_intFlt(scaleLin_u64ToF32, iterations, rand_ints);
    perfMessure_intFlt(compiler_u64ToF32, iterations, rand_ints);
    
    printf("\nUnsigned 64-bit to float64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_intDbl(magicExpo_u64ToF64, iterations, rand_ints);
    perfMessure_intDbl(compiler_u64ToF64, iterations, rand_ints);

    #endif
    #ifdef PERF_FLOAT_TO_INT

    printf("\nFloat-32 to unsigned int32: Time taken to calculate %llu results...\n", iterations);
    perfMessure_fltInt(scaleBitCombind_f32ToU32, iterations, rand_flt_range32);
    perfMessure_fltInt(compiler_f32ToU32, iterations, rand_flt_range32);

    printf("\nFloat-32 to signed int64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_fltInt(scalarInstruction_f32ToI64, iterations, rand_flt_range64);
    perfMessure_fltInt(compiler_f32ToI64, iterations, rand_flt_range64);

    printf("\nFloat-32 to unsigned int64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_fltInt(scale_f32ToU64, iterations, rand_flt_range64);
    perfMessure_fltInt(scaleBranch_f32ToU64, iterations, rand_flt_range64);
    //perfMessure_fltInt(scaleBranchA_f32ToU64, iterations, rand_flt_range64);
    perfMessure_fltInt(compiler_f32ToU64, iterations, rand_flt_range64);

    #endif
    #ifdef PERF_DOUBLE_TO_INT
    
    printf("\nFloat-64 to unsigned int32: Time taken to calculate %llu results...\n", iterations);
    perfMessure_dblInt(scaleBitCombind_f64ToU32, iterations, rand_dbl_range32);
    perfMessure_dblInt(scale_f64ToU32, iterations, rand_dbl_range32);
    perfMessure_dblInt(scalarInstruction_f64ToU32, iterations, rand_dbl_range32);
    perfMessure_dblInt(compiler_f64ToU32, iterations, rand_dbl_range32);

    printf("\nFloat-64 to signed int64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_dblInt(scalarInstruction_f64ToI64, iterations, rand_dbl_range64);
    perfMessure_dblInt(compiler_f64ToI64, iterations, rand_dbl_range64);

    printf("\nFloat-64 to unsigned int64: Time taken to calculate %llu results...\n", iterations);
    perfMessure_dblInt(scaleBitCombind_f64ToU64, iterations, rand_dbl_range64);
    perfMessure_dblInt(scale_f64ToU64, iterations, rand_dbl_range64);
    perfMessure_dblInt(compiler_f64ToU64, iterations, rand_dbl_range64);
    
    #endif

    char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}