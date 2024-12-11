#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"
#include <time.h>      // Messuring clock cycles

#include "cvt_IntToFlt_Methods.h"
#include "cvt_FltToInt_Methods.h"


#define SAMPLES (1<<13) // ~8k

NOINLINE void _perfMessure_intFlt_fun(__m128 (*funToMessure)(__m128i), size_t iterations, char* funAsStr, uint64_t* rand_ints) {
    MAYBE_VOLATILE __m128 result = _mm_setzero_ps();
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128i toCvt = _mm_loadu_si128( (__m128i*)&rand_ints[i % SAMPLES]);

        result = funToMessure(toCvt);
    }

    clock_t end_time = clock();
    printf("%.2f\t %-25s: %.2f seconds\n", _mm_cvtss_f32(result), funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}


NOINLINE void _perfMessure_intDbl_fun(__m128d (*funToMessure)(__m128i), size_t iterations, char* funAsStr, uint64_t* rand_ints) {
    MAYBE_VOLATILE __m128d result = _mm_setzero_pd();
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128i toCvt = _mm_loadu_si128( (__m128i*)&rand_ints[i % SAMPLES]);

        result = funToMessure(toCvt);
    }

    clock_t end_time = clock();
    printf("%.2lf\t %-25s: %.2f seconds\n", _mm_cvtsd_f64(result), funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}


NOINLINE void _perfMessure_fltInt_fun(__m128i (*funToMessure)(__m128), size_t iterations, char* funAsStr, float* rand_floats) {
    MAYBE_VOLATILE __m128i result = _mm_setzero_si128();
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128 toCvt = _mm_loadu_ps(&rand_floats[i % SAMPLES]);

        result = funToMessure(toCvt);
    }

    clock_t end_time = clock();
    printf("%llu\t %-25s: %.2f seconds\n", _mm_cvtsi128_si64(result), funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}


NOINLINE void _perfMessure_dblInt_fun(__m128i (*funToMessure)(__m128d), size_t iterations, char* funAsStr, double* rand_doubles) {
    MAYBE_VOLATILE __m128i result = _mm_setzero_si128();
    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128d toCvt = _mm_loadu_pd(&rand_doubles[i % SAMPLES]);

        result = funToMessure(toCvt);
    }

    clock_t end_time = clock();
    printf("%llu\t %-25s: %.2f seconds\n", _mm_cvtsi128_si64(result), funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}




// Don't need to manually enter the function's name
#define perfMessure_intFlt(funToMessure, iterations, rand_ints)     _perfMessure_intFlt_fun(funToMessure, iterations, #funToMessure, rand_ints)
#define perfMessure_intDbl(funToMessure, iterations, rand_ints)     _perfMessure_intDbl_fun(funToMessure, iterations, #funToMessure, rand_ints)
#define perfMessure_fltInt(funToMessure, iterations, rand_floats)   _perfMessure_fltInt_fun(funToMessure, iterations, #funToMessure, rand_floats)
#define perfMessure_dblInt(funToMessure, iterations, rand_doubles)  _perfMessure_dblInt_fun(funToMessure, iterations, #funToMessure, rand_doubles)


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

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}