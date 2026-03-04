#pragma once

#include <stdint.h>
#include <string.h> // memcpy (type generic load/copy)
#include <inttypes.h>
#include <time.h>   // Messuring clock cycles
#include <immintrin.h>
#include "../include/sseCom_parts/_common.h"

#define MAYBE_VOLATILE

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    #define GNUC_EXTENTION
    #define NOINLINE __attribute__((noinline))
    #define UNUSED __attribute__((unused))
    #define _GNUC_ONLY(x) x
#elif defined(_MSC_VER)
    #define UNUSED
    #define NOINLINE __declspec(noinline)
#else
    #define NOINLINE
    #define UNUSED
    // If we can't inline, prevent the result variable from being optimized away
    #undef MAYBE_VOLATILE
    #define MAYBE_VOLATILE volatile
#endif

#ifndef _GNUC_ONLY
    #define _GNUC_ONLY(x)   // Nothing
#endif

// More readable than a #ifdef macro around every statement
#ifdef __INTEL_LLVM_COMPILER
    #define _ICC_ONLY(x) x
#else
    #define _ICC_ONLY(x)    // Nothing
#endif


// Credit: Pelle Evensen (http://mostlymangling.blogspot.com/2018/07/on-mixing-functions-in-fast-splittable.html)
static uint64_t rrmxmx_64(uint64_t x)
{
    x ^= (x<<49 | x>>15) ^ (x<<24 | x>>40);
    x *= 0x9fb21c651e98df25; x ^= x >> 28;
    x *= 0x9fb21c651e98df25; x ^= x >> 28;
    return x;
}

#if 0
// Credit: Pelle Evensen (http://mostlymangling.blogspot.com/2019/01/better-stronger-mixer-and-test-procedure.html)
static uint64_t rrxmrrxmsx_0_64(uint64_t x) {
    x ^= (x<<25 | x>>39) ^ (x<<50 | x>>14);
    x *= 0xA24BAED4963EE407UL;
    x ^= (x<<24 | x>>40) ^ (x<<49 | x>>15);
    x *= 0x9FB21C651E98DF25UL;
    return x ^ x >> 28;
}
#endif

// Uses a condition mask to select between two values, when one of them is a precomputed xor between the two
// cond? a : b = a ^ (aXORb & cond)
static inline __m128i _selectXorBoth_i128(__m128i valIfNot, __m128i bothValXor, __m128i conditionMask) {
    return _mm_xor_si128(valIfNot, _mm_and_si128(bothValXor, conditionMask));
}




enum { SAMPLES = (1<<13) };



typedef void (*genericFunc_t)(void);

NOINLINE void perfMessure2int(genericFunc_t funcGeneric, size_t iterations, char* funAsStr, void* rand_ints) {
    MAYBE_VOLATILE uint64_t result = 0;
    __m128i (*funcToMessure)(__m128i,__m128i) = (__m128i (*)(__m128i,__m128i))funcGeneric;

    clock_t start_time = clock(); 

    for (size_t i=0; i < iterations; i++) {
        __m128i arg1 = _mm_loadu_si128( (__m128i*)((uint64_t*)rand_ints + (i % SAMPLES)) );
        __m128i arg2 = _mm_loadu_si128( (__m128i*)((uint64_t*)rand_ints + ((i+1) % SAMPLES)) );

        result = _mm_cvtsi128_si64(funcToMessure(arg1, arg2));
    }

    clock_t end_time = clock();
    printf("%"PRIu64"\t %-20s: %.2f seconds\n", result, funAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);
}



// Function generator for messuring multiple single argument functions with diffrent argument/return types
#define GEN_PERF_SINGLE_ARG(returnType, funcName, inputType, scalarResType, scalarInpType, rndValSize, resFormatSpecifyStr) \
NOINLINE void perfMessure_ ## funcName (                            \
    genericFunc_t funcToMessure, size_t iterations, char* funcAsStr, void* randVals\
){\
    const union {__m128i intDefault; returnType ret; inputType inp;} zeroVal = {_mm_setzero_si128()};\
    \
    MAYBE_VOLATILE returnType result = zeroVal.ret;                         \
    \
    clock_t start_time = clock();                                           \
    \
    for (size_t i=0; i < iterations; i++) {                                 \
        inputType toCvt = zeroVal.inp;                                      \
        memcpy(&toCvt, (scalarInpType*)randVals + (i % (rndValSize)), sizeof(inputType));     \
        \
        result = ( (returnType (*)(inputType))funcToMessure )(toCvt);                                      \
    }                                                                       \
    clock_t end_time = clock();                                             \
    \
    scalarResType scalarRes = (scalarResType)0;                             \
    memcpy(&scalarRes, &result, sizeof(scalarRes));                         \
    printf(resFormatSpecifyStr"\t %-25s: %.2f seconds\n", scalarRes, funcAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);\
}


typedef void (*_messure2Ints_t)(genericFunc_t, size_t, char*, void*);

UNUSED static void _groupPerfMessure_func(
    genericFunc_t funcList[], char *funcsAsStr, size_t funcCount, _messure2Ints_t messureFunc, size_t iterations, void* randValues
) {

    funcsAsStr = strtok(funcsAsStr, ", ");

    for (size_t i=0; i < funcCount; i++) {
        messureFunc(funcList[i], iterations, funcsAsStr, randValues);
        funcsAsStr = strtok(NULL, ", ");
    }
}

// Allow macros to expand first
#define STRVAR(...) #__VA_ARGS__

#define PERF_MESSURE_GROUP(messureFunction, iterations, randValues, func1, ...) do {\
    char funcListStr[] = STRVAR(func1, __VA_ARGS__);\
    static __typeof__(func1) *funcList[] = {func1, __VA_ARGS__};\
    \
    _groupPerfMessure_func((genericFunc_t*)funcList, funcListStr, sizeof(funcList)/sizeof(void*), messureFunction, iterations, randValues);\
} while (0)
