#pragma once

#include <stdint.h>
#include <string.h> // memcpy (type generic load/copy)
#include <time.h>   // Messuring clock cycles

#define MAYBE_VOLATILE
#define _GNUC_ONLY(x)   // Nothing

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    #define GNUC_EXTENTION
    #define NOINLINE __attribute__((noinline))
    #define UNUSED __attribute__((unused))
    #undef _GNUC_ONLY
    #define _GNUC_ONLY(x) x
#elif defined(_MSC_VER)
    #define UNUSED
    #define NOINLINE __declspec(noinline)
    #define _GNUC_ONLY(x) 
#else
    #define NOINLINE
    #define UNUSED
    #define _GNUC_ONLY(x)
    // If we can't inline, prevent the result variable from being optimized away
    #undef MAYBE_VOLATILE
    #define MAYBE_VOLATILE volatile
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





// Function generator for messuring multiple single argument functions with diffrent argument/return types
#define GEN_PERF_SINGLE_ARG(returnType, funcName, inputType, scalarResType, scalarInpType, rndValSize, resFormatSpecifyStr) \
NOINLINE void _perfMessure_ ## funcName ## _func(                            \
    returnType (*funcToMessure)(inputType), size_t iterations, char* funcAsStr, scalarInpType* randVals\
){\
    const union {__m128i intDefault; returnType ret; inputType inp;} zeroVal = {_mm_setzero_si128()};\
    \
    MAYBE_VOLATILE returnType result = zeroVal.ret;                         \
    \
    clock_t start_time = clock();                                           \
    \
    for (size_t i=0; i < iterations; i++) {                                 \
        inputType toCvt = zeroVal.inp;                                      \
        memcpy(&toCvt, &randVals[i % (rndValSize)], sizeof(inputType));     \
        \
        result = funcToMessure(toCvt);                                      \
    }                                                                       \
    clock_t end_time = clock();                                             \
    \
    scalarResType scalarRes = (scalarResType)0;                             \
    memcpy(&scalarRes, &result, sizeof(scalarRes));                         \
    printf(resFormatSpecifyStr"\t %-25s: %.2f seconds\n", scalarRes, funcAsStr, (float)(end_time-start_time) / CLOCKS_PER_SEC);\
}


