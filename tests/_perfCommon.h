#pragma once

#include <stdint.h>


#define MAYBE_VOLATILE

#ifdef __GNUC__
    #define NOINLINE __attribute__((noinline))
    #define UNUSED __attribute__((unused))
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
