#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2

#include "../_perfCommon.h"

#include "shufMethods_8.h"
#include "shufMethods_16.h"
#include "shufMethods_32.h"
#include "shufMethods_64.h"

#define SAMPLES (1<<13) // ~8k



int main() {

    // Comment the ones out that you don't want to messure
    #define PERF_u8
    #define PERF_u16
    #define PERF_u32
    #define PERF_u64

    _GNUC_ONLY(puts("Compiler can use the GNUC extention..."));

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i); // & 0x0707070707070707ull;


    size_t iterations = 200000000ull; // 1000000000ull

    


    #ifdef PERF_u8

    printf("\nShuffle 8-bit: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints, 
        SSSE3_u8, 
        scalarForce_u8,
        shiftUnroll_u8,
        shiftUnNoMem_u8,
        indexUnroll_u8, 
        loopMemIndex_u8, 
        msvc_u8,
        _GNUC_ONLY(gccAutoVec_u8),
        viaXor_u8,
        viaXorA_u8,
        viaXorB_u8,

        bitByBit_u8,
        bitByBitA_u8
    );

    #endif
    #ifdef PERF_u16

    iterations = 400000000ull; // 400000000ull

    printf("\nShuffle 16-bit: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        memLoop_u16,
        //switch_u16,
        insertExtractMem_u16,
        clang_u16,
        viaXor_u16,
        viaXorA_u16,
        viaXorB_u16,
        viaXorC_u16
    );

    #endif
    #ifdef PERF_u32

    iterations = 800000000ull; // 800000000ull

    printf("\nShuffle 32-bit: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_u32,
        shiftSIMD_u32,
        gcc_u32,
        shift_u32,

        clang_u32,

        AVX_u32,
        viaXor_u32,
        viaXorA_u32,
        viaXorB_u32,
        viaXorC_u32,

        //_GNUC_ONLY(instructionJmp_u32),
        //switch_u32
    );

    #endif
    #ifdef PERF_u64

    iterations = 800000000ull; // 800000000ull

    printf("\nShuffle 64-bit: Time taken to calculate %zu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        scalar_u64,
        cmov_u64,
        selectHiLo_u64,
        selecRev_u64,
        viaXor_u64
    );

    #endif

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}