#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "../_perfCommon.h"
#include <time.h>      // Messuring clock cycles

#include "divisionMethods.h"
#include "moduloMethods.h"


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



__m128i divPrecomp7_u64(__m128i nume, UNUSED __m128i sink) {
    struct sseCom_divMagic_u64 MAGIC = _getDivMagic_set1_u64x2(7);

    return _divP_u64x2(nume, &MAGIC);
}


int main() {

    // Comment the ones out that you don't want to messure
    #define PERF_u8
    #define PERF_mod_u8
    #define PERF_u16
    #define PERF_u32
    #define PERF_mod_u32
    #define PERF_u64



    #if 0
        printf("Verifying correctness...\n");
        for (size_t i=1; i <= UINT32_MAX; i++) {
            uint32_t testDenom = i;
            uint32_t test_nume = UINT32_MAX;

            uint32_t result = _mm_cvtsi128_si32( vecDiv_u32( _mm_cvtsi32_si128(test_nume), _mm_cvtsi32_si128(testDenom) ) );
            uint32_t truth = test_nume / testDenom;
            
            if (result != truth) {
                printf("Failed with: `%u / %u` (expected: %u, got: %u)\n", test_nume, testDenom, truth, result);
            }
        }
        printf("Endof verification!\n");
    #endif


    const uint64_t PREVENT_DIV0_MASK = UINT64_C(0x0101010101010101);

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rand_ints[i] = rrmxmx_64(i) | PREVENT_DIV0_MASK;


    const size_t iterations = 1000000000ull; // 1000000000ull

    #ifdef PERF_u8

    printf("\nDivision unsigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        //longDiv_u8,
        //linDiv_u8,
        linfDiv_u8,
        vecDiv_u8,
        //magicDiv_u8,
        lut64_sse41_u8,
        _ICC_ONLY(_svmlDiv_u8)
    );

    printf("\nSigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        vecDiv_i8, vecDivA_i8, _ICC_ONLY(_svmlDiv_i8)
    );

    #endif
    #ifdef PERF_mod_u8

    printf("\nModulo unsigned 8-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        vecMod_u8, longMod_u8, linMod_u8, _ICC_ONLY(_svmlMod_u8)
    );

    #endif
    #ifdef PERF_u16

    printf("\nUnsigned 16-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        longDiv_u16, linDiv_u16, linUnrolledDiv_u16,
        linDivf_u16, vecDiv_u16, vecRCPDiv_u16,
        _ICC_ONLY(_svmlDiv_u16)
    );
    
    printf("\nSigned 16-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        vecDiv_i16, _ICC_ONLY(_svmlDiv_i16)
    );



    #endif
    #ifdef PERF_u32

    printf("\nDivision unsigned 32-bit: Time taken to calculate %llu results...\n", iterations);
        PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        linDiv_u32, linUnrolledDiv_u32, linDivf_u32,
        //longDiv_u32,
        vecDiv_u32, _ICC_ONLY(_svmlDiv_u32)
    );

    printf("\nDivision signed 32-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        vecDiv_i32, _ICC_ONLY(_svmlDiv_i32)
    );

    #endif
    #ifdef PERF_mod_u32

    printf("\nModulo unsigned 32-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        linMod_u32, linUnrollMod_u32, vecMod_u32, _ICC_ONLY(_svmlMod_u32)
    );

    #endif

    #ifdef PERF_u64

    printf("\nDivision unsigned 64-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        linDiv_u64, //longDiv_u64,
        linDivf_u64, vecLin_u64, divPrecomp7_u64,
        _ICC_ONLY(_svmlDiv_u64)
    );

    printf("\nDivision signed 64-bit: Time taken to calculate %llu results...\n", iterations);
    PERF_MESSURE_GROUP(perfMessure2int, iterations, rand_ints,
        linDiv_i64, _ICC_ONLY(_svmlDiv_i64)
    );

    #endif

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}
