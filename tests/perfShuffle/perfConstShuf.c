#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h> // SSE2
#include "../_perfCommon.h"
#include "../../include/sseCom_parts/shuffle.h"

#undef NOINLINE
#define NOINLINE static inline

#define SAMPLES (1<<13) // ~8k

enum {
    C0  = 0x1C, C1  = 0x74, C2  = 0x3A, C3  = 0x2C,
    C4  = 0x1D, C5  = 0xED, C6  = 0x16, C7  = 0x79,
    C8  = 0x83, C9  = 0xDE, C10 = 0xFB, C11 = 0x0C,
    C12 = 0x0B, C13 = 0x7A, C14 = 0xB1, C15 = 0x5B
};

NOINLINE __m128i xorPrecal_u8(UNUSED __m128i dummy, __m128i indexes) {
    const __m128i toSel16FF_lo = _mm_set1_epi8(C0);
    const __m128i toSel16FF_HL = _mm_set1_epi8(C0 ^ C8);

    const __m128i toSel16FT_lo = _mm_set1_epi8(C0 ^ C4);
    const __m128i toSel16FT_HL = _mm_set1_epi8(C0 ^ C4 ^ C8 ^ C12);

    const __m128i toSel16TF_lo = _mm_set1_epi8(C0 ^ C2);
    const __m128i toSel16TF_HL = _mm_set1_epi8(C0 ^ C2 ^ C8 ^ C10);

    const __m128i toSel16TT_lo = _mm_set1_epi8(C0 ^ C2 ^ C4 ^ C6);
    const __m128i toSel16TT_HL = _mm_set1_epi8(C0 ^ C2 ^ C4 ^ C6 ^ C8 ^ C10 ^ C12 ^ C14);


    const __m128i toSel32F_lo = _mm_set1_epi8(C0 ^ C1);
    const __m128i toSel32F_HL = _mm_set1_epi8(C0 ^ C1 ^ C8 ^ C9);

    const __m128i toSel32T_lo = _mm_set1_epi8(C0 ^ C1 ^ C4 ^ C5);
    const __m128i toSel32T_HL = _mm_set1_epi8(C0 ^ C1 ^ C4 ^ C5 ^ C8 ^ C9 ^ C12 ^ C13);


    const __m128i toSel64_lo = _mm_set1_epi8(C0 ^ C1 ^ C2 ^ C3);
    const __m128i toSel64_HL = _mm_set1_epi8(C0 ^ C1 ^ C2 ^ C3 ^ C8 ^ C9 ^ C10 ^ C11);

    const __m128i LOW = _mm_set1_epi8(C0 ^ C1 ^ C2 ^ C3 ^ C4 ^ C5 ^ C6 ^ C7);
    const __m128i FULL = _mm_set1_epi8(C0 ^ C1 ^ C2 ^ C3 ^ C4 ^ C5 ^ C6 ^ C7 ^ C8 ^ C9 ^ C10 ^ C11 ^ C12 ^ C13 ^ C14 ^ C15);



    
    __m128i isEven64 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-4), _mm_setzero_si128());

    __m128i cur =       _selectXorBoth_i128(LOW, FULL, isEven64);
    __m128i toSel64 =   _selectXorBoth_i128(toSel64_lo, toSel64_HL, isEven64);

    __m128i toSel32T =  _selectXorBoth_i128(toSel32T_lo, toSel32T_HL, isEven64);
    __m128i toSel32F =  _selectXorBoth_i128(toSel32F_lo, toSel32F_HL, isEven64);

    __m128i toSel16TT = _selectXorBoth_i128(toSel16TT_lo, toSel16TT_HL, isEven64);
    __m128i toSel16TF = _selectXorBoth_i128(toSel16TF_lo, toSel16TF_HL, isEven64);
    __m128i toSel16FT = _selectXorBoth_i128(toSel16FT_lo, toSel16FT_HL, isEven64);
    __m128i toSel16FF = _selectXorBoth_i128(toSel16FF_lo, toSel16FF_HL, isEven64);


    __m128i isEven32 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-3), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel64, cur, isEven32);
    __m128i toSel32 =   _selectXorBoth_i128(toSel32F, toSel32T, isEven32);
    __m128i toSel16T =  _selectXorBoth_i128(toSel16TF, toSel16TT, isEven32);
    __m128i toSel16F =  _selectXorBoth_i128(toSel16FF, toSel16FT, isEven32);


    __m128i isEven16 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-2), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel32, cur, isEven16);
    __m128i toSel16 = _selectXorBoth_i128(toSel16F, toSel16T, isEven16);


    __m128i isEven8 = _mm_cmplt_epi8(_mm_slli_epi16(indexes, 8-1), _mm_setzero_si128());

    cur = _selectXorBoth_i128(toSel16, cur, isEven8);

    return cur;
}



NOINLINE __m128i xorPrecal_u16(UNUSED __m128i dummy, __m128i indexes) {
    __m128i even64Lo = _mm_set1_epi16(C0);
    __m128i even64 = _mm_set1_epi16(C0 ^ C4);

    __m128i even32Lo = _mm_set1_epi16(C0 ^ C1);
    __m128i even32 = _mm_set1_epi16(C0 ^ C1 ^ C4 ^ C5);

    __m128i even16Lo = _mm_set1_epi16(C0 ^ C2);
    __m128i even16 = _mm_set1_epi16(C0 ^ C2 ^ C4 ^ C6);

    __m128i low = _mm_set1_epi16(C0 ^ C1 ^ C2 ^ C3);
    __m128i full = _mm_set1_epi16(C0 ^ C1 ^ C2 ^ C3 ^ C4 ^ C5 ^ C6 ^ C7);



    __m128i isOdd64 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-3), 15);
    __m128i isOdd32 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-2), 15);
    __m128i isOdd16 = _mm_srai_epi16(_mm_slli_epi16(indexes, 16-1), 15);
    
    __m128i cur =           _selectXorBoth_i128(low, full, isOdd64);
    __m128i toSelHalfLo =   _selectXorBoth_i128(even32Lo, even32, isOdd64);
    __m128i toSelHalfEven =     _selectXorBoth_i128(even16Lo, even16, isOdd64);
    __m128i toSelHalfEvenLo =   _selectXorBoth_i128(even64Lo, even64, isOdd64);

    cur =           _selectXorBoth_i128(toSelHalfLo, cur, isOdd32);
    toSelHalfLo =   _selectXorBoth_i128(toSelHalfEvenLo, toSelHalfEven, isOdd32);

    return _selectXorBoth_i128(toSelHalfLo, cur, isOdd16);
}


NOINLINE __m128i xorPrecal_u32(UNUSED __m128i dummy, __m128i indexes) {
    const __m128i FIRST = _mm_set1_epi32(C0);
    const __m128i EVEN =  _mm_set1_epi32(C0 ^ C2);

    const __m128i LO =    _mm_set1_epi32(C0 ^ C1);
    const __m128i FULL =  _mm_set1_epi32(C0 ^ C1 ^ C2 ^ C3);


    __m128i isInHi = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-2), 31);
    __m128i isInOdd = _mm_srai_epi32(_mm_slli_epi32(indexes, 32-1), 31);

    // a ^ ((a^b) & isOdd) = (isOdd)? a^a^b : a
    __m128i part_half = _mm_xor_si128(LO, _mm_and_si128(FULL, isInHi));
    __m128i selIfEven =  _mm_xor_si128(FIRST, _mm_and_si128(EVEN, isInHi));

    return _mm_xor_si128(selIfEven, _mm_and_si128(part_half, isInOdd));
}






#define perfMessure(funToMessure, iterations, rand_ints, to_shuf) do {\
    MAYBE_VOLATILE __m128i result = _mm_setzero_si128();\
    clock_t start_time = clock(); \
    \
    for (size_t i=0; i < iterations; i++) {\
        __m128i indexes = _mm_loadu_si128( (__m128i*)&rand_ints[i % SAMPLES]);\
    \
        result = funToMessure(to_shuf, indexes);\
        _GNUC_ONLY(__asm__("" : "+x" (result)));\
    }\
    \
    clock_t end_time = clock();\
    printf("%llu\t %-20s: %.2f seconds\n", _mm_cvtsi128_si64(result), #funToMessure, (float)(end_time-start_time) / CLOCKS_PER_SEC);\
} while (0)\

__attribute__((target("ssse3")))
__m128i ssse3_shuf(__m128i x, __m128i idx) {
    return _mm_shuffle_epi8(x, _mm_and_si128(idx, _mm_set1_epi8(0x0f)));
}

int main() {

    uint64_t rand_ints[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++) {
        rand_ints[i] = rrmxmx_64(i);
    }

    
    const __m128i dummy = _mm_setzero_si128();
    const __m128i LUT8  =  _mm_setr_epi8(C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15);
    const __m128i LUT16 = _mm_setr_epi16(C0,C1,C2,C3,C4,C5,C6,C7);
    const __m128i LUT32 = _mm_setr_epi32(C0,C1,C2,C3);
    const __m128i LUT64 = _mm_set_epi64x(C1,C0);

    size_t iterations = 600000000ull;



    _GNUC_ONLY(puts("Compiler can use the GNUC extention..."));

    printf("\nShuffle 8-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(ssse3_shuf, iterations, rand_ints, LUT8);
    perfMessure(xorPrecal_u8, iterations, rand_ints, dummy);


    printf("\nShuffle 16-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(_shuffleVar_i16x8, iterations, rand_ints, LUT16);
    perfMessure(xorPrecal_u16, iterations, rand_ints, dummy);


    iterations *= 4;

    printf("\nShuffle 32-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(_shuffleVar_i32x4, iterations, rand_ints, LUT32);
    perfMessure(xorPrecal_u32, iterations, rand_ints, dummy);


    printf("\nShuffle 64-bit: Time taken to calculate %zu results...\n", iterations);
    perfMessure(_shuffleVar_i64x2, iterations, rand_ints, LUT64);

    //char dummy; printf("\nEnd of test..."); scanf("\n%c", &dummy);
}
