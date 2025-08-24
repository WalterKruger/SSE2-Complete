#pragma once

#include <stdint.h>
#include <math.h> // rintl
#include <emmintrin.h> // SSE2
#include "_perfCommon.h"
#include "../include/sseComplete.h"

#ifdef __GNUC__
// Use the FPU as it uses a 80-bit float capable of representing all 
// 64-bit integers exactly and has hardware support for square root

// The hard part is doing the truncation convertion to integer
// which doesn't have direct hardware support until SSSE3

NOINLINE __m128i fpu_compiler_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; ++i) {
        long double valAsFlt = (long double)u64_array[i];
        
        __asm__("fsqrt" : "+t" (valAsFlt));

        sqrts[i] = (uint32_t)valAsFlt;
    }
    
    return _mm_loadu_si128((__m128i*)sqrts);
}

// Use SSSE3 hardware instruction
NOINLINE __m128i fpu_ssse3_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; ++i) {
        __asm__(".att_syntax;"
            "fsqrt      ;"
            "fisttpll   %0"
            : "=m" (sqrts[i]) : "t" ((long double)u64_array[i]) : "st"
        );
    }
    
    return _mm_loadu_si128((__m128i*)sqrts);
}

// Use the remainder instruction as it truncates
NOINLINE __m128i fpu_remTrunc_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        __asm__(".att_syntax\n\t"
        // st(0) = input     st(1) = 1.0L
            "fsqrt              ;" // st(0) = sqrt     st(1) = 1.0L
            "fst     %%st(2)    ;" // st(0) = sqrt     st(1) = 1.0L    st(2) = sqrt
            "fprem              ;" // st(0) = REM      st(1) = 1.0L    st(2) = sqrt
            "fsubrp  %%st(0), %%st(2);" // st(1-1) = 1.0L    st(2-1) = floor(sqrt)
            "fstp   %%st        ;" // st(1-1) = floor(sqrt)
            "fistpll  %0"
            : "=m" (sqrts[i]) 
            : "t" ((long double)u64_array[i]), "u" (1.0L)
            : "st", "st(1)", "st(2)"
        );
    }
    
    return _mm_loadu_si128((__m128i*)sqrts);
}

// Use the frndint instruction as it is based on the current rounding mode.
// If that value is greater than the non-int, subtract one to round down instead
NOINLINE __m128i fpu_rndInt_u64(__m128i input) {
    static const double NEG_ONE = -1.0;

    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        __asm__(".att_syntax\n\t"
        // Calculate double extended-precision square root
            "fsqrt                  ;"
        // Round to int to check rounding mode
            "fld     %%st           ;"
            "frndint                ;"
        // if (rounded up) {  result -= 1.0  }  [Round down instead]
            "fcomi   %%st(1), %%st  ;"
            "jbe     .keepRounding_sqrtu64_%=;"
            "faddl   %[NEG_ONE]     ;"
        ".keepRounding_sqrtu64_%=:  ;"
        // Store result and clear stack
            "fistpll     %[result]  ;"
            "fstp        %%st       ;"
        : [result] "=m" (sqrts[i]) 
        : "t" ((long double)u64_array[i]), [NEG_ONE] "m" (NEG_ONE)
        : "st", "st(1)"
        );
    }
    
    return _mm_loadu_si128((__m128i*)sqrts);
}

// Use the store-as-int then load-int-to-float pattern as `frndint` is slow?
NOINLINE __m128i fpu_intViaStore_u64(__m128i input) {
    static const double NEG_ONE = -1.0;

    uint64_t u64_array[2], sqrts[2], memTmp;
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        __asm__(".att_syntax\n\t"
        // Calculate double extended-precision square root
            "fsqrt                  ;"
        // Round to int to check rounding mode
            "fld     %%st           ;"
            "fistpll %[MEM_TMP]     ;"
            "fildll  %[MEM_TMP]     ;"
        // if (rounded up) {  result -= 1.0  }  [Round down instead]
            "fcomi   %%st(1), %%st  ;"
            "jbe     .keepRounding_sqrtu64_%=;"
            "faddl   %[NEG_ONE]     ;"
        ".keepRounding_sqrtu64_%=:  ;"
        // Store result and clear stack
            "fistpll     %[result]  ;"
            "fstp        %%st       ;"
        : [result] "=m" (sqrts[i]), [MEM_TMP] "+m" (memTmp)
        : "t" ((long double)u64_array[i]), [NEG_ONE] "m" (NEG_ONE)
        : "st", "st(1)"
        );
    }

    return _mm_loadu_si128((__m128i*)sqrts);
}

// Alternative method where we subtract the memory address, not the float
NOINLINE __m128i fpu_intViaStoreA_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        __asm__(".att_syntax\n\t"
        // Calculate double extended-precision square root
            "fsqrt                  ;"
        // Round to int to check rounding mode
            "fld     %%st           ;"
            "fistpll %[result]      ;"
            "fildll  %[result]      ;"
        // if (rounded up) {  result -= 1.0  }  [Round down instead]
            "fcomip  %%st(1), %%st  ;"
            "jbe     .keepRounding_sqrtu64_%=;"
            "subq   $1, %[result]   ;"
        ".keepRounding_sqrtu64_%=:  ;"
        // Clear stack
            "fstp        %%st       ;"
        : [result] "+m" (sqrts[i])
        : "t" ((long double)u64_array[i])
        : "st", "st(1)"
        );
    }

    return _mm_loadu_si128((__m128i*)sqrts);
}

// Storing as an int sets a FPU status flag indicating rounding direction
// However, reading from this register is expensive!
NOINLINE __m128i fpu_ctrlReg_u64(__m128i input) {
    const int C1_POS = 9;

    uint64_t u64_array[2], sqrts[2], ctrlWords[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        __asm__(".att_syntax\n\t"
        // Calculate double extended-precision square root
            "fsqrt                  ;"
        // An inexact store sets C1 (9th bit) flag depending on rounding direction
            "fistpll %[sqrt]      ;"
            "fnstsw   %%ax;" // "fstsw   %[controlWord] ;"
        : [sqrt] "=m" (sqrts[i]), [controlWord] "=a" (ctrlWords[i])
        : "t" ((long double)u64_array[i])
        : "st"
        );
    }

    #if 1
    return _mm_set_epi64x(
        sqrts[1] - ((ctrlWords[1] >> C1_POS) & 1),
        sqrts[0] - ((ctrlWords[0] >> C1_POS) & 1)
    );
    #else
    return _mm_set_epi64x(
        ( (int64_t)(ctrlWords[1] << (63 - C1_POS)) >> 63 ) + sqrts[1],
        ( (int64_t)(ctrlWords[0] << (63 - C1_POS)) >> 63 ) + sqrts[0]
    );
    #endif
}

NOINLINE __m128i fpu_sqCheck_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    // Break into two loops to allow for better instruction scheduling
    double sqrt_dbl[2];
    for (size_t i=0; i<2; i++) {
        // GCC checks for negative inputs with `sqrtl`???
        #ifdef __GNUC__
        __asm__(".att_syntax\n\t"
            "fsqrt          ;"
            "fstpl   %0     ;"
        : "=m" (sqrt_dbl[i])  : "t" ((long double)u64_array[i]) : "st"
        );
        #else
        sqrt_dbl[i] = (double)sqrtl( (long double)u64_array[i] );
        #endif
    }

    for (size_t i=0; i<2; i++) {
        // Convert to double first as it has truncation convertion to integer
        // (Direct conversion needs to account for case where it rounds up above UINT32_MAX)
        uint64_t sqrt_u64 = (int64_t)sqrt_dbl[i]; // Sign doesn't matter (sqrt(u64) => u32)

        // However, if f80 => f64 conversion rounds up to the next integer,
        // the truncation won't help, so check for that case
        sqrts[i] = sqrt_u64 - (sqrt_u64 * sqrt_u64 > u64_array[i]);
    } 
    
    return _mm_loadu_si128((__m128i*)sqrts);
}

NOINLINE __m128i fpu_sqCheckA_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    // Break into two loops to allow for better instruction scheduling
    uint64_t sqrt_int[2];
    for (size_t i=0; i<2; i++) {
        #if 1
        __asm__(".att_syntax\n\t"
            "fsqrt          ;"
            "fistpll   %0   ;"
        : "=m" (sqrt_int[i])  : "t" ((long double)u64_array[i]) : "st"
        );
        #else
        // Signed doesn't matter since unsigned-32 in range of signed-64
        sqrt_int[i] = llrintl(sqrtl(u64_array[i]));
        #endif
    }

    for (size_t i=0; i<2; i++) {
        uint32_t sqrt_u32 = (uint32_t)sqrt_int[i];

        sqrts[i] = sqrt_int[i] - ((uint64_t)sqrt_u32 * sqrt_u32 > u64_array[i]) - (sqrt_int[i] != sqrt_u32);
    } 
    
    return _mm_loadu_si128((__m128i*)sqrts);
}


__m128i fpu_minAsm_u64(__m128i input) {
    uint64_t u64_array[2], sqrts[2];
    _mm_store_si128((__m128i*)u64_array, input);

    for (size_t i=0; i<2; i++) {
        long double fpu_sqrt;// = sqrtl((long double)u64_array[i]);
        __asm__("fsqrt" : "=t" (fpu_sqrt) : "0" ((long double)u64_array[i]));

        long double fpu_isqrt = rintl(fpu_sqrt);

        if (fpu_sqrt < fpu_isqrt) fpu_isqrt -= 1;

        __asm__(".att_syntax\n\t"
            "fistpll %0" : "=m" (sqrts[i]) : "t" (fpu_isqrt)
        );
    }

    return _mm_loadu_si128((__m128i*)sqrts);
}
#endif


// FIXME: Using double sqrt as inital guess fails for some reason
// Newton's method is by far the fastest non-FPU methods
NOINLINE __m128i newton_u64(__m128i input) {
    uint64_t n[2], sqrts[2];
    _mm_store_si128((__m128i*)n, input);

    for (size_t i=0; i<2; i++) {
        if (n[i] == 0) {  sqrts[i] = 0; continue;  };

        #if 0
        uint64_t guess = (uint64_t)sqrt(n[i]);
        guess = (guess + n[i] / guess) / 2;
        
        
        #else

        uint64_t guess = 1ull << (33 - __builtin_clzll(n[i]) / 2);
        //uint64_t guess = n[i] / 2;
        uint64_t nextGuess = (guess + 1) / 2;

        while (nextGuess < guess) {
            guess = nextGuess;
            nextGuess = (guess + n[i] / guess) / 2;
        } while(0);
        #endif
        sqrts[i] = guess;
    }
    
    return _mm_loadu_si128((__m128i*)sqrts);
}


NOINLINE __m128i newtonA_u64(__m128i input) {
    //input = _mm_sub_epi64(input, _cmpEq_i64x2(input, _mm_setzero_si128()));

    __m128d guess_dbl = _mm_sqrt_pd( _convert_u64x2_f64x2(input) );
    __m128i guess = _convert_f64x2_i64x2(guess_dbl); // Sign doesn't matter

    // Prevents div by zero
    __m128i inputIsZero = _mm_castpd_si128(_mm_cmpeq_pd(guess_dbl, _mm_set1_pd(0.0)));
    guess = _mm_sub_epi64(guess, inputIsZero);  // 0 - 0xff.. = 1 

    // x{n+1} = 0.5 * (x{n} + S / x{n})
    return _mm_srli_epi64( _mm_add_epi64(guess, _div_u64x2(input, guess)), 1 );
}

NOINLINE __m128i newtonB_u64(__m128i input) {
    // A double can represent most 64-bit ints and has hardware support for SIMD sqrt
    // Then use a single iteration of newton's method to get exact result
    // Newton's method: x{n+1} = 0.5 * (x{n} + S / x{n})

    // Let approximation x{n} = sqrt((double)S)
    __m128d guess_dbl = _mm_sqrt_pd( _convert_u64x2_f64x2(input) );

    // Both `div_u64` & `convert_f64x2_u64x2` extract from vec and act one-by-one
    // so doing their operations manually avoids redundent extract+repack

    uint64_t input_lo = _mm_cvtsi128_si64(input);
    uint64_t input_hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(input, input));

    // Convert from double => "uint64_t" (The instruction being signed doesn't)
    uint64_t guess_int_lo = _mm_cvtsd_si64(guess_dbl);
    uint64_t guess_int_hi = _mm_cvtsd_si64(_mm_unpackhi_pd(guess_dbl, guess_dbl));

    // S / x{n}
    uint64_t quot_lo = input_lo / (guess_int_lo + (guess_int_lo == 0)); // Prevents div by zero
    uint64_t quot_hi = input_hi / (guess_int_hi + (guess_int_hi == 0));

    __m128i guess_vec = _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(guess_int_lo), _mm_cvtsi64_si128(guess_int_hi)
    );

    __m128i quot_vec = _mm_unpacklo_epi64(_mm_cvtsi64_si128(quot_lo), _mm_cvtsi64_si128(quot_hi));

    // x{n+1} = (x{n} + QUOTENT) / 2
    return _mm_srli_epi64( _mm_add_epi64(guess_vec, quot_vec), 1 );
}