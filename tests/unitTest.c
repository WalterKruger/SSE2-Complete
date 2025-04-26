#include <stdio.h>
#include <immintrin.h>  // SSE2
#include <limits.h>
#include <stdint.h>
#include <math.h>       // For scalar functions to test against
#include "_perfCommon.h"
#include <string.h>     // memcpy (copy vector into array of any type)

#include "../include/sseComplete.h"

uint32_t satAdd_u32(uint32_t x, uint32_t y) {
	uint32_t sum = x + y;
	return sum | -(sum < x);    // only need to check one
}

uint64_t satAdd_u64(uint64_t x, uint64_t y) {
	uint64_t sum = x + y;
	return sum | -(sum < x);    // only need to check one1
}

uint32_t satSub_u32(uint32_t x, uint32_t y) {
	uint32_t difference = x - y;
	return difference & -(difference <= x);
}

uint64_t satSub_u64(uint64_t x, uint64_t y) {
	uint64_t difference = x - y;
	return difference & -(difference <= x);
}


int32_t satAdd_i32(int32_t x, int32_t y) {
	int64_t sumNoOverflow = (int64_t)x + (int64_t)y;

	if (sumNoOverflow < INT32_MIN) sumNoOverflow = INT32_MIN;
	if (sumNoOverflow > INT32_MAX) sumNoOverflow = INT32_MAX;
		
	return (int32_t)sumNoOverflow;
}

int32_t satSub_i32(int32_t x, int32_t y) {
	int64_t differenceNoOverflow = (int64_t)x - (int64_t)y;

	if (differenceNoOverflow < INT32_MIN) differenceNoOverflow = INT32_MIN;
	if (differenceNoOverflow > INT32_MAX) differenceNoOverflow = INT32_MAX;
		
	return (int32_t)differenceNoOverflow;
}

#ifdef __SIZEOF_INT128__
int64_t satAdd_i64(int64_t x, int64_t y) {
	__int128_t sumNoOverflow = (__int128_t)x + (__int128_t)y;

	if (sumNoOverflow < INT64_MIN) sumNoOverflow = INT64_MIN;
	if (sumNoOverflow > INT64_MAX) sumNoOverflow = INT64_MAX;
		
	return (int64_t)sumNoOverflow;
}

int64_t satSub_i64(int64_t x, int64_t y) {
	__int128_t differenceNoOverflow = (__int128_t)x - (__int128_t)y;

	if (differenceNoOverflow < INT64_MIN) differenceNoOverflow = INT64_MIN;
	if (differenceNoOverflow > INT64_MAX) differenceNoOverflow = INT64_MAX;
		
	return (int64_t)differenceNoOverflow;
}
#endif


// Checked: div, divP, mod, modP, sqrt, cmp[all], cvtTo_f32, cvtTo_f64
//          addSat[all], subSat[all], mulHi_[all], mulLo_[all]

// FIXME None...



// ============ Start of macros you should change ============

/*
    Change the following macros based on the characteristics of your input element size
    their signess: ELEMENT_SIZE & SIGNED

    SCALAR_OP(x) is the scalar version that you want the vector version to equal to
    VECTOR_OP(x) is the vector version you want to verify
*/

#define ELEMENT_SIZE 32
#define SIGNED 0
#define CONST_ARG 1
#define SCALAR_OP(x) x / CONST_ARG
#define VECTOR_OP(x) _div_u32x4(x, _mm_set1_epi32(CONST_ARG))

//#define RES_TYPE int16_t              // Only uncomment if result type is diffrent from input
#define RES_VEC __m128i
//#define VEC_DIV_MAGIC _getDivMagic_set1_u32x4(CONST_ARG)  // Magic num input to VECTOR_OP should be "&DIV_MAGIC"



// ============ End of macros you should change ============

#define PRIMATIVE_CAT2(a,b) a ## b
#define CAT(a,b) PRIMATIVE_CAT2(a,b)
#define PRIMATIVE_CAT3(a,b,c) a ## b ## c
#define CAT3(a,b,c) PRIMATIVE_CAT3(a,b,c)

#if SIGNED == 1
    #define MACRO_TYPE  CAT3(int, ELEMENT_SIZE, _t)
    #define MAX_VALUE   CAT3(INT, ELEMENT_SIZE, _MAX)
    #define MIN_VALUE   CAT3(INT, ELEMENT_SIZE, _MIN)
#else
    #define MACRO_TYPE  CAT3(uint, ELEMENT_SIZE, _t)
    #define MAX_VALUE   CAT3(UINT, ELEMENT_SIZE, _MAX)
    #define MIN_VALUE   0
#endif

#if ELEMENT_SIZE == 64
    #define ITERATIONS UINT32_MAX
    #undef MIN_VALUE
    #define MIN_VALUE 0
    #define broadcast CAT3(_mm_set1_epi, ELEMENT_SIZE, x)
#else
    #define ITERATIONS MAX_VALUE
    #define broadcast CAT(_mm_set1_epi, ELEMENT_SIZE)
#endif

typedef MACRO_TYPE scalar_t;

#ifndef RES_TYPE
    #define RES_TYPE scalar_t
    #define SCALAR_RESULTS 128/ELEMENT_SIZE
#else
    #define SCALAR_RESULTS 128/((ELEMENT_SIZE > 8*sizeof(RES_TYPE))? ELEMENT_SIZE : 8*sizeof(RES_TYPE) )
#endif

int main() {
    #ifdef VEC_DIV_MAGIC
        CAT(struct sseCom_divMagic_u, ELEMENT_SIZE) DIV_MAGIC = VEC_DIV_MAGIC;
    #endif

    for (int64_t i = MIN_VALUE; i <= ITERATIONS; i++) {
        scalar_t testValue = (ELEMENT_SIZE == 64)? (scalar_t)rrmxmx_64(i) : (scalar_t)i;
        __m128i testVector = broadcast(testValue);

        volatile RES_TYPE resultScalar = SCALAR_OP(testValue);
        RES_VEC resultVec = VECTOR_OP(testVector);

        volatile RES_TYPE vecArray[SCALAR_RESULTS]; memcpy((RES_TYPE*)vecArray, &resultVec, sizeof(vecArray));
        
        for (size_t el=0; el < SCALAR_RESULTS; el++) {
            if (resultScalar != vecArray[el]) {
                #if SIGNED == 1
                printf("Failed with test value: %lld (%lld)\n\t", (int64_t)testValue, i);
                printf("Expected: %lld, Got[%llu]: %lld\n", (int64_t)resultScalar, el, (int64_t)vecArray[el]);
                #else
                printf("Failed with test value: %llu (%lld)\n\t", (uint64_t)testValue, i);
                printf("Expected: 0x%llx, Got[%llu]: 0x%llx\n", (uint64_t)resultScalar, el, (uint64_t)vecArray[el]);
                #endif
                abort();
            }
        }

        if (i % 100000000ull == 0) printf("Tested %lldk\n", i / 1000000);
    }

    printf("Finished test with no errors.");

}
