#pragma once
#include "../_perfCommon.h"


NOINLINE __m128i scalar_u64(__m128i toShuff, __m128i indexes) {
    uint64_t toShuff_s[2], indexes_s[2], result[2];

    _mm_storeu_si128((__m128i*)toShuff_s, toShuff);
    _mm_storeu_si128((__m128i*)indexes_s, indexes);

    for (size_t i=0; i < 2; i++)
        result[i] = toShuff_s[indexes_s[i] & 0b1];

    return _mm_loadu_si128((__m128i*)result);
}

NOINLINE __m128i cmov_u64(__m128i toShuff, __m128i indexes) {
    
    __m128i toShuffHi = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2, 3,2));
    __m128i indexesHi = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(3,2, 3,2));

    uint64_t partLo = _mm_cvtsi128_si64(toShuff);
    uint64_t partHi = _mm_cvtsi128_si64(toShuffHi);

    uint64_t indexLo = _mm_cvtsi128_si64(indexes);
    uint64_t indexHi = _mm_cvtsi128_si64(indexesHi);

    uint64_t resLo = partLo; resLo = (indexLo & 1)? partHi : partLo;
    uint64_t resHi = partLo; resHi = (indexHi & 1)? partHi : partLo;

    return _mm_unpacklo_epi64(_mm_cvtsi64_si128(resLo), _mm_cvtsi64_si128(resHi));
}


NOINLINE __m128i selectHiLo_u64(__m128i toShuff, __m128i indexes) {

    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsHi = _mm_srai_epi32(_mm_slli_epi32(dupLo32, 31), 31);

    __m128i loDupe = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 1,0));
    __m128i hiDupe = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(3,2, 3,2));

    return _either_i128(hiDupe, loDupe, selectsHi);
}

NOINLINE __m128i selecRev_u64(__m128i toShuff, __m128i indexes) {
    
    // Shift clears all but LSB
    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsRev = _mm_cmpeq_epi32( _mm_slli_epi32(dupLo32, 31), _mm_setr_epi32(1<<31, 1<<31, 0, 0) );

    __m128i toShuffRev = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 3,2));

    return _either_i128(toShuffRev, toShuff, selectsRev);
}

NOINLINE __m128i viaXor_u64(__m128i toShuff, __m128i indexes) {
    __m128i dupLo32 = _mm_shuffle_epi32(indexes, _MM_SHUFFLE(2,2, 0,0));
    __m128i selectsHi = _mm_srai_epi32(_mm_slli_epi32(dupLo32, 31), 31);

    __m128i LO = _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 1,0));
    __m128i FULL = _mm_xor_si128(toShuff, _mm_shuffle_epi32(toShuff, _MM_SHUFFLE(1,0, 3,2)));

    return _selectXorBoth_i128(LO, FULL, selectsHi);
}