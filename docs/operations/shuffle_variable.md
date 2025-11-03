# Variable shuffle

Shuffle elements by using the corresponding variable elements as indexes. The indexes are the same width as the elements being shifted, with only the bits that are in range being considered.

There is no special behavior for the most significant bit (unlike `_mm_shuffle_epi8`).

## Signature

```C
__m128i _shuffleVar_i8x16(__m128i toShuffle, __m128i indexes)
__m128i _shuffleVar_i16x8(__m128i toShuffle, __m128i indexes)
__m128i _shuffleVar_i32x4(__m128i toShuffle, __m128i indexes)
__m128i _shuffleVar_i64x2(__m128i toShuffle, __m128i indexes)
```

## Pseudocode

```text
FOR i from 0...total_elements {
    result.element[i] = toShuffle.element[indexes.element[i]]
}

return result 
```
