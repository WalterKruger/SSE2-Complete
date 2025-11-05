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

## Future extensions

These shuffles are vastly slower than hardware, so **replacing them with implementations utilizing newer extensions should be prioritized**.

### SSSE3

- Variable shuffles are now supported via `_mm_shuffle_epi8`. 8, 16, and maybe 32-bit should be implemented using it. The larger widths should duplicate their indexes to all their lane's bytes and be offsetted. Remember to deal with the zeroing behaviour of the index's MSB.

### AVX

- 32 and 64-bit are now supported directly (via the floating-point intrinsics). 16-bit can be implemented using them.

### AVX512-BW

- 16-bit is directly supported.

### AVX512-VBMI

- Unlike SSSE3's shuffle, `_mm_permutexvar_epi8` doesn't zero elements when the index's MSB is set.
