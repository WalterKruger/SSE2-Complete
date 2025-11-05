# Half width via saturation

Converts integer elements to half width by clamping their value to the nearest representation if it can’t be represented in their smaller width. This function takes two arguments:

1. The elements to be placed in the result’s lower half
2. The elements to be placed in the result’s upper half

## Signature

```C
__m128i _satConvert_u32x4_u16x8(__m128i loHalf, __m128i hiHalf)
__m128i _satConvert_u64x2_u32x4(__m128i loHalf, __m128i hiHalf)
```

## Pseudocode

```text
return truncate(min(loHalf, HALF_MAX), min(hiHalf, HALF_MAX))
```

## Future extensions

### SSE4.1

- Signed 32-bit to unsigned 16-bit saturation is directly supported. Might be useful for unsigned to unsigned.

### AVX512-F

- Integer truncation is supported for all types. Can be adapted to saturate by comparing upper to non-zero.
