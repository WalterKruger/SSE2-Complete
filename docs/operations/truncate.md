# Truncation (Convert to half width by removing upper bits)

Converts integer elements to half width type by discarding their upper bits. This function takes two arguments:

1. The truncated elements to be placed in the result’s lower half
2. The truncated elements to be placed in the result’s upper half

## Signature

```C
__m128i _trunc_u16x8_u8x16(__m128i loHalf, __m128i hiHalf)
__m128i _trunc_u32x4_u16x8(__m128i loHalf, __m128i hiHalf)
__m128i _trunc_u64x2_u32x4(__m128i loHalf, __m128i hiHalf)
```

## Pseudocode

```text
FOR i from 0...total_elements/2 {
    result.half_element[i] = (halfWidth_t)loHalf.element[i]
}
FOR i from 0...total_elements/2 {
    result.half_element[i + total_elements/2] = (halfWidth_t)hiHalf.element[i]
}

return result
```

## Future extensions

### SSE4.1

- `32-bit`: Zero out the upper 16-bit then use `_mm_packus_epi32`.

### AVX512-F

- Integer truncation is directly supported for all types.
