# Absolute value

Negates each signed integer element when negative, resulting in its unsigned magnitude.

**Note**: Inputing `INT_MIN` results in the same value.

## Signature

```C
__m128i _abs_i8x16(__m128i)
__m128i _abs_i16x8(__m128i)
__m128i _abs_i32x4(__m128i)
__m128i _abs_i64x2(__m128i)
```

## Pseudocode

```text
return (x.element < 0)? -x.element : x.element
```

## Future extensions

### SSSE3

- `8, 16, and 32-bit`: Directly supported.

### SSE4.1

- `64-bit`: Select between input and negation using `_mm_blendv_pd`, which acts based on the sign/MSB.

### AVX512-F

- `64-bit`: Directly supported.
