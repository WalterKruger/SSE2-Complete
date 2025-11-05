# Sign extension (signed to larger signed)

Converts signed integers to double width signed type by extending each element with their sign bit.

You can only extend half of all elements at a time (the upper or lower half based on the `Lo`/`Hi` suffix).

## Signature

```C
__m128i _signExtendLo_i8x16_i16x8(__m128i)
__m128i _signExtendLo_i16x8_i32x4(__m128i)
__m128i _signExtendLo_i32x4_i64x2(__m128i)

__m128i _signExtendHi_i8x16_i16x8(__m128i)
__m128i _signExtendHi_i16x8_i32x4(__m128i)
__m128i _signExtendHi_i32x4_i64x2(__m128i)
```

## Pseudocode

```text
return (doubleWidth_t)x.element
```

## Future extensions

### SSE4.1

- Sign extensions for all types (and to larger widths) are supported directly.
