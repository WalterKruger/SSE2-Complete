# Zero extension (unsigned to larger unsigned)

Converts unsigned integers to double width unsigned by extending each element with zeros.

You can only extend half of all elements at a time (the upper or lower half based on the `Lo`/`Hi` suffix).

## Signature

```C
__m128i _zeroExtendLo_u8x16_i16x8(__m128i)
__m128i _zeroExtendLo_u16x8_i32x4(__m128i)
__m128i _zeroExtendLo_u32x4_i64x2(__m128i)

__m128i _zeroExtendHi_u8x16_i16x8(__m128i)
__m128i _zeroExtendHi_u16x8_i32x4(__m128i)
__m128i _zeroExtendHi_u32x4_i64x2(__m128i)
```

## Pseudocode

```text
return (doubleWidth_t)x.element[i]
```

## Future extensions

### SSE4.1

- Zero extensions for all types (and to larger widths) are supported directly.
