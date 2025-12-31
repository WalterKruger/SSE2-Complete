# High multiplication

Multiplies corresponding elements and returns the upper bits of the double width intermediate product (e.g: `i8 * i8 => i16 >> 8`).

## Signature

```C
__m128i _mulHi_u8x16(__m128i, __m128i)
__m128i _mulHi_i8x16(__m128i, __m128i)

__m128i _mulHi_u16x8(__m128i, __m128i)
__m128i _mulHi_i16x8(__m128i, __m128i)

__m128i _mulHi_u32x4(__m128i, __m128i)
__m128i _mulHi_i32x4(__m128i, __m128i)

__m128i _mulHi_u64x2(__m128i, __m128i)
__m128i _mulHi_i64x2(__m128i, __m128i)
```

## Pseudocode

```text
return ((doubleWidth_t)a.element * (doubleWidth_t)b.element) >> ELEMENT_BITS
```

## Future extensions

### SSE4.1

- Signed 32-bit can use `_mm_mul_epi32` like unsigned version.

### AVX512-IFMA52

- Might be useful...
