# Low multiplication

Multiplies corresponding elements and returns the lower bits of the double width intermediate product (e.g: `i8 * i8 => (i8)i16`). This is equivalent to performing a scalar multiplication.

## Signature

Although variations exist for signed and unsigned integers, both of them behave identically.

```C
__m128i _mulLo_u8x16(__m128i, __m128i)
__m128i _mulLo_i8x16(__m128i, __m128i)

__m128i _mulLo_u16x8(__m128i, __m128i)
__m128i _mulLo_i16x8(__m128i, __m128i)

__m128i _mulLo_u32x4(__m128i, __m128i)
__m128i _mulLo_i32x4(__m128i, __m128i)

__m128i _mulLo_u64x2(__m128i, __m128i)
__m128i _mulLo_i64x2(__m128i, __m128i)
```

## Pseudocode

```text
return (element_t)(a.element * b.element)
```

## Future extensions

### SSE4.1

- 32-bit directly supported.

### AVX512-DQ

- 64-bit directly supported.
