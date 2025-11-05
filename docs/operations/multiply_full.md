# Full multiplication

Multiplies corresponding elements and returns their full, double width product. (e.g: `i8 * i8 => i16`)

You can only multiply half of the elements at a time (the upper or lower half based on the `Lo`/`Hi` suffix).

## Signature

```C
// 8-bit integer
__m128i _mulFull_u8x16Lo(__m128i, __m128i)
__m128i _mulFull_u8x16Hi(__m128i, __m128i)
__m128i _mulFull_i8x16Lo(__m128i, __m128i)
__m128i _mulFull_i8x16Hi(__m128i, __m128i)

// 16-bit integer
__m128i _mulFull_u16x16Lo(__m128i, __m128i)
__m128i _mulFull_u16x16Hi(__m128i, __m128i)
__m128i _mulFull_i16x16Even(__m128i, __m128i)
__m128i _mulFull_i16x16Odd(__m128i, __m128i)

// 32-bit integer
__m128i _mulFull_u32x4Even(__m128i, __m128i)
__m128i _mulFull_i32x4Even(__m128i, __m128i)
```

## Pseudocode

```text
result.doubleElement = (doubleWidth_t)a.element * (doubleWidth_t)b.element
```

## Future extensions

### SSE4.1

- Signed 32-bit now directly supported (for the even lanes).
