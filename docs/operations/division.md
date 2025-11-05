# Division by a variable

Divides integers by their corresponding divisor element. Performs the conventional floor definition of integer division. Dividing by zero and `INT_MIN/-1` is undefined behavior and may not always produce any arithmetic exceptions.

**Note**: This is relatively expensive. Use the “[division by a constant](division_precompute.md)” version instead if you reuse the same divisor vector or it known at compile time.

## Signature

```C
__m128i _div_u8x16(__m128i, __m128i)
__m128i _div_i8x16(__m128i, __m128i)

__m128i _div_u16x8(__m128i, __m128i)
__m128i _div_i16x8(__m128i, __m128i)

__m128i _div_u32x4(__m128i, __m128i)
__m128i _div_i32x4(__m128i, __m128i)

__m128i _div_u64x2(__m128i, __m128i)
__m128i _div_i64x2(__m128i, __m128i)
```

## Pseudocode

```text
return floor(n.element / d.element)
```

## Future extensions

### SSE4.1

- `8-bit`: Width extension to 32-bit is directly supported.
- `16-bit`: Can replace `_trunc` call with `_mm_packus_epi32`.
- `32-bit`: Check for and overwrite overflow with a single `_mm_blendv_ps`.

### AVX512-VBMI

- `8-bit`: Significantly faster to utilize the "precomputed" method by using `_mm512_permutex2var_epi8` as a lookup table to obtain the magic number. Still need to overwrite division by one. The LUT can be halved as a divisor with MSB set can be replaced with: `(n >= d)? 1 : 0`.
