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
