# Modulo by a variable

Calculates the remainder/modulo (`n%d`) between corresponding unsigned integer elements. If your divisor is zero, the result is undefined.

**Note**: This is relatively expensive. Use the “[modulo by a constant](modulo_precompute.md)” version instead if you reuse the same divisor vector or it known at compile time.

## Signature

```C
__m128i _mod_u8x16(__m128i, __m128i)
__m128i _mod_u16x8(__m128i, __m128i)
__m128i _mod_u32x4(__m128i, __m128i)
__m128i _mod_u64x2(__m128i, __m128i)
```

## Pseudocode

```text
return n.element - d.element * floor(n.element / d.element)
```
