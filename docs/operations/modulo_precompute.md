# Modulo by a constant

Calculates the modulo (`n%d`) using a precomputed magic number. See the [Division by a constant guide](../division_by_const_guide.md).

## Signature

```C
__m128i _modP_u8x16(__m128i n, struct sseCom_divMagic_u8 *magic, __m128i d)
__m128i _modP_u16x8(__m128i n, struct sseCom_divMagic_u16 *magic)
__m128i _modP_u32x4(__m128i n, struct sseCom_divMagic_u32 *magic, __m128i d)
__m128i _modP_u64x2(__m128i n, struct sseCom_divMagic_u64 *magic)
```

## Pseudocode

```text
return modByPrecompute(n.element, magic.element)
```

## Future extensions

See the [division page](division_precompute.md).
