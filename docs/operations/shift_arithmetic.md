# Arithmetic shift

Shift all integer elements right by the amount specified by the immediate value, well shifting in their sign bit.

## Signature

```C
__m128i _signShiftR_i8x16(__m128i toShift, imm8 amount)
__m128i _signShiftR_i64x2(__m128i toShift, imm8 amount)
```

## Pseudocode

```text
return (amount >= ELEMENT_BITS)? 0 : (signExtend(toShift.element) >> amount)
```

## Future extensions

### AVX512-F

- 64-bit directly supported.

### AVX512-GFNI

- 8-bit can be implemented using a single `_mm_gf2p8affine_epi64_epi8` with a particular matrix. See [Wunkolo’s gf2p8affine-based 8-bit shifts]( https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/).
