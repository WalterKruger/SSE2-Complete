# Variable logical shift

Shift each integer element left/right individually by the amount specified by the corresponding element, well shifting in zeros. If the shift amount is greater than the element's width, the amount is modulo the element size. (e.g: `i8 << 10 = i8 << 2`)

## Signature

```C
__m128i _shiftLvar_u8x16(__m128i toShift, __m128i amount)
__m128i _shiftRvar_u8x16(__m128i toShift, __m128i amount)

__m128i _shiftLvar_u16x8(__m128i toShift, __m128i amount)
__m128i _shiftRvar_u16x8(__m128i toShift, __m128i amount)

__m128i _shiftLvar_u32x4(__m128i toShift, __m128i amount)
__m128i _shiftRvar_u32x4(__m128i toShift, __m128i amount)

__m128i _shiftLvar_u64x2(__m128i toShift, __m128i amount)
__m128i _shiftRvar_u64x2(__m128i toShift, __m128i amount)
```

## Pseudocode

### Left

```text
return toShift.element << (amount.element % ELEMENT_BITS)
```

### Right

```text
return toShift.element >> (amount.element % ELEMENT_BITS)
```

## Future extensions

### SSSE3

- `8-bit`: Use a power of two multiply, obtained through using `_mm_shuffle_epi8` as a lookup table.

### SSE4.1

- `32-bit`: The low multiplication in the left shift is directly supported.

### AVX2

- 32 and 64-bit variables shifts are supported directly (although the amount isn't modulo). 16-bit should be implemented using a width extend.

### AVX512-BW

- 16-bit variables shifts are now supported directly (although the amount isn't modulo). 8-bit should be implemented using a width extend.
