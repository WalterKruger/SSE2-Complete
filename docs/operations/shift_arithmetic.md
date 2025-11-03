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
