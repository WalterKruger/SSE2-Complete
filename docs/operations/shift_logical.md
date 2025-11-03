# Logical shift

Shift all integer elements left or right by the amount specified by the immediate value, well shifting in zeros.

## Signature

```C
__m128i _shiftL_u8x16(__m128i toShift, imm8 amount)
__m128i _shiftR_u8x16(__m128i toShift, imm8 amount)
```

## Pseudocode

```text
return (amount >= ELEMENT_BITS)? 0 : (toShift.element >> amount)
```
