# Conditional negation

If the corresponding conditional mask input is set to all ones, the element is negated.

**Note**: The mask must either be all ones or all zeros.

## Signature

```C
__m128i _condNegate_i8x16(__m128i toNegate, __m128i condiationMask)
__m128i _condNegate_i16x8(__m128i toNegate, __m128i condiationMask)
__m128i _condNegate_i32x4(__m128i toNegate, __m128i condiationMask)
__m128i _condNegate_i64x2(__m128i toNegate, __m128i condiationMask)
```

## Pseudocode

```text
return (toNegate.element ^ conditionMask) - conditionMask
```
