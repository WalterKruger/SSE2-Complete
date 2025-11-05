# Negation

Negates each signed integer element.

## Signature

```C
__m128i _negate_i8x16(__m128i)
__m128i _negate_i16x8(__m128i)
__m128i _negate_i32x4(__m128i)
__m128i _negate_i64x2(__m128i)
```

## Pseudocode

```text
return -(x.element)
```
