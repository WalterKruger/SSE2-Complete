# Square root

Calculates the integer square root (⌊√n⌋) of each unsigned integer.

## Signature

```C
__m128i _sqrt_u8x16(__m128i)
__m128i _sqrt_u16x8(__m128i)
__m128i _sqrt_u32x4(__m128i)
__m128i _sqrt_u64x2(__m128i)
```

## Pseudocode

```text
return floor(sqrt(x.element))
```
