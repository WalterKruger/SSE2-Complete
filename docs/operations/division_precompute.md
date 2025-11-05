# Division by a constant

*See the [Division by a constant guide](../division_by_const_guide.md) for a more detailed description*

Divides unsigned integers by precomputed divisor magic vector. Each numerator element can have a unique divisor, they are not required to be the same.

You should either use a compile time constant for the divisor or only calculate the magic numbers once and then reuse them multiple times. Calculating the magic number is more expensive than doing the division normally. To obtain the precomputed magic vector, you need to use either one of the following functions:

- `_getDivMagic`: Calculate the magic numbers based on a variable vector input. Use this is you don’t know your divisors at compile time.
- `_getDivMagic _set`: Manually set each individual divisor based on scalar values. Sets the most significant element first to least significant **Only use this for compile time constants!**
- `_getDivMagic _set1`: Set all divisors to be a single scalar value. It shouldn’t be that bad to use a runtime value.

**Note**: This should be substantially faster than the regular division function if your divisors are known ahead of time or are reused multiple times. Keep in mind calculating the magic numbers might be more expensive than doing regular division.

The 8 and 32-bit methods **only work with divisors greater than 1**! Well the 16 & 64-bit functions works with only non-zero value.

## Signature

```C
__m128i _divP_u8x16(__m128i n, struct sseCom_divMagic_u8 *magic)
__m128i _divP_u16x8(__m128i n, struct sseCom_divMagic_u16 *magic)
__m128i _divP_u32x4(__m128i n, struct sseCom_divMagic_u32 *magic)
__m128i _divP_u64x2(__m128i n, struct sseCom_divMagic_u64 *magic)
```

## Pseudocode

```text
return divByPrecompute(n.element, magic.element)
```

### SSE4.1

- `32-bit`: Faster to use `(UMAX_32 / d) * n + (almostRem >= d)` method as low multiplies are directly supported.
