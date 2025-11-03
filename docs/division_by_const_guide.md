# Description

Division is one of the most expensive arithmetic operations a CPU can perform. However, if a divisor is known ahead of time it can be replaced with multiplication with a magic number + shift, which tends to be much faster and has constant time performance. This optimization can also be beneficial if the divisor is not known ahead of time but is reused multiple times.

This library includes this optimization for both division and modulo through the `_divP` and `_modP` functions. Notably, our method allows each element to be divided by different divisors unlike other implementations, which divide all elements by the same scalar value.

# Calculating the precomputed divisor(s)

The `_divP` and `_modP` functions expect a `sseCom_divMagic` struct input for the divisor, which is precomputed “magic” constant that is calculated based on the divisors (both use the same constants).

Keep in mind that **calculating the magic numbers may be slower than doing the divisor directly**, so they are most beneficial with divisors known at compile time or when you are dividing multiple vectors using the same divisor(s). The 32-bit precomputation is the only width that substantially more expensive to calculate, however all methods still have the additional overhead of the "magic multiplication" which actually performs the division. For other widths the precomputed methods may perform better even when the divisors are variable and can only be used once if there is a significant gap between when the divisors and numerators are known.

**Note: The 8 and 32-bit methods only works with divisors greater than 1.** Well the 16 & 64-bit `divP` and `modP` methods work with any non-zero divisors.

***

### Different divisors from literals: `_getDivMagic_set`/`setr`

Individually sets all divisors using multiple scalar values. This is designed to be used with **literal constants only** and should compile down to a constant. This uses the same ordering as the `_mm_set_` Intel Intrinsics functions, which sets from most significant element first (reverse order than setting then loading from an array). The `_getDivMagic_setr` works in reverse of that order (like `_mm_setr_`).

Equivalent to: `_div_u32x4(numerator, _mm_set_epi32(div4th, div3rd, div2nd, div1st))`

***

### Same divisor from scalar: `_getDivMagic_set1`

Sets all divisors to be the same magic value based on a single, scalar divisor. The divisor can be either a constant or a variable. 

Equivalent to: `_div_u32x4(numerator, _mm_set1_epi32(divisor))`

***

### Variable divisors from vector: `_getDivMagic`

Calculates the magic constants based on a vector input. This is designed for the case where the divisors are not known ahead of time but will be reused multiple times.

Equivalent to: `_div_u32x4(numerator, divisor)`

# Magic structs info

| Struct name | Element type | Size |
| - | - | - |
| `sseCom_divMagic_u8`  | Unsigned 8-bit int  | 32 bytes |
| `sseCom_divMagic_u16` | Unsigned 16-bit int | 32 bytes |
| `sseCom_divMagic_u32` | Unsigned 32-bit int | 64 bytes |
| `sseCom_divMagic_u64` | Unsigned 64-bit int | 32 bytes |

# Perf: Divide vector by variable scalar

This covers the cases where you want to divide all elements within a vector by the same scalar value that is not known ahead of time/reused. The benchmark measures the time taken for the following idioms to calculate 500 million results at `-O2`:

Normal: `_div(numerator, _mm_set1(divisor))`

Magic: `_divP(numerator, _getDivMagic_set1(divisor))`

<details><summary>Unsigned 8-bit division</summary>

|  | Raptor Lake | Zen3 | Goldmont Plus | Haswell | Ader Lake-N |
| - | - | - | - | - | - |
| Normal         | 1.64 | 2.06 | 13.01 | 4.06 | 5.45 |
| Magic          | 0.69 | 0.71 | 4.88 | 1.68 | 1.14 |
| Relative diff. | +1.38 | +1.90 | +1.67 | +1.42 | +3.78 |
</details>

<details><summary>Unsigned 16-bit division</summary>

|  | Raptor Lake | Zen3 | Goldmont Plus | Haswell | Ader Lake-N |
| - | - | - | - | - | - |
| Normal         | 0.70 | 0.82 | 7.18 | 1.99 | 2.67 |
| Magic          | # | # | 5.16 | 1.82 | 2.32 |
| Relative diff. | # | # | +0.39 | +0.09 | +0.15 |
</details>

<details><summary>Unsigned 32-bit division</summary>
Note: Values for "normal" are adjusted from data which used an outdated version.

|  | Raptor Lake | Zen3 | Goldmont Plus | Haswell | Ader Lake-N |
| - | - | - | - | - | - |
| Normal         | 0.94 | 1.06 | 6.57 | 3.18 | 3.52 |
| Magic          | 1.12 | 0.93 | 7.01 | 4.44 | 3.66 |
| Relative diff. | -0.16 | +0.14 | -0.06 | -0.28 | -0.04 |
</details>

<details><summary>Unsigned 64-bit division</summary>
Note: Values for "normal" are adjusted from data which used an outdated version.

|  | Raptor Lake | Zen4 | Goldmont Plus | Haswell | Ader Lake-N |
| - | - | - | - | - | - |
| Normal         | # | 1.48 | # | 6.55 | 1.69 |
| Magic          | # | 0.74 | # | 3.90 | 1.42 |
| Relative diff. | # | +1.00 | # | +0.68 | +0.19 |
</details>


# Credits
Based on technique by Daniel Lemire, Owen Kaser, and Nathan Kurz from the paper "*Faster Remainder by Direct Computation: Applications to Compilers and Software Libraries*"  [arXiv:1902.01961](https://arxiv.org/abs/1902.01961)

Also, see [libdiv](https://github.com/ridiculousfish/libdivide/) if you only need to divide by a vector scalar value. It may be faster than our methods, but much slower to calculate the magic constants.
