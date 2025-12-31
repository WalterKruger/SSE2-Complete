# All supported operations

Each public function in this library will have the following signature:

`_<str:operation name>_<char:type><int:bit width>x<int:elements>[str:Lo|Hi]`

This list details the `_<str:operation name>` portion of that signature. Suffixed to all those is their data type which is one of the following:

|  | 8-bit | 16-bit | 32-bit | 64-bit |
| :-: | :-: | :-: | :-: | :-: |
| **Signed int**  | `i8x16` | `i16x8` | `i32x4` | `i64x2` |
| **Unsigned int**  | `u8x16` | `u16x8` | `u32x4` | `u64x2` |
| **Floating point** |  |  | `f32x4` | `f64x2` |
| **Elements/vector** | 16 | 8 | 4 | 2 |

## Operation list

- [Sum all elements](operations/sum.md)
- [Square root](operations/square_root.md)
- [Saturation addition](operations/add_saturation.md)
- [Saturation subtraction](operations/sub_saturation.md)
- [Compare greater/less than](operations/compare.md)
- [Maximum](operations/maximum.md)
- [Zero extension (unsigned to larger unsigned)](operations/zero_extend.md)
- [Sign extension (signed to larger signed)](operations/sign_extend.md)
- [Truncation (Convert to half width by removing upper bits)](operations/truncate.md)
- [Half width via saturation](operations/truncate_saturate.md)
- [Integer to float conversion](operations/int_to_float_convert.md)
- [Float to integer conversion (truncation)](operations/float_to_int_convert.md)
- [Convert from two’s complement to sign-and-magnitude](operations/to_signandmag.md)
- [Division by a variable](operations/division.md)
- [Modulo by a variable](operations/modulo.md)
- [Division by a constant](operations/division_precompute.md)
- [Modulo by a constant](operations/modulo_precompute.md)
- [Get most significant bit of each element (movemask)](operations/get_msb.md)
- [Low multiplication](operations/multiply_low.md)
- [High multiplication](operations/multiply_high.md)
- [Full multiplication](operations/multiply_high.md)
- [Negation](operations/negation.md)
- [Conditional negation](operations/negation_conditional.md)
- [Logical shift](operations/shift_logical.md)
- [Arithmetic shift](operations/shift_arithmetic.md)
- [Variable logical shift](operations/shift_variable.md)
- [Shuffle Low & High](operations/shuffle_lohi.md)
- [Variable shuffle](operations/shuffle_variable.md)
