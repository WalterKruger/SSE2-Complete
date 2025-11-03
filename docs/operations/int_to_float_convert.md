# Integer to float conversion

Converts integer element to a floating-point type.

When converting between different widths:

1. Float width < integer: Their values are placed in the results lower half and their upper half is zeroed.
2. Float width > integer: Their will be two functions for either converting only the upper input or lower integers to floats.

## Signature

### Integer to single precision

```C
// 16-bit integer to single precision
__m128 _convertLo_u16x8_f32x4(__m128i)
__m128 _convertHi_u16x8_f32x4(__m128i)
__m128 _convertLo_i16x8_f32x4(__m128i)
__m128 _convertHi_i16x8_f32x4(__m128i)

// 32-bit integer to single precision
__m128 _convert_u32x4_f32x4(__m128i)
__m128 _convert_i32x4_f32x4(__m128i)

// 64-bit integer to single precision
__m128 _convert_u64x2_f32x4(__m128i)
__m128 _convert_i64x2_f32x4(__m128i)
```

### Integer to double precision

```C
// 32-bit integer to double precision
__m128d _convertLo_u32x4_f64x2(__m128i)
__m128d _convertHi_u32x4_f64x2(__m128i)
__m128d _convertLo_i32x4_f64x2(__m128i)

// 64-bit integer to double precision
__m128d _convert_u64x2_f64x2(__m128i)
__m128d _convert_i64x2_f64x2(__m128i)
```

## Pseudocode

```text
return convertIntToFloat(x.element)
```
