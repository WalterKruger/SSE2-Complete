# Float to integer conversion (truncation)

Converts floating point elements into integer types via truncation (round towards zero).

If the float’s value is too great in magnitude to be represented by the integer, the result is undefined. However, for converting a negative float into a unsinged integer the result will be equivalent to converting to a signed type then casting to unsigned (only if it is in range of that signed type).

## Signature

```C
// Single precision to 32-bit integer 
__m128i _convert_f32x4_u32x4(__m128)

// Single precision to 64-bit integer 
__m128i _convertLo_f32x4_i64x2(__m128)
__m128i _convertLo_f32x4_u64x2(__m128)

// Double precision to 64-bit integer 
__m128i _convert_f64x2_i32x4(__m128d)
__m128i _convert_f64x2_u32x4(__m128d)

// Double precision to 64-bit integer 
__m128i _convert_f64x2_i64x2(__m128d)
__m128i _convert_f64x2_u64x2(__m128d)
```

## Pseudocode

```text
return convertFloatToInt(x.element)
```

## Future extensions

### SSE4.1

- Selecting between scaled value can be done with a single `blendv`.

### AVX512-F

- Float to unsigned integer conversions directly supported.
