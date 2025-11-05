# Compare greater/less than

Compare elements for greater or less than. If the condition is true, the corresponding element is set to all 1s.

**Note**: Only unsigned comparisons and signed 64-bit are provided as other signed integers are well supported by SSE2.

## Signature

```C
// 8-bit integer
__m128i _cmpGrt_u8x16(__m128i, __m128i)
__m128i _cmpLss_u8x16(__m128i, __m128i)
__m128i _cmpGrtEq_u8x16(__m128i, __m128i)
__m128i _cmpLssEq_u8x16(__m128i, __m128i)

// 16-bit integer
__m128i _cmpGrt_u16x8(__m128i, __m128i)
__m128i _cmpLss_u16x8(__m128i, __m128i)
__m128i _cmpGrtEq_u16x8(__m128i, __m128i)
__m128i _cmpLssEq_u16x8(__m128i, __m128i)

// 32-bit integer
__m128i _cmpGrt_u32x4(__m128i, __m128i)
__m128i _cmpLss_u32x4(__m128i, __m128i)
__m128i _cmpGrtEq_u32x4(__m128i, __m128i)
__m128i _cmpLssEq_u32x4(__m128i, __m128i)

// 64-bit integer
__m128i _cmpEq_i64x2(__m128i, __m128i)

// Unsigned 64-bit integer
__m128i _cmpGrt_u64x2(__m128i, __m128i)
__m128i _cmpLss_u64x2(__m128i, __m128i)
__m128i _cmpGrtEq_u64x2(__m128i, __m128i)
__m128i _cmpLssEq_u64x2(__m128i, __m128i)

// Signed 64-bit integer
__m128i _cmpGrt_i64x2(__m128i, __m128i)
__m128i _cmpLss_i64x2(__m128i, __m128i)
__m128i _cmpGrtEq_i64x2(__m128i, __m128i)
__m128i _cmpLssEq_i64x2(__m128i, __m128i)
```

## Pseudocode

```text
return compareOp(a.element, b.element)? -1 : 0
```
