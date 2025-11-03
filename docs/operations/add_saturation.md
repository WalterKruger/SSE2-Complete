# Saturation addition

Add corresponding elements together but clamps their value to their maximum/minimum representable value instead of wrapping.

## Signature

```C
__m128i _addSat_u32x4(__m128i, __m128i)
__m128i _addSat_i32x4(__m128i, __m128i)

__m128i _addSat_u64x2(__m128i, __m128i)
__m128i _addSat_i64x2(__m128i, __m128i)
```

## Pseudocode

### Unsigned

```text
return min(a.element + b.element, ELEMENT_MAX)
```

### Signed

```text
return max(min(a.element + b.element, ELEMENT_MAX), ELEMENT_MIN)
```
