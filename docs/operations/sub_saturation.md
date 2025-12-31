# Saturation subtraction

Subtracts corresponding elements together but clamps their value to their maximum/minimum representable value instead of wrapping.

## Signature

```C
__m128i _subSat_u32x4(__m128i, __m128i)
__m128i _subSat_i32x4(__m128i, __m128i)

__m128i _subSat_u64x2(__m128i, __m128i)
__m128i _subSat_i64x2(__m128i, __m128i)
```

## Pseudocode

### Unsigned

```text
return max(a.element - b.element, 0)
```

### Signed

```text
return min(max(a.element - b.element, ELEMENT_MIN), ELEMENT_MAX)
```

## Future extensions

### SSE4.1

- Unsigned compares are cheaper.
- `Signed 32 and 64-bit`: Overwriting the difference with max/min can be done with a single `blendv`.
