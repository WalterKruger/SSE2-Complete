# Maximum

Returns the element which is larger (or equal).

## Signature

```C
__m128i _max_i8x16(__m128i, __m128i)
__m128i _max_i32x4(__m128i, __m128i)
```

## Pseudocode

```text
return (a.element > b.element)? a.element : b.elemenet 
```

## Future extensions

### SSE4.1

- i8, u16, u32, and i32 now directly supported.

### AVX512-F

- i64 and u64 now directly supported.
