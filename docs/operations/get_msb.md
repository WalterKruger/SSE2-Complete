# Get most significant bit of each element (movemask)

Gathers the most significant bit of each vector element and pack them into a scalar bit array.

## Signature

```C
int _getMsb_i8x16(__m128i)
int _getMsb_i16x8(__m128i)
int _getMsb_i32x4(__m128i)
int _getMsb_i64x2(__m128i)
```

## Pseudocode

```text
result.bit[i] = (vector.element[i] >> (ELEMENT_BITS - 1))
```
