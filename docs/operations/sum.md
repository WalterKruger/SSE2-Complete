# Sums all elements

Adds all elements within a vector together and returns a single scalar value.

## Signature

```C
uint16_t _sum_u8x16(__m128i)
```

## Pseudocode

```text
integer sum = 0
FOREACH element in vector {
    sum += element
}

return sum
```
