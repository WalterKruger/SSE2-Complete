# Convert from two's complement to sign-and-magnitude

Converts signed integers from their typical two's complement representation to their equivalent sign-and-magnitude representation.

`-INT_MIN` is undefined.

## Signature

```C
__m128i _toSignAndMag_i32x4(__m128i)
```

## Pseudocode

```text
return convertToSignAndMag(x.element)
```
