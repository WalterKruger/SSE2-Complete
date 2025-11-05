# Shuffle Low & High

Using a bit mask known at compile time, select elements from the first vector to be the result’s lower elements and use another mask to select the second vector to be the result’s upper elements. You can use the provided macro `_MM_SHUFHALF(idx1, idx0)` as the mask input to use index notation instead.

## Signature

```C
__m128i _shuffleLoHi_i32x4(__m128i loPart, imm8 loIndexes, __m128i hiPart, imm8 hiIndexes)
```

## Pseudocode

```text
FOR i from 0...2 {
    result.qword[i] = loPart.qword[loIndexes[i:i+2]]
}
FOR i from 0...2 {
    result.qword[i+2] = hiPart.qword[hiIndexes[i:i+2]]
}

return result
```
