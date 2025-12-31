# Shuffle Low & High

Using a bit mask known at compile time, select elements from the first vector to be the result's lower elements and use another mask to select the second vector to be the result's upper elements. You can use the provided macro `_MM_SHUFHALF(idx1, idx0)` as the mask input to use index notation instead.

Setting the the upper 4 bits of `loIndexes` will result in unexpected behaviour (see pseudocode).

## Signature

```C
__m128i _shuffleLoHi_i32x4(__m128i loPart, imm8 loIndexes, __m128i hiPart, imm8 hiIndexes)
```

## Pseudocode

```text
idx[0:7] = (hiIndexes[0:4] << 4) | loIndexes[0:7]

FOR i from 0...2 {
    result.qword[i] = loPart.qword[idx[i:i+2]]
}
FOR i from 0...2 {
    result.qword[i+2] = hiPart.qword[idx[i+4:i+6]]
}

return result
```
