# SSE2 Complete
SIMD on x86 has this bizarre and apparently arbitrary lack of many fundamental operations, particularly regarding integers, making it difficult to work with. Later extensions (like SSE4.1 and especially AVX-512) have gone a long way in improving this. Unfortunately, unlike SSE2 they aren’t available on all x86-64 CPUs requiring projects that utilize them to sacrifice compatibility with older machines or maintain both a SIMD and scalar fallback implementation then dynamic dispatch between them.

This header only library aims to provide fast and efficient software implements of these missing operations only using SIMD instructions that are supported on all 64-bit x86 CPUs. This means that project can safely use this library without increasing their minimum hardware requirements well still benefiting from a wealth of SIMD accelerated operations. It may also be useful as a more performant fallback implementation than scalar instructions.

# Supported operation

This library provides full support for the following operations via the [Intel Intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) `__m128` types:

<picture>
  <img alt="Add operations and their number of instructions" src="external/Instruction matrix minimal.png">
</picture>

...and more! See the documentation or the instruction matrix for a full list of all added operations.
