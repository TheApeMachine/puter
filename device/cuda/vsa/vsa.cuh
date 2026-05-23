#ifndef PUTER_DEVICE_CUDA_VSA_VSA_CUH
#define PUTER_DEVICE_CUDA_VSA_VSA_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float vsa_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void vsa_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float vsa_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void vsa_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float vsa_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void vsa_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

#define VSA_BINARY_VECTOR_KERNEL( \
    name, scalarType, vectorType, vectorWidth, loadFn, storeFn, scalarOp, vectorOp \
) \
extern "C" __global__ void name( \
    const scalarType* left, \
    const scalarType* right, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * vectorWidth; \
    if (base + (vectorWidth - 1u) < count) { \
        vectorType leftValue = reinterpret_cast<const vectorType*>(left)[vectorIndex]; \
        vectorType rightValue = reinterpret_cast<const vectorType*>(right)[vectorIndex]; \
        reinterpret_cast<vectorType*>(out)[vectorIndex] = vectorOp(leftValue, rightValue); \
        return; \
    } \
    for (unsigned int index = base; index < count; index++) { \
        storeFn(out, index, scalarOp(loadFn(left, index), loadFn(right, index))); \
    } \
}

static __device__ __forceinline__ float vsa_mul_f32(float left, float right) {
    return left * right;
}

static __device__ __forceinline__ float vsa_add_f32(float left, float right) {
    return left + right;
}

static __device__ __forceinline__ float4 vsa_mul_f4(float4 left, float4 right) {
    return make_float4(left.x * right.x, left.y * right.y, left.z * right.z, left.w * right.w);
}

static __device__ __forceinline__ float4 vsa_add_f4(float4 left, float4 right) {
    return make_float4(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
}

static __device__ __forceinline__ float vsa_mul_f16(float left, float right) {
    return left * right;
}

static __device__ __forceinline__ float vsa_add_f16(float left, float right) {
    return left + right;
}

static __device__ __forceinline__ __half2 vsa_mul_h2(__half2 left, __half2 right) {
    return __hmul2(left, right);
}

static __device__ __forceinline__ __half2 vsa_add_h2(__half2 left, __half2 right) {
    return __hadd2(left, right);
}

static __device__ __forceinline__ float vsa_mul_bf16(float left, float right) {
    return left * right;
}

static __device__ __forceinline__ float vsa_add_bf16(float left, float right) {
    return left + right;
}

static __device__ __forceinline__ __nv_bfloat162 vsa_mul_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hmul2(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 vsa_add_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hadd2(left, right);
}

VSA_BINARY_VECTOR_KERNEL(
    vsa_bind_float32, float, float4, 4u, vsa_load_f32, vsa_store_f32, vsa_mul_f32, vsa_mul_f4
)
VSA_BINARY_VECTOR_KERNEL(
    vsa_bundle_float32, float, float4, 4u, vsa_load_f32, vsa_store_f32, vsa_add_f32, vsa_add_f4
)
VSA_BINARY_VECTOR_KERNEL(
    vsa_bind_float16, __half, __half2, 2u, vsa_load_f16, vsa_store_f16, vsa_mul_f16, vsa_mul_h2
)
VSA_BINARY_VECTOR_KERNEL(
    vsa_bundle_float16, __half, __half2, 2u, vsa_load_f16, vsa_store_f16, vsa_add_f16, vsa_add_h2
)
VSA_BINARY_VECTOR_KERNEL(
    vsa_bind_bfloat16, __nv_bfloat16, __nv_bfloat162, 2u, vsa_load_bf16, vsa_store_bf16, vsa_mul_bf16, vsa_mul_b2
)
VSA_BINARY_VECTOR_KERNEL(
    vsa_bundle_bfloat16, __nv_bfloat16, __nv_bfloat162, 2u, vsa_load_bf16, vsa_store_bf16, vsa_add_bf16, vsa_add_b2
)

#define VSA_PERMUTE_KERNEL(name, scalarType, loadFn, storeFn, inverse) \
extern "C" __global__ void name( \
    const scalarType* input, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count || count == 0u) { \
        return; \
    } \
    unsigned int target = inverse \
        ? (index == 0u ? count - 1u : index - 1u) \
        : (index + 1u == count ? 0u : index + 1u); \
    storeFn(out, target, loadFn(input, index)); \
}

VSA_PERMUTE_KERNEL(vsa_permute_float32, float, vsa_load_f32, vsa_store_f32, false)
VSA_PERMUTE_KERNEL(vsa_inverse_permute_float32, float, vsa_load_f32, vsa_store_f32, true)
VSA_PERMUTE_KERNEL(vsa_permute_float16, __half, vsa_load_f16, vsa_store_f16, false)
VSA_PERMUTE_KERNEL(vsa_inverse_permute_float16, __half, vsa_load_f16, vsa_store_f16, true)
VSA_PERMUTE_KERNEL(vsa_permute_bfloat16, __nv_bfloat16, vsa_load_bf16, vsa_store_bf16, false)
VSA_PERMUTE_KERNEL(vsa_inverse_permute_bfloat16, __nv_bfloat16, vsa_load_bf16, vsa_store_bf16, true)

#endif
