#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_BINARY_MACROS_CUH
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_BINARY_MACROS_CUH

#define ELEMENTWISE_BINARY_KERNEL_F32(name, op_f4, op_f1) \
extern "C" __global__ void name##_float32( \
    const float* leftRaw, \
    const float* rightRaw, \
    float* outputRaw, \
    unsigned int count \
) { \
    const float4* leftVector = reinterpret_cast<const float4*>(leftRaw); \
    const float4* rightVector = reinterpret_cast<const float4*>(rightRaw); \
    float4* outputVector = reinterpret_cast<float4*>(outputRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        outputVector[vectorIndex] = op_f4(leftVector[vectorIndex], rightVector[vectorIndex]); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            outputRaw[scalarIndex] = op_f1(leftRaw[scalarIndex], rightRaw[scalarIndex]); \
        } \
    } \
}

#define ELEMENTWISE_BINARY_KERNEL_F16(name, op_h2, op_h1) \
extern "C" __global__ void name##_float16( \
    const __half* left, \
    const __half* right, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __half2 leftValue = *reinterpret_cast<const __half2*>(&left[base]); \
        __half2 rightValue = *reinterpret_cast<const __half2*>(&right[base]); \
        *reinterpret_cast<__half2*>(&output[base]) = op_h2(leftValue, rightValue); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_h1(left[base], right[base]); \
    } \
}

#define ELEMENTWISE_BINARY_KERNEL_BF16(name, op_b2, op_b1) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* left, \
    const __nv_bfloat16* right, \
    __nv_bfloat16* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __nv_bfloat162 leftValue = *reinterpret_cast<const __nv_bfloat162*>(&left[base]); \
        __nv_bfloat162 rightValue = *reinterpret_cast<const __nv_bfloat162*>(&right[base]); \
        *reinterpret_cast<__nv_bfloat162*>(&output[base]) = op_b2(leftValue, rightValue); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_b1(left[base], right[base]); \
    } \
}

#endif
