#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_UNARY_MACROS_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_UNARY_MACROS_CUH

#define ACTIVATION_UNARY_KERNEL_F32(name, op_f4, op_f1) \
extern "C" __global__ void name##_float32( \
    const float* inputRaw, \
    float* outputRaw, \
    unsigned int count \
) { \
    const float4* inputVector = reinterpret_cast<const float4*>(inputRaw); \
    float4* outputVector = reinterpret_cast<float4*>(outputRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        outputVector[vectorIndex] = op_f4(inputVector[vectorIndex]); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            outputRaw[scalarIndex] = op_f1(inputRaw[scalarIndex]); \
        } \
    } \
}

#define ACTIVATION_UNARY_KERNEL_F16(name, op_h2, op_h1) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __half2 value = *reinterpret_cast<const __half2*>(&input[base]); \
        *reinterpret_cast<__half2*>(&output[base]) = op_h2(value); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_h1(input[base]); \
    } \
}

#define ACTIVATION_UNARY_KERNEL_BF16(name, op_b2, op_b1) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    __nv_bfloat16* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&input[base]); \
        *reinterpret_cast<__nv_bfloat162*>(&output[base]) = op_b2(value); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_b1(input[base]); \
    } \
}

#endif
