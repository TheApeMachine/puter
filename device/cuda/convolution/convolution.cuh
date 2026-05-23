#ifndef PUTER_DEVICE_CUDA_CONVOLUTION_CONVOLUTION_CUH
#define PUTER_DEVICE_CUDA_CONVOLUTION_CONVOLUTION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float conv_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void conv_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float conv_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void conv_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float conv_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void conv_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void conv2d_kernel_body(
    const Scalar* input,
    const Scalar* weight,
    const Scalar* bias,
    Scalar* out,
    unsigned int batch,
    unsigned int inChannels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outChannels,
    unsigned int kernelHeight,
    unsigned int kernelWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index,
    LoadFn loadFn,
    StoreFn storeFn
) {
    unsigned int count = batch * outChannels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = index % outWidth;
    unsigned int outRow = (index / outWidth) % outHeight;
    unsigned int outChannel = (index / (outWidth * outHeight)) % outChannels;
    unsigned int batchIndex = index / (outWidth * outHeight * outChannels);
    float accumulator = loadFn(bias, outChannel);
    unsigned int totalPadHeight = outHeight + kernelHeight > inHeight + 1u ?
        outHeight + kernelHeight - inHeight - 1u : 0u;
    unsigned int totalPadWidth = outWidth + kernelWidth > inWidth + 1u ?
        outWidth + kernelWidth - inWidth - 1u : 0u;
    unsigned int padTop = totalPadHeight / 2u;
    unsigned int padLeft = totalPadWidth / 2u;

    for (unsigned int inChannel = 0; inChannel < inChannels; inChannel++) {
        for (unsigned int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
            unsigned int paddedRow = outRow + kernelRow;

            if (paddedRow < padTop) {
                continue;
            }

            unsigned int inRow = paddedRow - padTop;

            if (inRow >= inHeight) {
                continue;
            }

            for (unsigned int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                unsigned int paddedCol = outCol + kernelCol;

                if (paddedCol < padLeft) {
                    continue;
                }

                unsigned int inCol = paddedCol - padLeft;

                if (inCol >= inWidth) {
                    continue;
                }

                unsigned int inputIndex = ((batchIndex * inChannels + inChannel) * inHeight + inRow) *
                    inWidth + inCol;
                unsigned int weightIndex = ((outChannel * inChannels + inChannel) * kernelHeight +
                    kernelRow) * kernelWidth + kernelCol;
                accumulator += loadFn(input, inputIndex) * loadFn(weight, weightIndex);
            }
        }
    }

    storeFn(out, index, accumulator);
}

#define CONV2D_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* weight, \
    const scalarType* bias, \
    scalarType* out, \
    unsigned int batch, \
    unsigned int inChannels, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outChannels, \
    unsigned int kernelHeight, \
    unsigned int kernelWidth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    conv2d_kernel_body( \
        input, weight, bias, out, \
        batch, inChannels, inHeight, inWidth, \
        outChannels, kernelHeight, kernelWidth, outHeight, outWidth, \
        index, loadFn, storeFn \
    ); \
}

CONV2D_KERNEL(conv2d_float32, float, conv_load_f32, conv_store_f32)
CONV2D_KERNEL(conv2d_float16, __half, conv_load_f16, conv_store_f16)
CONV2D_KERNEL(conv2d_bfloat16, __nv_bfloat16, conv_load_bf16, conv_store_bf16)

#define CONV1D_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* weight, \
    const scalarType* bias, \
    scalarType* out, \
    unsigned int batch, \
    unsigned int inChannels, \
    unsigned int inLength, \
    unsigned int outChannels, \
    unsigned int kernelLength, \
    unsigned int outLength \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int count = batch * outChannels * outLength; \
    if (index >= count) { return; } \
    unsigned int outPosition = index % outLength; \
    unsigned int outChannel = (index / outLength) % outChannels; \
    unsigned int batchIndex = index / (outLength * outChannels); \
    float accumulator = loadFn(bias, outChannel); \
    for (unsigned int inChannel = 0; inChannel < inChannels; inChannel++) { \
        for (unsigned int kernelPosition = 0; kernelPosition < kernelLength; kernelPosition++) { \
            unsigned int inputPosition = outPosition + kernelPosition; \
            if (inputPosition >= inLength) { continue; } \
            unsigned int inputIndex = (batchIndex * inChannels + inChannel) * inLength + inputPosition; \
            unsigned int weightIndex = (outChannel * inChannels + inChannel) * kernelLength + kernelPosition; \
            accumulator += loadFn(input, inputIndex) * loadFn(weight, weightIndex); \
        } \
    } \
    storeFn(out, index, accumulator); \
}

#define CONV3D_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* weight, \
    const scalarType* bias, \
    scalarType* out, \
    unsigned int batch, \
    unsigned int inChannels, \
    unsigned int inDepth, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outChannels, \
    unsigned int kernelDepth, \
    unsigned int kernelHeight, \
    unsigned int kernelWidth, \
    unsigned int outDepth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int count = batch * outChannels * outDepth * outHeight * outWidth; \
    if (index >= count) { return; } \
    unsigned int outCol = index % outWidth; \
    unsigned int outRow = (index / outWidth) % outHeight; \
    unsigned int outPlane = (index / (outWidth * outHeight)) % outDepth; \
    unsigned int outChannel = (index / (outWidth * outHeight * outDepth)) % outChannels; \
    unsigned int batchIndex = index / (outWidth * outHeight * outDepth * outChannels); \
    float accumulator = loadFn(bias, outChannel); \
    for (unsigned int inChannel = 0; inChannel < inChannels; inChannel++) { \
        for (unsigned int kernelPlane = 0; kernelPlane < kernelDepth; kernelPlane++) { \
            unsigned int inPlane = outPlane + kernelPlane; \
            if (inPlane >= inDepth) { continue; } \
            for (unsigned int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) { \
                unsigned int inRow = outRow + kernelRow; \
                if (inRow >= inHeight) { continue; } \
                for (unsigned int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) { \
                    unsigned int inCol = outCol + kernelCol; \
                    if (inCol >= inWidth) { continue; } \
                    unsigned int inputIndex = (((batchIndex * inChannels + inChannel) * inDepth + \
                        inPlane) * inHeight + inRow) * inWidth + inCol; \
                    unsigned int weightIndex = (((outChannel * inChannels + inChannel) * kernelDepth + \
                        kernelPlane) * kernelHeight + kernelRow) * kernelWidth + kernelCol; \
                    accumulator += loadFn(input, inputIndex) * loadFn(weight, weightIndex); \
                } \
            } \
        } \
    } \
    storeFn(out, index, accumulator); \
}

#define CONV_TRANSPOSE2D_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* weight, \
    const scalarType* bias, \
    scalarType* out, \
    unsigned int batch, \
    unsigned int inChannels, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outChannels, \
    unsigned int kernelHeight, \
    unsigned int kernelWidth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int count = batch * outChannels * outHeight * outWidth; \
    if (index >= count) { return; } \
    unsigned int outCol = index % outWidth; \
    unsigned int outRow = (index / outWidth) % outHeight; \
    unsigned int outChannel = (index / (outWidth * outHeight)) % outChannels; \
    unsigned int batchIndex = index / (outWidth * outHeight * outChannels); \
    float accumulator = loadFn(bias, outChannel); \
    for (unsigned int inChannel = 0; inChannel < inChannels; inChannel++) { \
        for (unsigned int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) { \
            if (outRow < kernelRow) { continue; } \
            unsigned int inRow = outRow - kernelRow; \
            if (inRow >= inHeight) { continue; } \
            for (unsigned int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) { \
                if (outCol < kernelCol) { continue; } \
                unsigned int inCol = outCol - kernelCol; \
                if (inCol >= inWidth) { continue; } \
                unsigned int inputIndex = ((batchIndex * inChannels + inChannel) * inHeight + inRow) * \
                    inWidth + inCol; \
                unsigned int weightIndex = ((inChannel * outChannels + outChannel) * kernelHeight + \
                    kernelRow) * kernelWidth + kernelCol; \
                accumulator += loadFn(input, inputIndex) * loadFn(weight, weightIndex); \
            } \
        } \
    } \
    storeFn(out, index, accumulator); \
}

CONV1D_KERNEL(conv1d_float32, float, conv_load_f32, conv_store_f32)
CONV1D_KERNEL(conv1d_float16, __half, conv_load_f16, conv_store_f16)
CONV1D_KERNEL(conv1d_bfloat16, __nv_bfloat16, conv_load_bf16, conv_store_bf16)

CONV3D_KERNEL(conv3d_float32, float, conv_load_f32, conv_store_f32)
CONV3D_KERNEL(conv3d_float16, __half, conv_load_f16, conv_store_f16)
CONV3D_KERNEL(conv3d_bfloat16, __nv_bfloat16, conv_load_bf16, conv_store_bf16)

CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float32, float, conv_load_f32, conv_store_f32)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float16, __half, conv_load_f16, conv_store_f16)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_bfloat16, __nv_bfloat16, conv_load_bf16, conv_store_bf16)

#endif
