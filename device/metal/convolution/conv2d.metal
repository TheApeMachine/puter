#include "../pool/pool.metal"

using namespace metal;

#define CONV1D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inLength [[buffer(6)]], \
    constant uint& outChannels [[buffer(7)]], \
    constant uint& kernelLength [[buffer(8)]], \
    constant uint& outLength [[buffer(9)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv1d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inLength, outChannels, \
        kernelLength, outLength, index \
    ); \
}

#define CONV2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inHeight [[buffer(6)]], \
    constant uint& inWidth [[buffer(7)]], \
    constant uint& outChannels [[buffer(8)]], \
    constant uint& kernelHeight [[buffer(9)]], \
    constant uint& kernelWidth [[buffer(10)]], \
    constant uint& outHeight [[buffer(11)]], \
    constant uint& outWidth [[buffer(12)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv2d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inHeight, inWidth, \
        outChannels, kernelHeight, kernelWidth, outHeight, outWidth, index \
    ); \
}

#define CONV3D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inDepth [[buffer(6)]], \
    constant uint& inHeight [[buffer(7)]], \
    constant uint& inWidth [[buffer(8)]], \
    constant uint& outChannels [[buffer(9)]], \
    constant uint& kernelDepth [[buffer(10)]], \
    constant uint& kernelHeight [[buffer(11)]], \
    constant uint& kernelWidth [[buffer(12)]], \
    constant uint& outDepth [[buffer(13)]], \
    constant uint& outHeight [[buffer(14)]], \
    constant uint& outWidth [[buffer(15)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv3d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inDepth, inHeight, inWidth, \
        outChannels, kernelDepth, kernelHeight, kernelWidth, outDepth, outHeight, outWidth, index \
    ); \
}

#define CONV_TRANSPOSE2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inHeight [[buffer(6)]], \
    constant uint& inWidth [[buffer(7)]], \
    constant uint& outChannels [[buffer(8)]], \
    constant uint& kernelHeight [[buffer(9)]], \
    constant uint& kernelWidth [[buffer(10)]], \
    constant uint& outHeight [[buffer(11)]], \
    constant uint& outWidth [[buffer(12)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv_transpose2d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inHeight, inWidth, \
        outChannels, kernelHeight, kernelWidth, outHeight, outWidth, index \
    ); \
}

CONV1D_KERNEL(conv1d_float32, Float32VisionStorage, float)
CONV1D_KERNEL(conv1d_float16, Float16VisionStorage, half)
CONV1D_KERNEL(conv1d_bfloat16, BFloat16VisionStorage, ushort)

CONV2D_KERNEL(conv2d_float32, Float32VisionStorage, float)
CONV2D_KERNEL(conv2d_float16, Float16VisionStorage, half)
CONV2D_KERNEL(conv2d_bfloat16, BFloat16VisionStorage, ushort)

CONV3D_KERNEL(conv3d_float32, Float32VisionStorage, float)
CONV3D_KERNEL(conv3d_float16, Float16VisionStorage, half)
CONV3D_KERNEL(conv3d_bfloat16, BFloat16VisionStorage, ushort)

CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float32, Float32VisionStorage, float)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float16, Float16VisionStorage, half)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_bfloat16, BFloat16VisionStorage, ushort)
