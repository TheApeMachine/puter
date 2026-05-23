#ifndef PUTER_DEVICE_CUDA_POOL_POOL_CUH
#define PUTER_DEVICE_CUDA_POOL_POOL_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

static constexpr float poolAdaptiveMaxInitF32 = -1.0e30f;

struct Float32PoolStorage {
    __device__ __forceinline__ static float load(const float* values, unsigned int index) {
        return values[index];
    }

    __device__ __forceinline__ static void store(float* values, unsigned int index, float value) {
        values[index] = value;
    }
};

struct Float16PoolStorage {
    __device__ __forceinline__ static __half load(const __half* values, unsigned int index) {
        return values[index];
    }

    __device__ __forceinline__ static void store(__half* values, unsigned int index, __half value) {
        values[index] = value;
    }

    __device__ __forceinline__ static float load_accum(const __half* values, unsigned int index) {
        return __half2float(values[index]);
    }
};

struct BFloat16PoolStorage {
    __device__ __forceinline__ static __nv_bfloat16 load(const __nv_bfloat16* values, unsigned int index) {
        return values[index];
    }

    __device__ __forceinline__ static void store(__nv_bfloat16* values, unsigned int index, __nv_bfloat16 value) {
        values[index] = value;
    }

    __device__ __forceinline__ static float load_accum(const __nv_bfloat16* values, unsigned int index) {
        return __bfloat162float(values[index]);
    }
};

__device__ __forceinline__ void pool2d_decode_index(
    unsigned int index,
    unsigned int outWidth,
    unsigned int outHeight,
    unsigned int channels,
    unsigned int& outCol,
    unsigned int& outRow,
    unsigned int& channel,
    unsigned int& batchIndex
) {
    outCol = index % outWidth;
    outRow = (index / outWidth) % outHeight;
    channel = (index / (outWidth * outHeight)) % channels;
    batchIndex = index / (outWidth * outHeight * channels);
}

__device__ __forceinline__ void pool2d_max_float32(
    const float* input,
    float* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    float value = -CUDART_INF_F;

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            float candidate = Float32PoolStorage::load(input, inputIndex);
            value = fmaxf(value, candidate);
        }
    }

    Float32PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void pool2d_avg_float32(
    const float* input,
    float* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += Float32PoolStorage::load(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    Float32PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void pool2d_max_float16(
    const __half* input,
    __half* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    __half value = __float2half(-CUDART_INF_F);

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            __half candidate = Float16PoolStorage::load(input, inputIndex);
            value = __hmax(value, candidate);
        }
    }

    Float16PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void pool2d_avg_float16(
    const __half* input,
    __half* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += Float16PoolStorage::load_accum(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    Float16PoolStorage::store(out, index, __float2half(value));
}

__device__ __forceinline__ void pool2d_max_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    __nv_bfloat16 value = __float2bfloat16(-CUDART_INF_F);

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            __nv_bfloat16 candidate = BFloat16PoolStorage::load(input, inputIndex);
            value = __hmax(value, candidate);
        }
    }

    BFloat16PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void pool2d_avg_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = outRow * 2U;
    unsigned int startCol = outCol * 2U;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int kernelRow = 0U; kernelRow < 2U; kernelRow++) {
        unsigned int inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (unsigned int kernelCol = 0U; kernelCol < 2U; kernelCol++) {
            unsigned int inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += BFloat16PoolStorage::load_accum(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    BFloat16PoolStorage::store(out, index, __float2bfloat16(value));
}

__device__ __forceinline__ void adaptive_pool2d_max_float32(
    const float* input,
    float* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    float value = poolAdaptiveMaxInitF32;

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            float candidate = Float32PoolStorage::load(input, inputIndex);
            value = fmaxf(value, candidate);
        }
    }

    Float32PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void adaptive_pool2d_avg_float32(
    const float* input,
    float* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += Float32PoolStorage::load(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    Float32PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void adaptive_pool2d_max_float16(
    const __half* input,
    __half* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    __half value = __float2half(poolAdaptiveMaxInitF32);

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            __half candidate = Float16PoolStorage::load(input, inputIndex);
            value = __hmax(value, candidate);
        }
    }

    Float16PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void adaptive_pool2d_avg_float16(
    const __half* input,
    __half* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += Float16PoolStorage::load_accum(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    Float16PoolStorage::store(out, index, __float2half(value));
}

__device__ __forceinline__ void adaptive_pool2d_max_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    __nv_bfloat16 value = __float2bfloat16(poolAdaptiveMaxInitF32);

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            __nv_bfloat16 candidate = BFloat16PoolStorage::load(input, inputIndex);
            value = __hmax(value, candidate);
        }
    }

    BFloat16PoolStorage::store(out, index, value);
}

__device__ __forceinline__ void adaptive_pool2d_avg_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    unsigned int batch,
    unsigned int channels,
    unsigned int inHeight,
    unsigned int inWidth,
    unsigned int outHeight,
    unsigned int outWidth,
    unsigned int index
) {
    unsigned int count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    unsigned int outCol = 0U;
    unsigned int outRow = 0U;
    unsigned int channel = 0U;
    unsigned int batchIndex = 0U;
    pool2d_decode_index(index, outWidth, outHeight, channels, outCol, outRow, channel, batchIndex);

    unsigned int startRow = (outRow * inHeight) / outHeight;
    unsigned int endRow = ((outRow + 1U) * inHeight) / outHeight;
    unsigned int startCol = (outCol * inWidth) / outWidth;
    unsigned int endCol = ((outCol + 1U) * inWidth) / outWidth;
    float value = 0.0f;
    unsigned int elements = 0U;

    for (unsigned int inRow = startRow; inRow < endRow; inRow++) {
        for (unsigned int inCol = startCol; inCol < endCol; inCol++) {
            unsigned int inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            value += BFloat16PoolStorage::load_accum(input, inputIndex);
            elements++;
        }
    }

    if (elements > 0U) {
        value /= static_cast<float>(elements);
    }

    BFloat16PoolStorage::store(out, index, __float2bfloat16(value));
}

#define POOL2D_KERNEL_F32(name, body) \
extern "C" __global__ void name##_float32( \
    const float* input, \
    float* out, \
    unsigned int batch, \
    unsigned int channels, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int count = batch * channels * outHeight * outWidth; \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base); \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base + 1u); \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base + 2u); \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base + 3u); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int index = base + offset; \
        if (index < count) { \
            body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, index); \
        } \
    } \
}

#define POOL2D_KERNEL_F16(name, body) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* out, \
    unsigned int batch, \
    unsigned int channels, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int count = batch * channels * outHeight * outWidth; \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base); \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base + 1u); \
        return; \
    } \
    if (base < count) { \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base); \
    } \
}

#define POOL2D_KERNEL_BF16(name, body) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    __nv_bfloat16* out, \
    unsigned int batch, \
    unsigned int channels, \
    unsigned int inHeight, \
    unsigned int inWidth, \
    unsigned int outHeight, \
    unsigned int outWidth \
) { \
    unsigned int count = batch * channels * outHeight * outWidth; \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base); \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base + 1u); \
        return; \
    } \
    if (base < count) { \
        body(input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, base); \
    } \
}

#endif
