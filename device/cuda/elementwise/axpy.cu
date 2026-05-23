#include "elementwise.cuh"

extern "C" __global__ void axpy_float32(
    float* yRaw,
    const float* xRaw,
    unsigned int count,
    float alpha
) {
    float4* yVector = reinterpret_cast<float4*>(yRaw);
    const float4* xVector = reinterpret_cast<const float4*>(xRaw);
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = vectorIndex * 4u;

    if (base + 3u < count) {
        float4 xValue = xVector[vectorIndex];
        float4 yValue = yVector[vectorIndex];
        yVector[vectorIndex] = make_float4(
            yValue.x + alpha * xValue.x,
            yValue.y + alpha * xValue.y,
            yValue.z + alpha * xValue.z,
            yValue.w + alpha * xValue.w
        );
        return;
    }

    for (unsigned int offset = 0u; offset < 4u; offset++) {
        unsigned int scalarIndex = base + offset;

        if (scalarIndex < count) {
            yRaw[scalarIndex] += alpha * xRaw[scalarIndex];
        }
    }
}

extern "C" __global__ void axpy_float16(
    __half* y,
    const __half* x,
    unsigned int count,
    float alpha
) {
    __half alphaHalf = __float2half(alpha);
    __half2 alphaPair = __half2half2(alphaHalf);
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = pairIndex * 2u;

    if (base + 1u < count) {
        __half2 xValue = *reinterpret_cast<const __half2*>(&x[base]);
        __half2 yValue = *reinterpret_cast<const __half2*>(&y[base]);
        *reinterpret_cast<__half2*>(&y[base]) = __hfma2(alphaPair, xValue, yValue);
        return;
    }

    if (base < count) {
        y[base] = __hadd(y[base], __hmul(alphaHalf, x[base]));
    }
}

extern "C" __global__ void axpy_bfloat16(
    __nv_bfloat16* y,
    const __nv_bfloat16* x,
    unsigned int count,
    float alpha
) {
    __nv_bfloat16 alphaBf16 = __float2bfloat16(alpha);
    __nv_bfloat162 alphaPair = __bfloat162bfloat162(alphaBf16);
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = pairIndex * 2u;

    if (base + 1u < count) {
        __nv_bfloat162 xValue = *reinterpret_cast<const __nv_bfloat162*>(&x[base]);
        __nv_bfloat162 yValue = *reinterpret_cast<const __nv_bfloat162*>(&y[base]);
        *reinterpret_cast<__nv_bfloat162*>(&y[base]) = __hfma2(alphaPair, xValue, yValue);
        return;
    }

    if (base < count) {
        y[base] = __hadd(y[base], __hmul(alphaBf16, x[base]));
    }
}
