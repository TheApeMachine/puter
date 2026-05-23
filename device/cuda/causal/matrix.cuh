#ifndef PUTER_DEVICE_CUDA_CAUSAL_MATRIX_CUH
#define PUTER_DEVICE_CUDA_CAUSAL_MATRIX_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "causal.cuh"

#define CAUSAL_CHOLESKY_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    scalarType* output, \
    unsigned int matrixOrder \
) { \
    if (blockIdx.x != 0u || threadIdx.x != 0u) { \
        return; \
    } \
    for (unsigned int index = 0u; index < matrixOrder * matrixOrder; index++) { \
        storeFn(output, index, 0.0f); \
    } \
    for (unsigned int rowIndex = 0u; rowIndex < matrixOrder; rowIndex++) { \
        for (unsigned int colIndex = 0u; colIndex <= rowIndex; colIndex++) { \
            double sum = static_cast<double>(loadFn(input, rowIndex * matrixOrder + colIndex)); \
            for (unsigned int innerIndex = 0u; innerIndex < colIndex; innerIndex++) { \
                sum -= static_cast<double>(loadFn(output, rowIndex * matrixOrder + innerIndex)) * \
                    static_cast<double>(loadFn(output, colIndex * matrixOrder + innerIndex)); \
            } \
            if (rowIndex == colIndex) { \
                if (sum <= 0.0) { \
                    return; \
                } \
                storeFn(output, rowIndex * matrixOrder + colIndex, static_cast<float>(sqrt(sum))); \
            } else { \
                storeFn( \
                    output, \
                    rowIndex * matrixOrder + colIndex, \
                    static_cast<float>(sum / static_cast<double>(loadFn(output, colIndex * matrixOrder + colIndex))) \
                ); \
            } \
        } \
    } \
}

#endif
