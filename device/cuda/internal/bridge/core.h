#ifndef PUTER_DEVICE_CUDA_INTERNAL_BRIDGE_CORE_H
#define PUTER_DEVICE_CUDA_INTERNAL_BRIDGE_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define CUDA_STATUS_MESSAGE_BYTES 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CUDADeviceRef;
typedef void* CUDABufferRef;
typedef void* CUDAStreamRef;
typedef void* CUDAEventRef;
typedef void* CUDAModuleRef;
typedef void* CUDAKernelRef;

typedef struct CUDAStatus {
    int code;
    char message[CUDA_STATUS_MESSAGE_BYTES];
} CUDAStatus;

typedef enum CUDAElementDType {
    CUDAElementDTypeFloat32 = 0,
    CUDAElementDTypeFloat16 = 1,
    CUDAElementDTypeBFloat16 = 2,
    CUDAElementDTypeFloat64 = 3,
    CUDAElementDTypeFloat8E4M3 = 4,
    CUDAElementDTypeFloat8E5M2 = 5,
} CUDAElementDType;

typedef enum CUDAUnaryFloat32Op {
    CUDAUnaryFloat32Relu = 0,
    CUDAUnaryFloat32Abs = 1,
    CUDAUnaryFloat32Neg = 2,
    CUDAUnaryFloat32Square = 3,
    CUDAUnaryFloat32Recip = 4,
    CUDAUnaryFloat32Sqrt = 5,
    CUDAUnaryFloat32Sign = 6,
    CUDAUnaryFloat32Rsqrt = 7,
    CUDAUnaryFloat32Exp = 8,
    CUDAUnaryFloat32Log = 9,
    CUDAUnaryFloat32Sin = 10,
    CUDAUnaryFloat32Cos = 11,
    CUDAUnaryFloat32Tanh = 12,
    CUDAUnaryFloat32Sigmoid = 13,
    CUDAUnaryFloat32Silu = 14,
    CUDAUnaryFloat32Swish = 15,
    CUDAUnaryFloat32Softsign = 16,
    CUDAUnaryFloat32ELU = 17,
    CUDAUnaryFloat32SELU = 18,
    CUDAUnaryFloat32LeakyReLU = 19,
    CUDAUnaryFloat32HardSigmoid = 20,
    CUDAUnaryFloat32HardSwish = 21,
    CUDAUnaryFloat32Gelu = 22,
    CUDAUnaryFloat32Log1p = 23,
    CUDAUnaryFloat32Expm1 = 24,
    CUDAUnaryFloat32CELU = 25,
    CUDAUnaryFloat32Softplus = 26,
    CUDAUnaryFloat32Mish = 27,
    CUDAUnaryFloat32LogSigmoid = 28,
    CUDAUnaryFloat32GeluTanh = 29,
    CUDAUnaryFloat32HardTanh = 30,
    CUDAUnaryFloat32HardGelu = 31,
    CUDAUnaryFloat32QuickGelu = 32,
    CUDAUnaryFloat32TanhShrink = 33,
} CUDAUnaryFloat32Op;

void cuda_status_clear(CUDAStatus* status);

void cuda_status_set(CUDAStatus* status, int code, const char* message);

CUDAKernelRef cuda_bridge_resolve_kernel(
    CUDADeviceRef contextRef,
    const char* moduleSource,
    const char* kernelName,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
