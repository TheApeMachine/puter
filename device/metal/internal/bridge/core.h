#ifndef PUTER_DEVICE_METAL_INTERNAL_BRIDGE_CORE_H
#define PUTER_DEVICE_METAL_INTERNAL_BRIDGE_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define METAL_STATUS_MESSAGE_BYTES 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MetalDeviceRef;
typedef void* MetalBufferRef;

typedef struct MetalStatus {
    int code;
    char message[METAL_STATUS_MESSAGE_BYTES];
} MetalStatus;

typedef enum MetalUnaryFloat32Op {
    MetalUnaryFloat32Relu = 0,
    MetalUnaryFloat32Abs = 1,
    MetalUnaryFloat32Neg = 2,
    MetalUnaryFloat32Square = 3,
    MetalUnaryFloat32Recip = 4,
    MetalUnaryFloat32Sqrt = 5,
    MetalUnaryFloat32Sign = 6,
    MetalUnaryFloat32Rsqrt = 7,
    MetalUnaryFloat32Exp = 8,
    MetalUnaryFloat32Log = 9,
    MetalUnaryFloat32Sin = 10,
    MetalUnaryFloat32Cos = 11,
    MetalUnaryFloat32Tanh = 12,
    MetalUnaryFloat32Sigmoid = 13,
    MetalUnaryFloat32Silu = 14,
    MetalUnaryFloat32Swish = 15,
    MetalUnaryFloat32Softsign = 16,
    MetalUnaryFloat32ELU = 17,
    MetalUnaryFloat32SELU = 18,
    MetalUnaryFloat32LeakyReLU = 19,
    MetalUnaryFloat32HardSigmoid = 20,
    MetalUnaryFloat32HardSwish = 21,
    MetalUnaryFloat32Gelu = 22,
    MetalUnaryFloat32Log1p = 23,
    MetalUnaryFloat32Expm1 = 24,
    MetalUnaryFloat32CELU = 25,
    MetalUnaryFloat32Softplus = 26,
    MetalUnaryFloat32Mish = 27,
    MetalUnaryFloat32LogSigmoid = 28,
    MetalUnaryFloat32GeluTanh = 29,
    MetalUnaryFloat32HardTanh = 30,
    MetalUnaryFloat32HardGelu = 31,
    MetalUnaryFloat32QuickGelu = 32,
    MetalUnaryFloat32TanhShrink = 33,
} MetalUnaryFloat32Op;

typedef enum MetalElementDType {
    MetalElementDTypeFloat32 = 0,
    MetalElementDTypeFloat16 = 1,
    MetalElementDTypeBFloat16 = 2,
    MetalElementDTypeFloat64 = 3,
} MetalElementDType;

MetalDeviceRef metal_open_default_device(
    const uint8_t* libraryBytes,
    long long libraryLength,
    MetalStatus* status
);

void metal_device_release(MetalDeviceRef contextRef);

long long metal_recommended_max_working_set(MetalDeviceRef contextRef);

MetalBufferRef metal_buffer_new_shared(MetalDeviceRef contextRef, long long bytes);

void metal_buffer_release(MetalBufferRef bufferRef);

void* metal_buffer_contents(MetalBufferRef bufferRef);

void metal_device_wait_idle(MetalDeviceRef contextRef);

void metal_layer_begin(MetalDeviceRef contextRef);

int metal_layer_end(MetalDeviceRef contextRef, MetalStatus* status);

#ifdef __cplusplus
}
#endif

#endif
