#ifndef PUTER_DEVICE_METAL_ELEMENTWISE_ARITHMETIC_H
#define PUTER_DEVICE_METAL_ELEMENTWISE_ARITHMETIC_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MetalBinaryFloat32Op {
    MetalBinaryFloat32Add = 0,
    MetalBinaryFloat32Sub = 1,
    MetalBinaryFloat32Mul = 2,
    MetalBinaryFloat32Div = 3,
    MetalBinaryFloat32Max = 4,
    MetalBinaryFloat32Min = 5,
    MetalBinaryFloat32Eq = 6,
    MetalBinaryFloat32Ne = 7,
    MetalBinaryFloat32Lt = 8,
    MetalBinaryFloat32Le = 9,
    MetalBinaryFloat32Gt = 10,
    MetalBinaryFloat32Ge = 11,
    MetalBinaryFloat32Pow = 12,
    MetalBinaryFloat32Atan2 = 13,
    MetalBinaryFloat32Mod = 14,
} MetalBinaryFloat32Op;

int metal_dispatch_binary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
