#include "differential.h"
#include "physics.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static int cuda_physics_dispatch_vector(
    CUDADeviceRef contextRef,
    const char* operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (inputRef == NULL || spacingRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_physics_kernel_name(kernelName, sizeof(kernelName), operation, elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* spacingPtr = cuda_buffer_device_ptr(spacingRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &spacingPtr, &outPtr, &count};

    return cuda_physics_launch_kernel(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

int cuda_dispatch_laplacian(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t rank,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (inputRef == NULL || spacingRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_physics_kernel_name(kernelName, sizeof(kernelName), "laplacian", elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* spacingPtr = cuda_buffer_device_ptr(spacingRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &spacingPtr, &outPtr, &count, &rank, &dim0, &dim1, &dim2};

    return cuda_physics_launch_kernel(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}

#define PHYSICS_VECTOR_DISPATCH(name, operation) \
int name( \
    CUDADeviceRef contextRef, \
    int elementDType, \
    CUDABufferRef inputRef, \
    CUDABufferRef spacingRef, \
    CUDABufferRef outRef, \
    uint32_t count, \
    uint64_t completionToken, \
    CUDAStatus* status \
) { \
    return cuda_physics_dispatch_vector( \
        contextRef, \
        operation, \
        elementDType, \
        inputRef, \
        spacingRef, \
        outRef, \
        count, \
        completionToken, \
        status \
    ); \
}

PHYSICS_VECTOR_DISPATCH(cuda_dispatch_laplacian4, "laplacian4")
PHYSICS_VECTOR_DISPATCH(cuda_dispatch_grad1d, "grad1d")
PHYSICS_VECTOR_DISPATCH(cuda_dispatch_divergence1d, "divergence1d")
PHYSICS_VECTOR_DISPATCH(cuda_dispatch_quantum_potential, "quantum_potential")
PHYSICS_VECTOR_DISPATCH(cuda_dispatch_bohmian_velocity, "bohmian_velocity")

int cuda_dispatch_madelung_continuity(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef densityRef,
    CUDABufferRef velocityRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (densityRef == NULL || velocityRef == NULL || spacingRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_physics_kernel_name(
        kernelName,
        sizeof(kernelName),
        "madelung_continuity",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    void* densityPtr = cuda_buffer_device_ptr(densityRef);
    void* velocityPtr = cuda_buffer_device_ptr(velocityRef);
    void* spacingPtr = cuda_buffer_device_ptr(spacingRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&densityPtr, &velocityPtr, &spacingPtr, &outPtr, &count};

    return cuda_physics_launch_kernel(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
