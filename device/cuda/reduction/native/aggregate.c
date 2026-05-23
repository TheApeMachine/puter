#include "aggregate.h"
#include "reduction.h"
#include "../internal/bridge/core_private.h"

static int cuda_reduction_dispatch_phase(
    CUDADeviceRef contextRef,
    const char* phase,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scratchARef,
    CUDABufferRef scratchBRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint32_t operation,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_reduction_status_clear(status);

    if (count == 0) {
        return 0;
    }

    char kernelName[128];
    int nameCode = cuda_reduction_kernel_name(
        kernelName,
        sizeof(kernelName),
        phase,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_reduction_module_source();

    if (moduleSource == NULL) {
        cuda_reduction_status_set(status, -7, "CUDA reduction module source not registered");
        return -7;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    if (phase[0] == 'p') {
        if (inputRef == NULL || scratchARef == NULL || scratchBRef == NULL) {
            cuda_reduction_status_set(status, -2, "nil CUDA reduction partial buffer");
            return -2;
        }

        void* inputPtr = cuda_buffer_device_ptr(inputRef);
        void* scratchAPtr = cuda_buffer_device_ptr(scratchARef);
        void* scratchBPtr = cuda_buffer_device_ptr(scratchBRef);
        void* args[] = {&inputPtr, &scratchAPtr, &scratchBPtr, &count, &operation};
        uint32_t gridSize = (count + 255u) / 256u;
        int launchCode = cuda_launch_1d(context, kernel, stream, gridSize, args, sizeof(args), status);

        if (launchCode != 0) {
            return launchCode;
        }
    }

    if (phase[0] == 'f') {
        if (scratchARef == NULL || scratchBRef == NULL || outRef == NULL) {
            cuda_reduction_status_set(status, -2, "nil CUDA reduction finalize buffer");
            return -2;
        }

        void* scratchAPtr = cuda_buffer_device_ptr(scratchARef);
        void* scratchBPtr = cuda_buffer_device_ptr(scratchBRef);
        void* outPtr = cuda_buffer_device_ptr(outRef);
        void* args[] = {&scratchAPtr, &scratchBPtr, &outPtr, &partialCount, &count, &operation};
        int launchCode = cuda_launch_grid(
            context,
            kernel,
            stream,
            1,
            1,
            1,
            256,
            1,
            1,
            0,
            args,
            sizeof(args),
            status
        );

        if (launchCode != 0) {
            return launchCode;
        }
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_dispatch_reduction(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scratchARef,
    CUDABufferRef scratchBRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    uint32_t operationCode = (uint32_t)operation;
    int partialCode = cuda_reduction_dispatch_phase(
        contextRef,
        "partial",
        elementDType,
        inputRef,
        scratchARef,
        scratchBRef,
        NULL,
        count,
        partialCount,
        operationCode,
        0,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    return cuda_reduction_dispatch_phase(
        contextRef,
        "finalize",
        elementDType,
        NULL,
        scratchARef,
        scratchBRef,
        outRef,
        count,
        partialCount,
        operationCode,
        completionToken,
        status
    );
}
