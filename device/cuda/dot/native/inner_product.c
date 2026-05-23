#include "inner_product.h"
#include "dot.h"
#include "../internal/bridge/core_private.h"

static int cuda_dot_dispatch_phase(
    CUDADeviceRef contextRef,
    const char* phase,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_dot_status_clear(status);

    if (count == 0) {
        return 0;
    }

    char kernelName[128];
    int nameCode = cuda_dot_kernel_name(
        kernelName,
        sizeof(kernelName),
        phase,
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_dot_module_source();

    if (moduleSource == NULL) {
        cuda_dot_status_set(status, -7, "CUDA dot module source not registered");
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

    if (phase[0] == 'd') {
        void* leftPtr = cuda_buffer_device_ptr(leftRef);
        void* rightPtr = cuda_buffer_device_ptr(rightRef);
        void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
        void* args[] = {&leftPtr, &rightPtr, &scratchPtr, &count};
        uint32_t gridSize = (count + 255u) / 256u;
        int launchCode = cuda_launch_1d(context, kernel, stream, gridSize, args, sizeof(args), status);

        if (launchCode != 0) {
            return launchCode;
        }
    }

    if (phase[0] == 'f') {
        void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
        void* outPtr = cuda_buffer_device_ptr(outRef);
        void* args[] = {&scratchPtr, &outPtr, &partialCount};
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

int cuda_dispatch_dot(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    int partialCode = cuda_dot_dispatch_phase(
        contextRef,
        "partial",
        elementDType,
        leftRef,
        rightRef,
        scratchRef,
        NULL,
        count,
        partialCount,
        0,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    return cuda_dot_dispatch_phase(
        contextRef,
        "finalize",
        elementDType,
        NULL,
        NULL,
        scratchRef,
        outRef,
        count,
        partialCount,
        completionToken,
        status
    );
}
