#include "intensity.h"
#include "hawkes_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_hawkes_intensity(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef eventsRef,
    CUDABufferRef queryTimesRef,
    CUDABufferRef baselineRef,
    CUDABufferRef alphaRef,
    CUDABufferRef betaRef,
    CUDABufferRef outRef,
    uint32_t eventCount,
    uint32_t queryCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (queryCount == 0) {
        return 0;
    }

    if (
        eventsRef == NULL ||
        queryTimesRef == NULL ||
        baselineRef == NULL ||
        alphaRef == NULL ||
        betaRef == NULL ||
        outRef == NULL
    ) {
        cuda_status_set(status, -2, "nil CUDA Hawkes intensity buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_hm_kernel_name(kernelName, sizeof(kernelName), "hawkes_intensity", elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    CUDAKernelRef kernel = NULL;
    int prepareCode = cuda_hm_get_kernel(contextRef, kernelName, status, &context, &stream, &kernel);

    if (prepareCode != 0) {
        return prepareCode;
    }

    void* eventsPtr = cuda_buffer_device_ptr(eventsRef);
    void* queryTimesPtr = cuda_buffer_device_ptr(queryTimesRef);
    void* baselinePtr = cuda_buffer_device_ptr(baselineRef);
    void* alphaPtr = cuda_buffer_device_ptr(alphaRef);
    void* betaPtr = cuda_buffer_device_ptr(betaRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {
        &eventsPtr,
        &queryTimesPtr,
        &baselinePtr,
        &alphaPtr,
        &betaPtr,
        &outPtr,
        &eventCount,
    };
    int launchCode = cuda_launch_grid(
        context,
        kernel,
        stream,
        queryCount,
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

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
