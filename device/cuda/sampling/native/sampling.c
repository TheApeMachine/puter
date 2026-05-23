#include "sampling_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_sampling_module_source = NULL;

void cuda_sampling_register_module_source(const char* source) {
    g_cuda_sampling_module_source = source;
}

const char* cuda_sampling_module_source(void) {
    return g_cuda_sampling_module_source;
}

static int cuda_sampling_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA sampling kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA sampling kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_sampling_launch_greedy(
    CUDAContext* context,
    CUDAStreamRef stream,
    const char* moduleSource,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
) {
    char kernelName[128];
    int nameCode = cuda_sampling_kernel_name(
        kernelName,
        sizeof(kernelName),
        "greedy_sample",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    CUDAKernelRef kernel = cuda_get_kernel(context, moduleSource, kernelName, status);

    if (kernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* logitsPtr = cuda_buffer_device_ptr(logitsRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&logitsPtr, &outPtr, &count};
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

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

static int cuda_sampling_launch_init(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef logitsRef,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    uint32_t count,
    uint32_t paddedCount,
    CUDAStatus* status
) {
    void* logitsPtr = cuda_buffer_device_ptr(logitsRef);
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* indicesPtr = cuda_buffer_device_ptr(indicesRef);
    void* args[] = {&logitsPtr, &scoresPtr, &indicesPtr, &count, &paddedCount};
    int launchCode = cuda_launch_1d(context, kernel, stream, paddedCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    return 0;
}

static int cuda_sampling_launch_bitonic(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef kernel,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    uint32_t stageSize,
    uint32_t passSize,
    uint32_t paddedCount,
    CUDAStatus* status
) {
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* indicesPtr = cuda_buffer_device_ptr(indicesRef);
    void* args[] = {&scoresPtr, &indicesPtr, &stageSize, &passSize, &paddedCount};
    int launchCode = cuda_launch_1d(context, kernel, stream, paddedCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    return 0;
}

static int cuda_sampling_launch_sort(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef bitonicKernel,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    uint32_t paddedCount,
    CUDAStatus* status
) {
    for (uint32_t stageSize = 2u; stageSize <= paddedCount; stageSize <<= 1u) {
        for (uint32_t passSize = stageSize >> 1u; passSize > 0u; passSize >>= 1u) {
            int launchCode = cuda_sampling_launch_bitonic(
                context,
                stream,
                bitonicKernel,
                scoresRef,
                indicesRef,
                stageSize,
                passSize,
                paddedCount,
                status
            );

            if (launchCode != 0) {
                return launchCode;
            }
        }

        if (stageSize == paddedCount) {
            break;
        }
    }

    return 0;
}

static int cuda_sampling_launch_draw(
    CUDAContext* context,
    CUDAStreamRef stream,
    CUDAKernelRef drawKernel,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    CUDABufferRef outRef,
    uint32_t count,
    float target,
    uint64_t completionToken,
    CUDAStatus* status
) {
    void* scoresPtr = cuda_buffer_device_ptr(scoresRef);
    void* indicesPtr = cuda_buffer_device_ptr(indicesRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scoresPtr, &indicesPtr, &outPtr, &count, &target};
    int launchCode = cuda_launch_grid(
        context,
        drawKernel,
        stream,
        1,
        1,
        1,
        1,
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

static int cuda_sampling_launch_nucleus(
    CUDAContext* context,
    CUDAStreamRef stream,
    const char* moduleSource,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    CUDAStatus* status
) {
    char initName[128];
    int initNameCode = cuda_sampling_kernel_name(
        initName,
        sizeof(initName),
        "sampling_init",
        elementDType,
        status
    );

    if (initNameCode != 0) {
        return initNameCode;
    }

    CUDAKernelRef initKernel = cuda_get_kernel(context, moduleSource, initName, status);

    if (initKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    CUDAKernelRef bitonicKernel = cuda_get_kernel(context, moduleSource, "sampling_bitonic_step", status);

    if (bitonicKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    CUDAKernelRef drawKernel = cuda_get_kernel(context, moduleSource, "sampling_draw_sorted", status);

    if (drawKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    int initCode = cuda_sampling_launch_init(
        context,
        stream,
        initKernel,
        logitsRef,
        scoresRef,
        indicesRef,
        count,
        paddedCount,
        status
    );

    if (initCode != 0) {
        return initCode;
    }

    int sortCode = cuda_sampling_launch_sort(
        context,
        stream,
        bitonicKernel,
        scoresRef,
        indicesRef,
        paddedCount,
        status
    );

    if (sortCode != 0) {
        return sortCode;
    }

    return cuda_sampling_launch_draw(
        context,
        stream,
        drawKernel,
        scoresRef,
        indicesRef,
        outRef,
        count,
        target,
        completionToken,
        status
    );
}

int cuda_dispatch_sampling(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (logitsRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    if (operation != 0 && (scoresRef == NULL || indicesRef == NULL)) {
        cuda_status_set(status, -2, "nil CUDA sampling scratch buffer");
        return -2;
    }

    const char* moduleSource = cuda_sampling_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA sampling module source not registered");
        return -7;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    if (operation == 0) {
        return cuda_sampling_launch_greedy(
            context,
            stream,
            moduleSource,
            elementDType,
            logitsRef,
            outRef,
            count,
            completionToken,
            status
        );
    }

    return cuda_sampling_launch_nucleus(
        context,
        stream,
        moduleSource,
        elementDType,
        logitsRef,
        scoresRef,
        indicesRef,
        outRef,
        count,
        paddedCount,
        target,
        completionToken,
        status
    );
}
