#include "losses_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_losses_module_source = NULL;

void cuda_losses_register_module_source(const char* source) {
    g_cuda_losses_module_source = source;
}

const char* cuda_losses_module_source(void) {
    return g_cuda_losses_module_source;
}

static const char* cuda_pair_loss_operation_name(int operation) {
    switch (operation) {
    case 0: return "mse_loss";
    case 1: return "mae_loss";
    case 2: return "huber_loss";
    case 3: return "binary_cross_entropy";
    case 4: return "kl_divergence";
    default: return NULL;
    }
}

static int cuda_loss_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA loss kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA loss kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_loss_finalize_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA loss finalize kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "loss_finalize_%s", suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA loss finalize kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_loss_launch_finalize(
    CUDAContext* context,
    CUDAStreamRef stream,
    const char* moduleSource,
    int elementDType,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t partialCount,
    uint32_t denominator,
    uint64_t completionToken,
    CUDAStatus* status
) {
    char kernelName[128];
    int nameCode = cuda_loss_finalize_kernel_name(
        kernelName,
        sizeof(kernelName),
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

    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&scratchPtr, &outPtr, &partialCount, &denominator};
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

int cuda_dispatch_pair_loss(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef predictionsRef,
    CUDABufferRef targetsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0 || partialCount == 0) {
        return 0;
    }

    if (predictionsRef == NULL || targetsRef == NULL || scratchRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* operationName = cuda_pair_loss_operation_name(operation);

    if (operationName == NULL) {
        cuda_status_set(status, -6, "unknown CUDA pair loss operation");
        return -6;
    }

    const char* moduleSource = cuda_losses_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA losses module source not registered");
        return -7;
    }

    char partialName[128];
    int partialNameCode = cuda_loss_kernel_name(
        partialName,
        sizeof(partialName),
        operationName,
        "partial",
        elementDType,
        status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef partialKernel = cuda_get_kernel(context, moduleSource, partialName, status);

    if (partialKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* predictionsPtr = cuda_buffer_device_ptr(predictionsRef);
    void* targetsPtr = cuda_buffer_device_ptr(targetsRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* partialArgs[] = {&predictionsPtr, &targetsPtr, &scratchPtr, &count};
    int partialLaunchCode = cuda_launch_grid(
        context,
        partialKernel,
        stream,
        partialCount,
        1,
        1,
        256,
        1,
        1,
        0,
        partialArgs,
        sizeof(partialArgs),
        status
    );

    if (partialLaunchCode != 0) {
        return partialLaunchCode;
    }

    return cuda_loss_launch_finalize(
        context,
        stream,
        moduleSource,
        elementDType,
        scratchRef,
        outRef,
        partialCount,
        count,
        completionToken,
        status
    );
}

int cuda_dispatch_cross_entropy_loss(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef targetsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    CUDABufferRef errorFlagRef,
    uint32_t batch,
    uint32_t classes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (batch == 0 || classes == 0) {
        return 0;
    }

    if (logitsRef == NULL || targetsRef == NULL || scratchRef == NULL || outRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    const char* moduleSource = cuda_losses_module_source();

    if (moduleSource == NULL) {
        cuda_status_set(status, -7, "CUDA losses module source not registered");
        return -7;
    }

    char partialName[128];
    int partialNameCode = cuda_loss_kernel_name(
        partialName,
        sizeof(partialName),
        "cross_entropy",
        "partial",
        elementDType,
        status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    CUDAKernelRef partialKernel = cuda_get_kernel(context, moduleSource, partialName, status);

    if (partialKernel == NULL) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    void* logitsPtr = cuda_buffer_device_ptr(logitsRef);
    void* targetsPtr = cuda_buffer_device_ptr(targetsRef);
    void* scratchPtr = cuda_buffer_device_ptr(scratchRef);
    void* errorFlagPtr = cuda_buffer_device_ptr(errorFlagRef);
    void* partialArgs[] = {&logitsPtr, &targetsPtr, &scratchPtr, &errorFlagPtr, &batch, &classes};
    int partialLaunchCode = cuda_launch_grid(
        context,
        partialKernel,
        stream,
        batch,
        1,
        1,
        256,
        1,
        1,
        0,
        partialArgs,
        sizeof(partialArgs),
        status
    );

    if (partialLaunchCode != 0) {
        return partialLaunchCode;
    }

    return cuda_loss_launch_finalize(
        context,
        stream,
        moduleSource,
        elementDType,
        scratchRef,
        outRef,
        batch,
        batch,
        completionToken,
        status
    );
}
