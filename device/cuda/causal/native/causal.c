#include "causal_dispatch.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>
#include <string.h>

static const char* g_cuda_causal_module_source = NULL;

void cuda_causal_register_module_source(const char* source) {
    g_cuda_causal_module_source = source;
}

const char* cuda_causal_module_source(void) {
    return g_cuda_causal_module_source;
}

void cuda_causal_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_causal_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        cuda_causal_status_set(status, -6, "unknown CUDA causal kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_causal_status_set(status, -6, "CUDA causal kernel name overflow");
        return -6;
    }

    return 0;
}

static int cuda_causal_two_phase_names(
    char* partialName,
    char* finalizeName,
    size_t nameBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    char baseName[128];
    int nameCode = cuda_causal_kernel_name(
        baseName, sizeof(baseName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    int partialWritten = snprintf(partialName, nameBytes, "%s_partial", baseName);
    int finalizeWritten = snprintf(finalizeName, nameBytes, "%s_finalize", baseName);

    if (partialWritten <= 0 || finalizeWritten <= 0 ||
        (size_t)partialWritten >= nameBytes || (size_t)finalizeWritten >= nameBytes) {
        cuda_causal_status_set(status, -6, "CUDA causal two-phase kernel name overflow");
        return -6;
    }

    return 0;
}

int cuda_causal_named_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    const char* operationName,
    uint32_t launchCount,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (launchCount == 0) {
        return 0;
    }

    const char* moduleSource = cuda_causal_module_source();

    if (moduleSource == NULL) {
        cuda_causal_status_set(status, -7, "CUDA causal module source not registered");
        return -7;
    }

    char kernelName[128];
    int nameCode = cuda_causal_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
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

    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, argsBytes, status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

static int cuda_causal_launch_grid(
    CUDADeviceRef contextRef,
    const char* kernelName,
    uint32_t gridX,
    uint32_t blockX,
    void** args,
    size_t argsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    const char* moduleSource = cuda_causal_module_source();

    if (moduleSource == NULL) {
        cuda_causal_status_set(status, -7, "CUDA causal module source not registered");
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

    int launchCode = cuda_launch_grid(
        context,
        kernel,
        stream,
        gridX,
        1,
        1,
        blockX,
        1,
        1,
        0,
        args,
        argsBytes,
        status
    );

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}

int cuda_causal_two_phase_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    const char* operationName,
    uint32_t partialGridX,
    void** partialArgs,
    size_t partialArgsBytes,
    void** finalizeArgs,
    size_t finalizeArgsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (partialGridX == 0) {
        return 0;
    }

    char partialName[128];
    char finalizeName[128];
    int nameCode = cuda_causal_two_phase_names(
        partialName, finalizeName, sizeof(partialName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    int partialCode = cuda_causal_launch_grid(
        contextRef,
        partialName,
        partialGridX,
        256,
        partialArgs,
        partialArgsBytes,
        0,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    return cuda_causal_launch_grid(
        contextRef,
        finalizeName,
        1,
        256,
        finalizeArgs,
        finalizeArgsBytes,
        completionToken,
        status
    );
}

int cuda_causal_dag_two_phase_launch(
    CUDADeviceRef contextRef,
    int elementDType,
    uint32_t partialGridX,
    void** partialArgs,
    size_t partialArgsBytes,
    void** finalizeArgs,
    size_t finalizeArgsBytes,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (partialGridX == 0) {
        return 0;
    }

    char partialName[128];
    char finalizeName[128];
    int partialNameCode = cuda_causal_kernel_name(
        partialName, sizeof(partialName), "dag_markov_factorization", elementDType, status
    );

    if (partialNameCode != 0) {
        return partialNameCode;
    }

    int partialSuffixWritten = snprintf(
        partialName + strlen(partialName),
        sizeof(partialName) - strlen(partialName),
        "_partial"
    );

    int finalizeNameCode = cuda_causal_kernel_name(
        finalizeName, sizeof(finalizeName), "dag_markov_factorization", elementDType, status
    );

    if (finalizeNameCode != 0) {
        return finalizeNameCode;
    }

    int finalizeSuffixWritten = snprintf(
        finalizeName + strlen(finalizeName),
        sizeof(finalizeName) - strlen(finalizeName),
        "_finalize"
    );

    if (partialSuffixWritten <= 0 || finalizeSuffixWritten <= 0) {
        cuda_causal_status_set(status, -6, "CUDA causal DAG kernel name overflow");
        return -6;
    }

    int partialCode = cuda_causal_launch_grid(
        contextRef,
        partialName,
        partialGridX,
        256,
        partialArgs,
        partialArgsBytes,
        0,
        status
    );

    if (partialCode != 0) {
        return partialCode;
    }

    return cuda_causal_launch_grid(
        contextRef,
        finalizeName,
        1,
        256,
        finalizeArgs,
        finalizeArgsBytes,
        completionToken,
        status
    );
}
