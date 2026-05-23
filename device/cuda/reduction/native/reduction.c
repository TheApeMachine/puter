#include "reduction.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_reduction_module_source = NULL;

void cuda_reduction_register_module_source(const char* source) {
    g_cuda_reduction_module_source = source;
}

const char* cuda_reduction_module_source(void) {
    return g_cuda_reduction_module_source;
}

void cuda_reduction_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_reduction_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_reduction_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_reduction_status_set(status, -6, "unknown CUDA reduction dtype");
        return -6;
    }

    char prefix[128];
    int written = snprintf(prefix, sizeof(prefix), "reduction_%s", phase);

    if (written <= 0 || (size_t)written >= sizeof(prefix)) {
        cuda_reduction_status_set(status, -6, "CUDA reduction kernel name overflow");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, prefix, suffix, status);
}
