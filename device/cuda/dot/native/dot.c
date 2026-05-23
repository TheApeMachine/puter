#include "dot.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_dot_module_source = NULL;

void cuda_dot_register_module_source(const char* source) {
    g_cuda_dot_module_source = source;
}

const char* cuda_dot_module_source(void) {
    return g_cuda_dot_module_source;
}

void cuda_dot_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_dot_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_dot_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_dot_status_set(status, -6, "unknown CUDA dot dtype");
        return -6;
    }

    int written = snprintf(out, outBytes, "dot_%s_%s", suffix, phase);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_dot_status_set(status, -6, "CUDA dot kernel name overflow");
        return -6;
    }

    return 0;
}
