#include "activation.h"
#include "../internal/bridge/core_private.h"

static const char* activation_module_source = NULL;

void cuda_activation_register_module_source(const char* source) {
    activation_module_source = source;
}

const char* cuda_activation_module_source(void) {
    return activation_module_source;
}

uint32_t cuda_activation_vector_launch_count(uint32_t count, int elementDType) {
    if (count == 0) {
        return 0;
    }

    switch (elementDType) {
    case CUDAElementDTypeFloat32:
        return (count + 3u) / 4u;
    case CUDAElementDTypeFloat16:
    case CUDAElementDTypeBFloat16:
        return (count + 1u) / 2u;
    default:
        return count;
    }
}

void cuda_activation_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_activation_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

const char* cuda_activation_element_dtype_suffix(int elementDType) {
    return cuda_element_dtype_suffix(elementDType);
}

int cuda_activation_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    CUDAStatus* status
) {
    return cuda_compose_kernel_name(out, outBytes, prefix, suffix, status);
}
