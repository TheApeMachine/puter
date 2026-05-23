#include "elementwise.h"
#include "../internal/bridge/core_private.h"

static const char* elementwise_module_source = NULL;

void cuda_elementwise_register_module_source(const char* source) {
    elementwise_module_source = source;
}

const char* cuda_elementwise_module_source(void) {
    return elementwise_module_source;
}

void cuda_elementwise_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_elementwise_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

const char* cuda_elementwise_element_dtype_suffix(int elementDType) {
    return cuda_element_dtype_suffix(elementDType);
}

int cuda_elementwise_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* dtypeSuffix,
    CUDAStatus* status
) {
    return cuda_compose_kernel_name(out, outBytes, operationName, dtypeSuffix, status);
}
