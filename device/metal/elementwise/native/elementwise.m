#include "elementwise.h"

#include "../internal/bridge/core_private.h"

#include <stdio.h>

void metal_elementwise_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_elementwise_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

const char* metal_elementwise_element_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    case MetalElementDTypeFloat64: return "float64";
    default: return NULL;
    }
}

int metal_elementwise_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* dtypeSuffix,
    MetalStatus* status
) {
    if (operationName == NULL || dtypeSuffix == NULL) {
        metal_elementwise_status_set(status, -6, "unknown Metal elementwise kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, dtypeSuffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_elementwise_status_set(status, -6, "Metal elementwise kernel name overflow");
        return -6;
    }

    return 0;
}
