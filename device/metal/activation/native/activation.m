#include "activation.h"

#include "../internal/bridge/core_private.h"

#include <stdio.h>

void metal_activation_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_activation_status_set(MetalStatus* status, int code, const char* message) {
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

const char* metal_activation_element_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32:
        return "float32";
    case MetalElementDTypeFloat16:
        return "float16";
    case MetalElementDTypeBFloat16:
        return "bfloat16";
    case MetalElementDTypeFloat64:
        return "float64";
    default:
        return NULL;
    }
}

int metal_activation_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    MetalStatus* status
) {
    if (prefix == NULL || suffix == NULL) {
        metal_activation_status_set(status, -6, "unknown Metal activation kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_activation_status_set(status, -6, "Metal activation kernel name overflow");
        return -6;
    }

    return 0;
}

uint32_t metal_activation_vector_launch_count(uint32_t count, int elementDType) {
    if (count == 0) {
        return 0;
    }

    switch (elementDType) {
    case MetalElementDTypeFloat32:
    case MetalElementDTypeFloat16:
    case MetalElementDTypeBFloat16:
        return (count + 3u) / 4u;
    default:
        return count;
    }
}
