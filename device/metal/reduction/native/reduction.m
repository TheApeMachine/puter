#include "reduction.h"

#include "../internal/bridge/core_private.h"

#include <stdio.h>

void metal_reduction_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_reduction_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_reduction_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_reduction_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_reduction_dtype_suffix(elementDType);

    if (phase == NULL || suffix == NULL) {
        metal_reduction_status_set(status, -6, "unknown Metal reduction kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "reduction_%s_%s", phase, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_reduction_status_set(status, -6, "Metal reduction kernel name overflow");
        return -6;
    }

    return 0;
}
