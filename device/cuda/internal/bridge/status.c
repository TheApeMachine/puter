#include "core.h"

#include <stdio.h>

void cuda_status_clear(CUDAStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void cuda_status_set(CUDAStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, CUDA_STATUS_MESSAGE_BYTES, "%s", message);
}

uint32_t cuda_vector_launch_count(uint32_t count, int elementDType) {
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

const char* cuda_element_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case CUDAElementDTypeFloat32:
        return "float32";
    case CUDAElementDTypeFloat16:
        return "float16";
    case CUDAElementDTypeBFloat16:
        return "bfloat16";
    case CUDAElementDTypeFloat64:
        return "float64";
    case CUDAElementDTypeFloat8E4M3:
        return "float8_e4m3";
    case CUDAElementDTypeFloat8E5M2:
        return "float8_e5m2";
    default:
        return NULL;
    }
}

int cuda_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    CUDAStatus* status
) {
    if (prefix == NULL || suffix == NULL) {
        cuda_status_set(status, -6, "unknown CUDA kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        cuda_status_set(status, -6, "CUDA kernel name overflow");
        return -6;
    }

    return 0;
}
