#include "matrix.h"
#include "causal_dispatch.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_cholesky(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t matrixOrder,
    uint64_t completionToken,
    CUDAStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        cuda_causal_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {&inputPtr, &outPtr, &matrixOrder};

    return cuda_causal_named_launch(
        contextRef,
        elementDType,
        "cholesky",
        1,
        args,
        sizeof(args),
        completionToken,
        status
    );
}
