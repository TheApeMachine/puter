#include "conv_transpose2d.h"
#include "convolution.h"
#include "../internal/bridge/core_private.h"

int cuda_dispatch_conv_transpose2d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef weightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_convolution_status_clear(status);

    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        cuda_convolution_status_set(status, -2, "nil CUDA conv transpose2d buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = cuda_convolution_kernel_name(
        kernelName, sizeof(kernelName), "conv_transpose2d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    const char* moduleSource = cuda_convolution_module_source();

    if (moduleSource == NULL) {
        cuda_convolution_status_set(status, -7, "CUDA convolution module source not registered");
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

    void* inputPtr = cuda_buffer_device_ptr(inputRef);
    void* weightPtr = cuda_buffer_device_ptr(weightRef);
    void* biasPtr = cuda_buffer_device_ptr(biasRef);
    void* outPtr = cuda_buffer_device_ptr(outRef);
    void* args[] = {
        &inputPtr, &weightPtr, &biasPtr, &outPtr,
        &batch, &inChannels, &inHeight, &inWidth, &outChannels,
        &kernelHeight, &kernelWidth, &outHeight, &outWidth,
    };
    uint32_t launchCount = batch * outChannels * outHeight * outWidth;
    int launchCode = cuda_launch_1d(context, kernel, stream, launchCount, args, sizeof(args), status);

    if (launchCode != 0) {
        return launchCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
