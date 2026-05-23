#include "spectral.h"
#include "physics.h"
#include "../internal/bridge/core_private.h"

#include <stdbool.h>
#include <stdio.h>

static bool cuda_physics_is_power_of_two(uint32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static uint32_t cuda_physics_log2(uint32_t value) {
    uint32_t bits = 0;

    while (value > 1) {
        value >>= 1;
        bits++;
    }

    return bits;
}

static int cuda_fft_encode_naive(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef realInRef,
    CUDABufferRef imagInRef,
    CUDABufferRef realOutRef,
    CUDABufferRef imagOutRef,
    CUDABufferRef twiddleRealRef,
    CUDABufferRef twiddleImagRef,
    uint32_t count,
    int inverse,
    CUDAStatus* status
) {
    char kernelName[128];
    int nameCode = cuda_physics_prefixed_kernel_name(kernelName, sizeof(kernelName), elementDType, "dft_naive", status);

    if (nameCode != 0) {
        return nameCode;
    }

    if (twiddleRealRef == NULL || twiddleImagRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA FFT twiddle buffer");
        return -2;
    }

    void* realInPtr = cuda_buffer_device_ptr(realInRef);
    void* imagInPtr = cuda_buffer_device_ptr(imagInRef);
    void* realOutPtr = cuda_buffer_device_ptr(realOutRef);
    void* imagOutPtr = cuda_buffer_device_ptr(imagOutRef);
    void* twiddleRealPtr = cuda_buffer_device_ptr(twiddleRealRef);
    void* twiddleImagPtr = cuda_buffer_device_ptr(twiddleImagRef);
    uint32_t inverseValue = inverse ? 1u : 0u;
    void* args[] = {
        &realInPtr,
        &imagInPtr,
        &realOutPtr,
        &imagOutPtr,
        &twiddleRealPtr,
        &twiddleImagPtr,
        &count,
        &inverseValue,
    };

    return cuda_physics_launch_kernel_no_track(
        contextRef,
        kernelName,
        count,
        args,
        sizeof(args),
        status
    );
}

static int cuda_fft_encode_power2(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef realInRef,
    CUDABufferRef imagInRef,
    CUDABufferRef realOutRef,
    CUDABufferRef imagOutRef,
    uint32_t count,
    int inverse,
    CUDAStatus* status
) {
    char bitKernelName[128];
    int bitNameCode = cuda_physics_prefixed_kernel_name(
        bitKernelName,
        sizeof(bitKernelName),
        elementDType,
        "fft_bit_reverse",
        status
    );

    if (bitNameCode != 0) {
        return bitNameCode;
    }

    void* realInPtr = cuda_buffer_device_ptr(realInRef);
    void* imagInPtr = cuda_buffer_device_ptr(imagInRef);
    void* realOutPtr = cuda_buffer_device_ptr(realOutRef);
    void* imagOutPtr = cuda_buffer_device_ptr(imagOutRef);
    uint32_t bits = cuda_physics_log2(count);
    void* bitArgs[] = {&realInPtr, &imagInPtr, &realOutPtr, &imagOutPtr, &count, &bits};
    int bitCode = cuda_physics_launch_kernel_no_track(
        contextRef,
        bitKernelName,
        count,
        bitArgs,
        sizeof(bitArgs),
        status
    );

    if (bitCode != 0) {
        return bitCode;
    }

    char stageKernelName[128];
    int stageNameCode = cuda_physics_prefixed_kernel_name(
        stageKernelName,
        sizeof(stageKernelName),
        elementDType,
        "fft_stage",
        status
    );

    if (stageNameCode != 0) {
        return stageNameCode;
    }

    uint32_t inverseValue = inverse ? 1u : 0u;

    for (uint32_t length = 2; length <= count; length <<= 1) {
        void* stageArgs[] = {&realOutPtr, &imagOutPtr, &length, &inverseValue};
        int stageCode = cuda_physics_launch_kernel_no_track(
            contextRef,
            stageKernelName,
            count / 2u,
            stageArgs,
            sizeof(stageArgs),
            status
        );

        if (stageCode != 0) {
            return stageCode;
        }
    }

    if (!inverse) {
        return 0;
    }

    char scaleKernelName[128];
    int scaleNameCode = cuda_physics_prefixed_kernel_name(
        scaleKernelName,
        sizeof(scaleKernelName),
        elementDType,
        "fft_scale",
        status
    );

    if (scaleNameCode != 0) {
        return scaleNameCode;
    }

    void* scaleArgs[] = {&realOutPtr, &imagOutPtr, &count};

    return cuda_physics_launch_kernel_no_track(
        contextRef,
        scaleKernelName,
        count,
        scaleArgs,
        sizeof(scaleArgs),
        status
    );
}

int cuda_dispatch_fft1d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef realInRef,
    CUDABufferRef imagInRef,
    CUDABufferRef realOutRef,
    CUDABufferRef imagOutRef,
    CUDABufferRef twiddleRealRef,
    CUDABufferRef twiddleImagRef,
    uint32_t count,
    int inverse,
    uint64_t completionToken,
    CUDAStatus* status
) {
    cuda_status_clear(status);

    if (count == 0) {
        return 0;
    }

    if (realInRef == NULL || imagInRef == NULL || realOutRef == NULL || imagOutRef == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    int encodeCode = 0;

    if (cuda_physics_is_power_of_two(count)) {
        encodeCode = cuda_fft_encode_power2(
            contextRef,
            elementDType,
            realInRef,
            imagInRef,
            realOutRef,
            imagOutRef,
            count,
            inverse,
            status
        );
    } else {
        encodeCode = cuda_fft_encode_naive(
            contextRef,
            elementDType,
            realInRef,
            imagInRef,
            realOutRef,
            imagOutRef,
            twiddleRealRef,
            twiddleImagRef,
            count,
            inverse,
            status
        );
    }

    if (encodeCode != 0) {
        return encodeCode;
    }

    CUDAContext* context = NULL;
    CUDAStreamRef stream = NULL;
    int prepareCode = cuda_context_prepare(contextRef, status, &context, &stream);

    if (prepareCode != 0) {
        return prepareCode;
    }

    cuda_track_completion(context, stream, completionToken, NULL, status);
    return 0;
}
