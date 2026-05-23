#include "core_private.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CUDAResidentTensor {
    void* devicePtr;
    size_t byteCount;
    int dtype;
    int closed;
} CUDAResidentTensor;

static CUDAContext* cuda_global_context = NULL;

int cuda_open_device(CUDADeviceRef* outDevice, CUDAStatus* status) {
    cuda_status_clear(status);

    int deviceCount = 0;
    cudaError_t runtimeStatus = cudaGetDeviceCount(&deviceCount);

    if (runtimeStatus != cudaSuccess || deviceCount <= 0) {
        cuda_status_set(status, -1, "no CUDA device");
        return -1;
    }

    CUresult driverInit = cuInit(0);

    if (driverInit != CUDA_SUCCESS) {
        cuda_status_set(status, -1, "cuInit failed");
        return -1;
    }

    runtimeStatus = cudaSetDevice(0);

    if (runtimeStatus != cudaSuccess) {
        cuda_status_set(status, -1, "cudaSetDevice failed");
        return -1;
    }

    CUDAContext* context = (CUDAContext*)calloc(1, sizeof(CUDAContext));
    context->deviceIndex = 0;

    runtimeStatus = cudaStreamCreate((cudaStream_t*)&context->defaultStream);

    if (runtimeStatus != cudaSuccess) {
        free(context);
        cuda_status_set(status, -1, "cudaStreamCreate failed");
        return -1;
    }

    runtimeStatus = cudaStreamCreate((cudaStream_t*)&context->uploadStream);

    if (runtimeStatus != cudaSuccess) {
        cudaStreamDestroy((cudaStream_t)context->defaultStream);
        free(context);
        cuda_status_set(status, -1, "cudaStreamCreate upload failed");
        return -1;
    }

    cuda_global_context = context;
    *outDevice = (CUDADeviceRef)context;
    return 0;
}

void cuda_close_device(CUDADeviceRef device) {
    CUDAContext* context = (CUDAContext*)device;

    if (context == NULL) {
        return;
    }

    if (context->uploadStream != NULL) {
        cudaStreamDestroy((cudaStream_t)context->uploadStream);
    }

    if (context->defaultStream != NULL) {
        cudaStreamDestroy((cudaStream_t)context->defaultStream);
    }

    free(context);

    if (cuda_global_context == context) {
        cuda_global_context = NULL;
    }
}

long long cuda_device_total_memory(CUDADeviceRef device) {
    CUDAContext* context = (CUDAContext*)device;
    struct cudaDeviceProp properties;

    if (context == NULL) {
        return 0;
    }

    if (cudaGetDeviceProperties(&properties, context->deviceIndex) != cudaSuccess) {
        return 0;
    }

    return (long long)properties.totalGlobalMem;
}

int cuda_device_capability_major(CUDADeviceRef device) {
    CUDAContext* context = (CUDAContext*)device;
    struct cudaDeviceProp properties;

    if (context == NULL) {
        return 0;
    }

    if (cudaGetDeviceProperties(&properties, context->deviceIndex) != cudaSuccess) {
        return 0;
    }

    return properties.major;
}

CUDABufferRef cuda_buffer_alloc(CUDADeviceRef device, long long bytes) {
    CUDAContext* context = (CUDAContext*)device;
    void* devicePtr = NULL;

    if (context == NULL || bytes <= 0) {
        return NULL;
    }

    if (cudaMalloc(&devicePtr, (size_t)bytes) != cudaSuccess) {
        return NULL;
    }

    CUDAResidentTensor* resident = (CUDAResidentTensor*)calloc(1, sizeof(CUDAResidentTensor));
    resident->devicePtr = devicePtr;
    resident->byteCount = (size_t)bytes;
    return (CUDABufferRef)resident;
}

void cuda_buffer_release(CUDABufferRef buffer) {
    CUDAResidentTensor* resident = (CUDAResidentTensor*)buffer;

    if (resident == NULL) {
        return;
    }

    if (resident->devicePtr != NULL) {
        cudaFree(resident->devicePtr);
    }

    free(resident);
}

void* cuda_buffer_device_ptr(CUDABufferRef buffer) {
    CUDAResidentTensor* resident = (CUDAResidentTensor*)buffer;

    if (resident == NULL) {
        return NULL;
    }

    return resident->devicePtr;
}

int cuda_memcpy_async_h2d(
    CUDABufferRef dst,
    const void* src,
    long long bytes,
    CUDAStreamRef stream,
    CUDAStatus* status
) {
    CUDAResidentTensor* resident = (CUDAResidentTensor*)dst;

    if (resident == NULL || src == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    if (cudaMemcpyAsync(resident->devicePtr, src, (size_t)bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream) != cudaSuccess) {
        cuda_status_set(status, -7, "cudaMemcpyAsync H2D failed");
        return -7;
    }

    return 0;
}

int cuda_memcpy_async_d2h(
    void* dst,
    CUDABufferRef src,
    long long bytes,
    CUDAStreamRef stream,
    CUDAStatus* status
) {
    CUDAResidentTensor* resident = (CUDAResidentTensor*)src;

    if (resident == NULL || dst == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    if (cudaMemcpyAsync(dst, resident->devicePtr, (size_t)bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream) != cudaSuccess) {
        cuda_status_set(status, -7, "cudaMemcpyAsync D2H failed");
        return -7;
    }

    return 0;
}

int cuda_memcpy_async_d2d(
    CUDABufferRef dst,
    CUDABufferRef src,
    long long bytes,
    CUDAStreamRef stream,
    CUDAStatus* status
) {
    CUDAResidentTensor* dstResident = (CUDAResidentTensor*)dst;
    CUDAResidentTensor* srcResident = (CUDAResidentTensor*)src;

    if (dstResident == NULL || srcResident == NULL) {
        cuda_status_set(status, -2, "nil CUDA buffer");
        return -2;
    }

    if (cudaMemcpyAsync(
            dstResident->devicePtr,
            srcResident->devicePtr,
            (size_t)bytes,
            cudaMemcpyDeviceToDevice,
            (cudaStream_t)stream
        ) != cudaSuccess) {
        cuda_status_set(status, -7, "cudaMemcpyAsync D2D failed");
        return -7;
    }

    return 0;
}

int cuda_stream_synchronize(CUDAStreamRef stream, CUDAStatus* status) {
    if (cudaStreamSynchronize((cudaStream_t)stream) != cudaSuccess) {
        cuda_status_set(status, -7, "cudaStreamSynchronize failed");
        return -7;
    }

    return 0;
}

CUDADeviceRef cuda_default_context(void) {
    return (CUDADeviceRef)cuda_global_context;
}

CUDAStreamRef cuda_context_upload_stream(CUDADeviceRef device) {
    CUDAContext* context = (CUDAContext*)device;

    if (context == NULL) {
        return NULL;
    }

    return context->uploadStream;
}
