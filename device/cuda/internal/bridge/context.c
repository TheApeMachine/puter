#include "core_private.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CUDAModuleEntry {
    char* sourceHash;
    CUmodule module;
    struct CUDAModuleEntry* next;
} CUDAModuleEntry;

typedef struct CUDAKernelEntry {
    char* kernelName;
    CUfunction function;
    struct CUDAKernelEntry* next;
} CUDAKernelEntry;

typedef struct CUDAModuleCache {
    CUDAModuleEntry* modules;
    CUDAKernelEntry* kernels;
    pthread_mutex_t lock;
} CUDAModuleCache;

static CUDAModuleCache* cuda_module_cache(CUDAContext* context) {
    if (context->moduleCache == NULL) {
        CUDAModuleCache* cache = (CUDAModuleCache*)calloc(1, sizeof(CUDAModuleCache));
        pthread_mutex_init(&cache->lock, NULL);
        context->moduleCache = cache;
    }

    return (CUDAModuleCache*)context->moduleCache;
}

CUDAContext* cuda_context_from_ref(CUDADeviceRef contextRef) {
    return (CUDAContext*)contextRef;
}

int cuda_context_prepare(
    CUDADeviceRef contextRef,
    CUDAStatus* status,
    CUDAContext** context,
    CUDAStreamRef* stream
) {
    *context = cuda_context_from_ref(contextRef);

    if (*context == NULL) {
        cuda_status_set(status, -1, "invalid CUDA context");
        return -1;
    }

    *stream = (*context)->defaultStream;

    if (*stream == NULL) {
        cuda_status_set(status, -1, "invalid CUDA stream");
        return -1;
    }

    return 0;
}

CUDAStreamRef cuda_context_default_stream(CUDADeviceRef contextRef) {
    CUDAContext* context = cuda_context_from_ref(contextRef);

    if (context == NULL) {
        return NULL;
    }

    return context->defaultStream;
}

static int cuda_nvrtc_check(
    nvrtcResult result,
    const char* operation,
    CUDAStatus* status
) {
    if (result == NVRTC_SUCCESS) {
        return 0;
    }

    cuda_status_set(status, -7, operation);
    return -7;
}

static CUmodule cuda_compile_module(
    CUDAContext* context,
    const char* moduleSource,
    CUDAStatus* status
) {
    CUDAModuleCache* cache = cuda_module_cache(context);
    nvrtcProgram program = NULL;
    nvrtcResult nvrtcStatus = nvrtcCreateProgram(
        &program,
        moduleSource,
        "module.cu",
        0,
        NULL,
        NULL
    );

    if (cuda_nvrtc_check(nvrtcStatus, "nvrtcCreateProgram failed", status) != 0) {
        return NULL;
    }

    const char* options[] = {
        "--gpu-architecture=compute_70",
        "--std=c++17",
    };
    nvrtcStatus = nvrtcCompileProgram(program, 2, options);

    if (nvrtcStatus != NVRTC_SUCCESS) {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(program, &logSize);
        char* log = (char*)calloc(logSize + 1, 1);
        nvrtcGetProgramLog(program, log);
        cuda_status_set(status, -7, log);
        free(log);
        nvrtcDestroyProgram(&program);
        return NULL;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(program, &ptxSize);
    char* ptx = (char*)calloc(ptxSize + 1, 1);
    nvrtcGetPTX(program, ptx);
    nvrtcDestroyProgram(&program);

    CUmodule module = NULL;
    CUresult driverStatus = cuModuleLoadData(&module, ptx);
    free(ptx);

    if (driverStatus != CUDA_SUCCESS) {
        cuda_status_set(status, -7, "cuModuleLoadData failed");
        return NULL;
    }

    CUDAModuleEntry* entry = (CUDAModuleEntry*)calloc(1, sizeof(CUDAModuleEntry));
    entry->sourceHash = strdup(moduleSource);
    entry->module = module;
    entry->next = cache->modules;
    cache->modules = entry;

    return module;
}

CUDAKernelRef cuda_get_kernel(
    CUDAContext* context,
    const char* moduleSource,
    const char* kernelName,
    CUDAStatus* status
) {
    CUDAModuleCache* cache = cuda_module_cache(context);
    pthread_mutex_lock(&cache->lock);

    for (CUDAKernelEntry* entry = cache->kernels; entry != NULL; entry = entry->next) {
        if (strcmp(entry->kernelName, kernelName) == 0) {
            pthread_mutex_unlock(&cache->lock);
            return entry->function;
        }
    }

    CUmodule module = NULL;

    for (CUDAModuleEntry* entry = cache->modules; entry != NULL; entry = entry->next) {
        if (strcmp(entry->sourceHash, moduleSource) == 0) {
            module = entry->module;
            break;
        }
    }

    if (module == NULL) {
        module = cuda_compile_module(context, moduleSource, status);

        if (module == NULL) {
            pthread_mutex_unlock(&cache->lock);
            return NULL;
        }
    }

    CUfunction function = NULL;
    CUresult driverStatus = cuModuleGetFunction(&function, module, kernelName);

    if (driverStatus != CUDA_SUCCESS) {
        cuda_status_set(status, -7, "cuModuleGetFunction failed");
        pthread_mutex_unlock(&cache->lock);
        return NULL;
    }

    CUDAKernelEntry* kernelEntry = (CUDAKernelEntry*)calloc(1, sizeof(CUDAKernelEntry));
    kernelEntry->kernelName = strdup(kernelName);
    kernelEntry->function = function;
    kernelEntry->next = cache->kernels;
    cache->kernels = kernelEntry;

    pthread_mutex_unlock(&cache->lock);
    return function;
}

int cuda_launch_1d(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    uint32_t count,
    void** args,
    size_t argBytes,
    CUDAStatus* status
) {
    (void)context;
    (void)argBytes;

    if (kernel == NULL || count == 0) {
        return 0;
    }

    CUfunction function = (CUfunction)kernel;
    CUstream cuStream = (CUstream)stream;
    uint32_t blockSize = 256;
    uint32_t gridSize = (count + blockSize - 1) / blockSize;
    CUresult driverStatus = cuLaunchKernel(
        function,
        gridSize,
        1,
        1,
        blockSize,
        1,
        1,
        0,
        cuStream,
        args,
        NULL
    );

    if (driverStatus != CUDA_SUCCESS) {
        cuda_status_set(status, -7, "cuLaunchKernel failed");
        return -7;
    }

    return 0;
}

int cuda_launch_grid(
    CUDAContext* context,
    CUDAKernelRef kernel,
    CUDAStreamRef stream,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t blockX,
    uint32_t blockY,
    uint32_t blockZ,
    uint32_t sharedBytes,
    void** args,
    size_t argBytes,
    CUDAStatus* status
) {
    (void)context;
    (void)argBytes;

    if (kernel == NULL || gridX == 0 || blockX == 0) {
        return 0;
    }

    CUfunction function = (CUfunction)kernel;
    CUstream cuStream = (CUstream)stream;
    CUresult driverStatus = cuLaunchKernel(
        function,
        gridX,
        gridY,
        gridZ,
        blockX,
        blockY,
        blockZ,
        sharedBytes,
        cuStream,
        args,
        NULL
    );

    if (driverStatus != CUDA_SUCCESS) {
        cuda_status_set(status, -7, "cuLaunchKernel grid failed");
        return -7;
    }

    return 0;
}

void cuda_track_completion(
    CUDAContext* context,
    CUDAStreamRef stream,
    uint64_t completionToken,
    CUDAEventRef event,
    CUDAStatus* status
) {
    (void)context;
    (void)completionToken;

    if (event != NULL) {
        cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
        return;
    }

    cudaStreamSynchronize((cudaStream_t)stream);
    cuda_status_clear(status);
}
