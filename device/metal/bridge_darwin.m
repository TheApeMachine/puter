#include "bridge_darwin_private.h"

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include "_cgo_export.h"
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void metal_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_status_set(MetalStatus* status, int code, const char* message) {
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

static void metal_status_set_ns_error(
    MetalStatus* status,
    int code,
    NSString* operation,
    NSError* error
) {
    NSString* message = operation;

    if (error != nil) {
        message = [NSString
            stringWithFormat:@"%@: %@",
            operation,
            [error localizedDescription]
        ];
    }

    metal_status_set(status, code, [message UTF8String]);
}

static void metal_release_context(MetalContext* context) {
    if (context == NULL) {
        return;
    }

    if (context->pipelineLock != NULL) {
        CFRelease(context->pipelineLock);
        context->pipelineLock = NULL;
    }

    if (context->pipelineCache != NULL) {
        CFRelease(context->pipelineCache);
        context->pipelineCache = NULL;
    }

    if (context->library != NULL) {
        CFRelease(context->library);
        context->library = NULL;
    }

    if (context->queue != NULL) {
        CFRelease(context->queue);
        context->queue = NULL;
    }

    if (context->device != NULL) {
        CFRelease(context->device);
        context->device = NULL;
    }

    free(context);
}

static const char* metal_binary_float32_kernel_name(int operation) {
    switch (operation) {
    case MetalBinaryFloat32Add: return "add_float32";
    case MetalBinaryFloat32Sub: return "sub_float32";
    case MetalBinaryFloat32Mul: return "mul_float32";
    case MetalBinaryFloat32Div: return "div_float32";
    case MetalBinaryFloat32Max: return "max_float32";
    case MetalBinaryFloat32Min: return "min_float32";
    case MetalBinaryFloat32Eq: return "eq_float32";
    case MetalBinaryFloat32Ne: return "ne_float32";
    case MetalBinaryFloat32Lt: return "lt_float32";
    case MetalBinaryFloat32Le: return "le_float32";
    case MetalBinaryFloat32Gt: return "gt_float32";
    case MetalBinaryFloat32Ge: return "ge_float32";
    case MetalBinaryFloat32Pow: return "pow_float32";
    case MetalBinaryFloat32Atan2: return "atan2_float32";
    case MetalBinaryFloat32Mod: return "mod_float32";
    default:
        return NULL;
    }
}


id<MTLComputeCommandEncoder> metal_get_encoder(MetalContext* context, id<MTLCommandBuffer>* outCommandBuffer) {
    if (context->isBatching) {
        if (context->currentCommandBuffer == NULL) {
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            context->currentCommandBuffer = (__bridge_retained void*)cb;
            context->currentEncoder = (__bridge_retained void*)[cb computeCommandEncoder];
        }
        *outCommandBuffer = (__bridge id<MTLCommandBuffer>)context->currentCommandBuffer;
        return (__bridge id<MTLComputeCommandEncoder>)context->currentEncoder;
    } else {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        *outCommandBuffer = cb;
        return [cb computeCommandEncoder];
    }
}

void metal_end_encoder(MetalContext* context, id<MTLComputeCommandEncoder> encoder, id<MTLCommandBuffer> commandBuffer) {
    if (!context->isBatching) {
        [encoder endEncoding];
        [commandBuffer commit];
    }
}

void metal_begin_batch(MetalDeviceRef contextRef) {
    MetalContext* context = (MetalContext*)contextRef;
    context->isBatching = true;
}

void metal_end_batch(MetalDeviceRef contextRef) {
    MetalContext* context = (MetalContext*)contextRef;
    context->isBatching = false;
    if (context->currentEncoder != NULL) {
        id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)context->currentEncoder;
        [enc endEncoding];
        context->currentEncoder = NULL;
    }
    if (context->currentCommandBuffer != NULL) {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)context->currentCommandBuffer;
        [cb commit];
        context->currentCommandBuffer = NULL;
    }
}

id<MTLComputePipelineState> metal_get_pipeline(
    MetalContext* context,
    const char* name,
    MetalStatus* status
) {
    if (context == NULL ||
        context->device == NULL ||
        context->library == NULL ||
        context->pipelineCache == NULL ||
        context->pipelineLock == NULL) {
        metal_status_set(status, -6, "invalid Metal pipeline context");
        return nil;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)context->library;
    NSCache* pipelineCache =
        (__bridge NSCache*)context->pipelineCache;
    NSLock* pipelineLock = (__bridge NSLock*)context->pipelineLock;
    NSString* functionName = [NSString stringWithUTF8String:name];

    id<MTLComputePipelineState> cachedPipeline = [pipelineCache objectForKey:functionName];

    if (cachedPipeline != nil) {
        return cachedPipeline;
    }

    id<MTLFunction> function = [library newFunctionWithName:functionName];

    if (function == nil) {
        metal_status_set(status, -6, "newFunctionWithName returned nil");
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];

    if (pipeline == nil) {
        metal_status_set_ns_error(status, -7, @"newComputePipelineStateWithFunction", error);
        return nil;
    }

    [pipelineCache setObject:pipeline forKey:functionName];

    return pipeline;
}

MetalDeviceRef metal_open_default_device(
    const uint8_t* libraryBytes,
    long long libraryLength,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_status_clear(status);

        if (libraryBytes == NULL || libraryLength <= 0) {
            metal_status_set(status, -1, "empty Metal library");
            return NULL;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device == nil) {
            metal_status_set(status, -2, "MTLCreateSystemDefaultDevice returned nil");
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (queue == nil) {
            metal_status_set(status, -3, "newCommandQueue returned nil");
            return NULL;
        }

        dispatch_data_t libraryData = dispatch_data_create(
            libraryBytes,
            (size_t)libraryLength,
            nil,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT
        );

        if (libraryData == nil) {
            metal_status_set(status, -4, "dispatch_data_create returned nil");
            return NULL;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:libraryData error:&error];

        if (library == nil) {
            metal_status_set_ns_error(status, -5, @"newLibraryWithData", error);
            return NULL;
        }

        MetalContext* context = (MetalContext*)calloc(1, sizeof(MetalContext));

        if (context == NULL) {
            metal_status_set(status, -8, "calloc MetalContext failed");
            return NULL;
        }

        context->device = (__bridge_retained void*)device;
        context->queue = (__bridge_retained void*)queue;
        context->library = (__bridge_retained void*)library;
        context->pipelineCache = (__bridge_retained void*)[[NSCache alloc] init];
        context->pipelineLock = (__bridge_retained void*)[[NSLock alloc] init];

        if (context->pipelineCache == NULL || context->pipelineLock == NULL) {
            metal_status_set(status, -9, "Metal pipeline cache initialization failed");
            metal_release_context(context);
            return NULL;
        }

        return context;
    }
}

long long metal_recommended_max_working_set(MetalDeviceRef contextRef) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->device == NULL) {
        return 0;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    return (long long)[device recommendedMaxWorkingSetSize];
}

MetalBufferRef metal_buffer_new_shared(MetalDeviceRef contextRef, long long bytes) {
    @autoreleasepool {
        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->device == NULL || bytes <= 0) {
            return NULL;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        id<MTLBuffer> buffer = [device
            newBufferWithLength:(NSUInteger)bytes
            options:MTLResourceStorageModeShared
        ];

        if (buffer == nil) {
            return NULL;
        }

        return (__bridge_retained void*)buffer;
    }
}

void metal_buffer_release(MetalBufferRef bufferRef) {
    if (bufferRef != NULL) {
        CFRelease(bufferRef);
    }
}

void* metal_buffer_contents(MetalBufferRef bufferRef) {
    if (bufferRef == NULL) {
        return NULL;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferRef;
    return [buffer contents];
}

int metal_dispatch_binary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    return metal_dispatch_binary_elementwise(
        contextRef,
        operation,
        MetalElementDTypeFloat32,
        leftRef,
        rightRef,
        outRef,
        count,
        completionToken,
        status
    );
}

void metal_device_release(MetalDeviceRef contextRef) {
    metal_release_context((MetalContext*)contextRef);
}
