#include "core_private.h"

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
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

    if (context->deferredCompletions != NULL) {
        free(context->deferredCompletions);
        context->deferredCompletions = NULL;
    }

    free(context);
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

void metal_suspend_compute_encoder(MetalContext* context) {
    if (context == NULL || context->currentEncoder == NULL) {
        return;
    }

    id<MTLComputeCommandEncoder> encoder =
        (__bridge_transfer id<MTLComputeCommandEncoder>)context->currentEncoder;
    [encoder endEncoding];
    context->currentEncoder = NULL;
}

static void metal_command_completed(uint64_t completionToken, int code, const char* message) {
    (void)completionToken;
    (void)code;
    (void)message;
}

static void metal_notify_command_completed(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer,
    id<MTLBuffer> validationBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] != MTLCommandBufferStatusCompleted) {
            NSError* error = [completedBuffer error];
            NSString* message = @"Metal command buffer failed";

            if (error != nil) {
                message = [NSString
                    stringWithFormat:@"%@: %@",
                    message,
                    [error localizedDescription]
                ];
            }

            metal_command_completed(completionToken, -5, (char*)[message UTF8String]);
            return;
        }

        if (validationBuffer != nil) {
            uint32_t* validation = (uint32_t*)[validationBuffer contents];

            if (validation != NULL && validation[0] != 0) {
                metal_command_completed(
                    completionToken,
                    -8,
                    "Metal kernel reported invalid scalar data"
                );
                return;
            }
        }

        metal_command_completed(completionToken, 0, "");
    }
}

static void metal_deferred_push(
    MetalContext* context,
    uint64_t completionToken,
    void* validationBufferRef
) {
    if (context->deferredCount == context->deferredCapacity) {
        size_t nextCapacity = context->deferredCapacity == 0 ? 256 : context->deferredCapacity * 2;
        MetalDeferredCompletion* next = (MetalDeferredCompletion*)realloc(
            context->deferredCompletions,
            nextCapacity * sizeof(MetalDeferredCompletion)
        );

        if (next == NULL) {
            return;
        }

        context->deferredCompletions = next;
        context->deferredCapacity = nextCapacity;
    }

    context->deferredCompletions[context->deferredCount].token = completionToken;
    context->deferredCompletions[context->deferredCount].validationBuffer = validationBufferRef;
    context->deferredCount++;
}

void metal_track_command_completion(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    uint64_t completionToken,
    void* validationBufferRef
) {
    if (context == NULL || commandBuffer == nil || completionToken == 0) {
        return;
    }

    if (context->isBatching) {
        metal_deferred_push(context, completionToken, validationBufferRef);
        return;
    }

    id<MTLBuffer> validationBuffer = (__bridge id<MTLBuffer>)validationBufferRef;

    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
        metal_notify_command_completed(completionToken, completedBuffer, validationBuffer);
    }];
}

static void metal_flush_deferred_completions(
    MetalContext* context,
    id<MTLCommandBuffer> completedBuffer
) {
    if (context == NULL || context->deferredCount == 0) {
        return;
    }

    for (size_t index = 0; index < context->deferredCount; index++) {
        MetalDeferredCompletion* entry = &context->deferredCompletions[index];
        id<MTLBuffer> validationBuffer = (__bridge id<MTLBuffer>)entry->validationBuffer;

        metal_notify_command_completed(entry->token, completedBuffer, validationBuffer);
    }

    context->deferredCount = 0;
}

void metal_begin_batch(MetalDeviceRef contextRef) {
    MetalContext* context = (MetalContext*)contextRef;
    context->isBatching = true;
    context->deferredCount = 0;
    context->lastBatchStatus = 0;
}

void metal_end_batch(MetalDeviceRef contextRef, MetalStatus* status) {
    MetalContext* context = (MetalContext*)contextRef;
    id<MTLCommandBuffer> batchCommandBuffer = nil;

    metal_status_clear(status);
    context->isBatching = false;

    if (context->currentEncoder != NULL) {
        id<MTLComputeCommandEncoder> encoder =
            (__bridge_transfer id<MTLComputeCommandEncoder>)context->currentEncoder;
        [encoder endEncoding];
        context->currentEncoder = NULL;
    }

    if (context->currentCommandBuffer != NULL) {
        batchCommandBuffer =
            (__bridge_transfer id<MTLCommandBuffer>)context->currentCommandBuffer;
        context->currentCommandBuffer = NULL;
        [batchCommandBuffer commit];
    }

    if (batchCommandBuffer == nil) {
        if (context->deferredCount > 0) {
            metal_status_set(status, -5, "Metal batch ended without a command buffer");

            for (size_t index = 0; index < context->deferredCount; index++) {
                metal_command_completed(
                    context->deferredCompletions[index].token,
                    -5,
                    "Metal batch ended without a command buffer"
                );
            }

            context->deferredCount = 0;
        }

        return;
    }

    [batchCommandBuffer waitUntilCompleted];

    if ([batchCommandBuffer status] != MTLCommandBufferStatusCompleted) {
        NSError* error = [batchCommandBuffer error];
        NSString* message = @"Metal batch command buffer failed";

        if (error != nil) {
            message = [NSString
                stringWithFormat:@"%@: %@",
                message,
                [error localizedDescription]
            ];
        }

        context->lastBatchStatus = -5;
        metal_status_set(status, -5, [message UTF8String]);

        for (size_t index = 0; index < context->deferredCount; index++) {
            metal_command_completed(
                context->deferredCompletions[index].token,
                -5,
                (char*)[message UTF8String]
            );
        }

        context->deferredCount = 0;
        return;
    }

    metal_flush_deferred_completions(context, batchCommandBuffer);
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

void metal_device_wait_idle(MetalDeviceRef contextRef) {
    @autoreleasepool {
        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            return;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            return;
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
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

void metal_layer_begin(MetalDeviceRef contextRef) {
    metal_begin_batch(contextRef);
}

int metal_layer_end(MetalDeviceRef contextRef, MetalStatus* status) {
    metal_end_batch(contextRef, status);

    if (status != NULL && status->code != 0) {
        return status->code;
    }

    return 0;
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

void metal_device_release(MetalDeviceRef contextRef) {
    metal_release_context((MetalContext*)contextRef);
}
