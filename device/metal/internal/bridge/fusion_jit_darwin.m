#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "fusion_jit.h"
#include "core_private.h"
#include <stdlib.h>

typedef struct MetalFusionProgram {
    void* pipeline;
    void* library;
} MetalFusionProgram;

static void metal_fusion_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    strncpy(status->message, message, METAL_STATUS_MESSAGE_BYTES - 1);
    status->message[METAL_STATUS_MESSAGE_BYTES - 1] = '\0';
}

static void metal_fusion_status_set_ns_error(
    MetalStatus* status,
    int code,
    NSString* operation,
    NSError* error
) {
    if (status == NULL) {
        return;
    }

    if (error == nil) {
        metal_fusion_status_set(status, code, [operation UTF8String]);
        return;
    }

    NSString* message = [NSString stringWithFormat:@"%@: %@", operation, error.localizedDescription];
    metal_fusion_status_set(status, code, message.UTF8String);
}

MetalFusionProgramRef metal_fusion_program_compile(
    MetalDeviceRef contextRef,
    const char* source,
    const char* kernelName,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_fusion_status_set(status, 0, NULL);

        if (contextRef == NULL || source == NULL || kernelName == NULL) {
            metal_fusion_status_set(status, -1, "fusion program: invalid compile arguments");
            return NULL;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context->device == NULL) {
            metal_fusion_status_set(status, -2, "fusion program: missing Metal device");
            return NULL;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        NSString* sourceText = [NSString stringWithUTF8String:source];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:sourceText options:nil error:&error];

        if (library == nil) {
            metal_fusion_status_set_ns_error(status, -3, @"newLibraryWithSource", error);
            return NULL;
        }

        NSString* functionName = [NSString stringWithUTF8String:kernelName];
        id<MTLFunction> function = [library newFunctionWithName:functionName];

        if (function == nil) {
            metal_fusion_status_set(status, -4, "newFunctionWithName returned nil");
            return NULL;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];

        if (pipeline == nil) {
            metal_fusion_status_set_ns_error(status, -5, @"newComputePipelineStateWithFunction", error);
            return NULL;
        }

        MetalFusionProgram* program = (MetalFusionProgram*)calloc(1, sizeof(MetalFusionProgram));

        if (program == NULL) {
            metal_fusion_status_set(status, -6, "calloc MetalFusionProgram failed");
            return NULL;
        }

        program->pipeline = (__bridge_retained void*)pipeline;
        program->library = (__bridge_retained void*)library;

        return program;
    }
}

void metal_fusion_program_release(MetalFusionProgramRef programRef) {
    @autoreleasepool {
        MetalFusionProgram* program = (MetalFusionProgram*)programRef;

        if (program == NULL) {
            return;
        }

        if (program->pipeline != NULL) {
            CFRelease(program->pipeline);
            program->pipeline = NULL;
        }

        if (program->library != NULL) {
            CFRelease(program->library);
            program->library = NULL;
        }

        free(program);
    }
}

int metal_fusion_program_dispatch(
    MetalDeviceRef contextRef,
    MetalFusionProgramRef programRef,
    MetalBufferRef* inputRefs,
    MetalBufferRef outputRef,
    int inputCount,
    uint32_t count,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_fusion_status_set(status, 0, NULL);

        if (count == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;
        MetalFusionProgram* program = (MetalFusionProgram*)programRef;

        if (context == NULL || program == NULL || program->pipeline == NULL) {
            metal_fusion_status_set(status, -10, "fusion program: invalid dispatch context");
            return -10;
        }

        if (inputRefs == NULL || outputRef == NULL || inputCount < 0) {
            metal_fusion_status_set(status, -11, "fusion program: invalid dispatch buffers");
            return -11;
        }

        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)program->pipeline;
        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = metal_get_encoder(context, &commandBuffer);

        if (encoder == nil) {
            metal_fusion_status_set(status, -12, "fusion program: compute encoder unavailable");
            return -12;
        }

        [encoder setComputePipelineState:pipeline];

        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            if (inputRefs[inputIndex] == NULL) {
                metal_fusion_status_set(status, -13, "fusion program: nil input buffer");
                return -13;
            }

            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRefs[inputIndex]
                          offset:0
                         atIndex:(NSUInteger)inputIndex];
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)outputRef offset:0 atIndex:(NSUInteger)inputCount];
        [encoder setBytes:&count length:sizeof(count) atIndex:(NSUInteger)inputCount + 1];

        NSUInteger threadgroupWidth = pipeline.threadExecutionWidth;

        if (threadgroupWidth == 0) {
            threadgroupWidth = 256;
        }

        NSUInteger threadgroups = (count + (uint32_t)threadgroupWidth - 1) / (uint32_t)threadgroupWidth;
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroupWidth, 1, 1)];
        metal_end_encoder(context, encoder, commandBuffer);

        return 0;
    }
}
