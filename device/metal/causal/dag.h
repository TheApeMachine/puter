#ifndef PUTER_DEVICE_METAL_CAUSAL_DAG_H
#define PUTER_DEVICE_METAL_CAUSAL_DAG_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_dag_markov_factorization(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalsRef,
    MetalBufferRef parentsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
