#include "activation.cuh"

#define GATED_TENSOR_KERNEL_F32(name, expr_f4, expr_scalar) \
extern "C" __global__ void name##_float32( \
    float* destinationRaw, \
    const float* gateRaw, \
    const float* upRaw, \
    unsigned int count \
) { \
    float4* destination = reinterpret_cast<float4*>(destinationRaw); \
    const float4* gate = reinterpret_cast<const float4*>(gateRaw); \
    const float4* up = reinterpret_cast<const float4*>(upRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        destination[vectorIndex] = expr_f4; \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            float gateValue = gateRaw[scalarIndex]; \
            float upValue = upRaw[scalarIndex]; \
            destinationRaw[scalarIndex] = expr_scalar; \
        } \
    } \
}

#define SWIGLU_F4(gateValue, upValue) \
    make_float4( \
        activation_silu((gateValue).x) * (upValue).x, \
        activation_silu((gateValue).y) * (upValue).y, \
        activation_silu((gateValue).z) * (upValue).z, \
        activation_silu((gateValue).w) * (upValue).w \
    )

#define SWIGLU_SCALAR activation_silu(gateValue) * upValue

GATED_TENSOR_KERNEL_F32(swiglu, SWIGLU_F4(gate[vectorIndex], up[vectorIndex]), SWIGLU_SCALAR)

#define GEGLU_F4(gateValue, upValue) \
    make_float4( \
        (gateValue).x * activation_gelu((upValue).x), \
        (gateValue).y * activation_gelu((upValue).y), \
        (gateValue).z * activation_gelu((upValue).z), \
        (gateValue).w * activation_gelu((upValue).w) \
    )

#define GEGLU_SCALAR gateValue * activation_gelu(upValue)

GATED_TENSOR_KERNEL_F32(geglu, GEGLU_F4(gate[vectorIndex], up[vectorIndex]), GEGLU_SCALAR)

#define GLU_F4(gateValue, upValue) \
    make_float4( \
        (gateValue).x * activation_sigmoid((upValue).x), \
        (gateValue).y * activation_sigmoid((upValue).y), \
        (gateValue).z * activation_sigmoid((upValue).z), \
        (gateValue).w * activation_sigmoid((upValue).w) \
    )

#define GLU_SCALAR gateValue * activation_sigmoid(upValue)

GATED_TENSOR_KERNEL_F32(glu, GLU_F4(gate[vectorIndex], up[vectorIndex]), GLU_SCALAR)

#define REGLU_F4(gateValue, upValue) \
    make_float4( \
        (gateValue).x * activation_relu((upValue).x), \
        (gateValue).y * activation_relu((upValue).y), \
        (gateValue).z * activation_relu((upValue).z), \
        (gateValue).w * activation_relu((upValue).w) \
    )

#define REGLU_SCALAR gateValue * activation_relu(upValue)

GATED_TENSOR_KERNEL_F32(reglu, REGLU_F4(gate[vectorIndex], up[vectorIndex]), REGLU_SCALAR)

#define SIGLU_F4(gateValue, upValue) \
    make_float4( \
        activation_sigmoid((gateValue).x) * (upValue).x, \
        activation_sigmoid((gateValue).y) * (upValue).y, \
        activation_sigmoid((gateValue).z) * (upValue).z, \
        activation_sigmoid((gateValue).w) * (upValue).w \
    )

#define SIGLU_SCALAR activation_sigmoid(gateValue) * upValue

GATED_TENSOR_KERNEL_F32(siglu, SIGLU_F4(gate[vectorIndex], up[vectorIndex]), SIGLU_SCALAR)

#define SEGLU_F4(gateValue, upValue) \
    make_float4( \
        (upValue).x * activation_sigmoid((gateValue).x), \
        (upValue).y * activation_sigmoid((gateValue).y), \
        (upValue).z * activation_sigmoid((gateValue).z), \
        (upValue).w * activation_sigmoid((gateValue).w) \
    )

#define SEGLU_SCALAR upValue * activation_sigmoid(gateValue)

GATED_TENSOR_KERNEL_F32(seglu, SEGLU_F4(gate[vectorIndex], up[vectorIndex]), SEGLU_SCALAR)

#define LINGLU_F4(gateValue, upValue) \
    make_float4( \
        (gateValue).x * (upValue).x, \
        (gateValue).y * (upValue).y, \
        (gateValue).z * (upValue).z, \
        (gateValue).w * (upValue).w \
    )

#define LINGLU_SCALAR gateValue * upValue

GATED_TENSOR_KERNEL_F32(linglu, LINGLU_F4(gate[vectorIndex], up[vectorIndex]), LINGLU_SCALAR)

#define GEGLU_TANH_F4(gateValue, upValue) \
    make_float4( \
        (gateValue).x * activation_gelu_tanh((upValue).x), \
        (gateValue).y * activation_gelu_tanh((upValue).y), \
        (gateValue).z * activation_gelu_tanh((upValue).z), \
        (gateValue).w * activation_gelu_tanh((upValue).w) \
    )

#define GEGLU_TANH_SCALAR gateValue * activation_gelu_tanh(upValue)

GATED_TENSOR_KERNEL_F32(geglu_tanh, GEGLU_TANH_F4(gate[vectorIndex], up[vectorIndex]), GEGLU_TANH_SCALAR)

#define GATED_TENSOR_KERNEL_F16(name, expr_h1) \
extern "C" __global__ void name##_float16( \
    __half* destination, \
    const __half* gate, \
    const __half* up, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __half gateLeft = gate[base]; \
        __half gateRight = gate[base + 1u]; \
        __half upLeft = up[base]; \
        __half upRight = up[base + 1u]; \
        destination[base] = expr_h1(gateLeft, upLeft); \
        destination[base + 1u] = expr_h1(gateRight, upRight); \
        return; \
    } \
    if (base < count) { \
        destination[base] = expr_h1(gate[base], up[base]); \
    } \
}

#define GATED_TENSOR_KERNEL_BF16(name, expr_b1) \
extern "C" __global__ void name##_bfloat16( \
    __nv_bfloat16* destination, \
    const __nv_bfloat16* gate, \
    const __nv_bfloat16* up, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __nv_bfloat16 gateLeft = gate[base]; \
        __nv_bfloat16 gateRight = gate[base + 1u]; \
        __nv_bfloat16 upLeft = up[base]; \
        __nv_bfloat16 upRight = up[base + 1u]; \
        destination[base] = expr_b1(gateLeft, upLeft); \
        destination[base + 1u] = expr_b1(gateRight, upRight); \
        return; \
    } \
    if (base < count) { \
        destination[base] = expr_b1(gate[base], up[base]); \
    } \
}

static __device__ __forceinline__ __half gated_swiglu_h1(__half gateValue, __half upValue) {
    return __hmul(activation_silu_h1(gateValue), upValue);
}

static __device__ __forceinline__ __half gated_geglu_h1(__half gateValue, __half upValue) {
    return __hmul(gateValue, activation_gelu_h1(upValue));
}

static __device__ __forceinline__ __half gated_glu_h1(__half gateValue, __half upValue) {
    return __hmul(gateValue, activation_sigmoid_h1(upValue));
}

static __device__ __forceinline__ __half gated_reglu_h1(__half gateValue, __half upValue) {
    return __hmul(gateValue, activation_relu_h1(upValue));
}

static __device__ __forceinline__ __half gated_siglu_h1(__half gateValue, __half upValue) {
    return __hmul(activation_sigmoid_h1(gateValue), upValue);
}

static __device__ __forceinline__ __half gated_seglu_h1(__half gateValue, __half upValue) {
    return __hmul(upValue, activation_sigmoid_h1(gateValue));
}

static __device__ __forceinline__ __half gated_linglu_h1(__half gateValue, __half upValue) {
    return __hmul(gateValue, upValue);
}

static __device__ __forceinline__ __half gated_geglu_tanh_h1(__half gateValue, __half upValue) {
    return __hmul(gateValue, activation_gelu_tanh_h1(upValue));
}

GATED_TENSOR_KERNEL_F16(swiglu, gated_swiglu_h1)
GATED_TENSOR_KERNEL_F16(geglu, gated_geglu_h1)
GATED_TENSOR_KERNEL_F16(glu, gated_glu_h1)
GATED_TENSOR_KERNEL_F16(reglu, gated_reglu_h1)
GATED_TENSOR_KERNEL_F16(siglu, gated_siglu_h1)
GATED_TENSOR_KERNEL_F16(seglu, gated_seglu_h1)
GATED_TENSOR_KERNEL_F16(linglu, gated_linglu_h1)
GATED_TENSOR_KERNEL_F16(geglu_tanh, gated_geglu_tanh_h1)

static __device__ __forceinline__ __nv_bfloat16 gated_swiglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(activation_silu_bf16(gateValue), upValue);
}

static __device__ __forceinline__ __nv_bfloat16 gated_geglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(gateValue, activation_gelu_bf16(upValue));
}

static __device__ __forceinline__ __nv_bfloat16 gated_glu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(gateValue, activation_sigmoid_bf16(upValue));
}

static __device__ __forceinline__ __nv_bfloat16 gated_reglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(gateValue, activation_relu_bf16(upValue));
}

static __device__ __forceinline__ __nv_bfloat16 gated_siglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(activation_sigmoid_bf16(gateValue), upValue);
}

static __device__ __forceinline__ __nv_bfloat16 gated_seglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(upValue, activation_sigmoid_bf16(gateValue));
}

static __device__ __forceinline__ __nv_bfloat16 gated_linglu_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(gateValue, upValue);
}

static __device__ __forceinline__ __nv_bfloat16 gated_geglu_tanh_bf16(__nv_bfloat16 gateValue, __nv_bfloat16 upValue) {
    return __hmul(gateValue, activation_gelu_tanh_bf16(upValue));
}

GATED_TENSOR_KERNEL_BF16(swiglu, gated_swiglu_bf16)
GATED_TENSOR_KERNEL_BF16(geglu, gated_geglu_bf16)
GATED_TENSOR_KERNEL_BF16(glu, gated_glu_bf16)
GATED_TENSOR_KERNEL_BF16(reglu, gated_reglu_bf16)
GATED_TENSOR_KERNEL_BF16(siglu, gated_siglu_bf16)
GATED_TENSOR_KERNEL_BF16(seglu, gated_seglu_bf16)
GATED_TENSOR_KERNEL_BF16(linglu, gated_linglu_bf16)
GATED_TENSOR_KERNEL_BF16(geglu_tanh, gated_geglu_tanh_bf16)

#define GATED_PACKED_KERNEL_F32(name, expr_scalar) \
extern "C" __global__ void name##_packed_float32( \
    float* destination, \
    const float* packed, \
    unsigned int inner, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    unsigned int batchIndex = index / inner; \
    unsigned int lane = index - batchIndex * inner; \
    unsigned int rowBase = batchIndex * inner * 2u; \
    float gateValue = packed[rowBase + lane]; \
    float upValue = packed[rowBase + inner + lane]; \
    destination[index] = expr_scalar; \
}

GATED_PACKED_KERNEL_F32(swiglu, SWIGLU_SCALAR)
GATED_PACKED_KERNEL_F32(geglu, GEGLU_SCALAR)
GATED_PACKED_KERNEL_F32(glu, GLU_SCALAR)
GATED_PACKED_KERNEL_F32(reglu, REGLU_SCALAR)
GATED_PACKED_KERNEL_F32(siglu, SIGLU_SCALAR)
GATED_PACKED_KERNEL_F32(seglu, SEGLU_SCALAR)
GATED_PACKED_KERNEL_F32(linglu, LINGLU_SCALAR)
GATED_PACKED_KERNEL_F32(geglu_tanh, GEGLU_TANH_SCALAR)

#define GATED_PACKED_KERNEL_F16(name, expr_h1) \
extern "C" __global__ void name##_packed_float16( \
    __half* destination, \
    const __half* packed, \
    unsigned int inner, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    unsigned int batchIndex = index / inner; \
    unsigned int lane = index - batchIndex * inner; \
    unsigned int rowBase = batchIndex * inner * 2u; \
    destination[index] = expr_h1(packed[rowBase + lane], packed[rowBase + inner + lane]); \
}

#define GATED_PACKED_KERNEL_BF16(name, expr_b1) \
extern "C" __global__ void name##_packed_bfloat16( \
    __nv_bfloat16* destination, \
    const __nv_bfloat16* packed, \
    unsigned int inner, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    unsigned int batchIndex = index / inner; \
    unsigned int lane = index - batchIndex * inner; \
    unsigned int rowBase = batchIndex * inner * 2u; \
    destination[index] = expr_b1(packed[rowBase + lane], packed[rowBase + inner + lane]); \
}

GATED_PACKED_KERNEL_F16(swiglu, gated_swiglu_h1)
GATED_PACKED_KERNEL_F16(geglu, gated_geglu_h1)
GATED_PACKED_KERNEL_F16(glu, gated_glu_h1)
GATED_PACKED_KERNEL_F16(reglu, gated_reglu_h1)
GATED_PACKED_KERNEL_F16(siglu, gated_siglu_h1)
GATED_PACKED_KERNEL_F16(seglu, gated_seglu_h1)
GATED_PACKED_KERNEL_F16(linglu, gated_linglu_h1)
GATED_PACKED_KERNEL_F16(geglu_tanh, gated_geglu_tanh_h1)

GATED_PACKED_KERNEL_BF16(swiglu, gated_swiglu_bf16)
GATED_PACKED_KERNEL_BF16(geglu, gated_geglu_bf16)
GATED_PACKED_KERNEL_BF16(glu, gated_glu_bf16)
GATED_PACKED_KERNEL_BF16(reglu, gated_reglu_bf16)
GATED_PACKED_KERNEL_BF16(siglu, gated_siglu_bf16)
GATED_PACKED_KERNEL_BF16(seglu, gated_seglu_bf16)
GATED_PACKED_KERNEL_BF16(linglu, gated_linglu_bf16)
GATED_PACKED_KERNEL_BF16(geglu_tanh, gated_geglu_tanh_bf16)
