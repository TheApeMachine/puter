#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_F32_CUH
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_F32_CUH

static __device__ __forceinline__ float4 elementwise_abs_f4(float4 value) {
    return make_float4(fabsf(value.x), fabsf(value.y), fabsf(value.z), fabsf(value.w));
}

static __device__ __forceinline__ float elementwise_abs_f1(float value) {
    return fabsf(value);
}

static __device__ __forceinline__ float4 elementwise_neg_f4(float4 value) {
    return make_float4(-value.x, -value.y, -value.z, -value.w);
}

static __device__ __forceinline__ float elementwise_neg_f1(float value) {
    return -value;
}

static __device__ __forceinline__ float4 elementwise_sqrt_f4(float4 value) {
    return make_float4(sqrtf(value.x), sqrtf(value.y), sqrtf(value.z), sqrtf(value.w));
}

static __device__ __forceinline__ float elementwise_sqrt_f1(float value) {
    return sqrtf(value);
}

static __device__ __forceinline__ float4 elementwise_relu_f4(float4 value) {
    return make_float4(
        fmaxf(0.0f, value.x),
        fmaxf(0.0f, value.y),
        fmaxf(0.0f, value.z),
        fmaxf(0.0f, value.w)
    );
}

static __device__ __forceinline__ float elementwise_relu_f1(float value) {
    return fmaxf(0.0f, value);
}

static __device__ __forceinline__ float4 elementwise_add_f4(float4 left, float4 right) {
    return make_float4(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
        left.w + right.w
    );
}

static __device__ __forceinline__ float elementwise_add_f1(float left, float right) {
    return left + right;
}

static __device__ __forceinline__ float4 elementwise_sub_f4(float4 left, float4 right) {
    return make_float4(
        left.x - right.x,
        left.y - right.y,
        left.z - right.z,
        left.w - right.w
    );
}

static __device__ __forceinline__ float elementwise_sub_f1(float left, float right) {
    return left - right;
}

static __device__ __forceinline__ float4 elementwise_mul_f4(float4 left, float4 right) {
    return make_float4(
        left.x * right.x,
        left.y * right.y,
        left.z * right.z,
        left.w * right.w
    );
}

static __device__ __forceinline__ float elementwise_mul_f1(float left, float right) {
    return left * right;
}

static __device__ __forceinline__ float4 elementwise_div_f4(float4 left, float4 right) {
    return make_float4(
        left.x / right.x,
        left.y / right.y,
        left.z / right.z,
        left.w / right.w
    );
}

static __device__ __forceinline__ float elementwise_div_f1(float left, float right) {
    return left / right;
}

static __device__ __forceinline__ float4 elementwise_max_f4(float4 left, float4 right) {
    return make_float4(
        fmaxf(left.x, right.x),
        fmaxf(left.y, right.y),
        fmaxf(left.z, right.z),
        fmaxf(left.w, right.w)
    );
}

static __device__ __forceinline__ float elementwise_max_f1(float left, float right) {
    return fmaxf(left, right);
}

static __device__ __forceinline__ float4 elementwise_min_f4(float4 left, float4 right) {
    return make_float4(
        fminf(left.x, right.x),
        fminf(left.y, right.y),
        fminf(left.z, right.z),
        fminf(left.w, right.w)
    );
}

static __device__ __forceinline__ float elementwise_min_f1(float left, float right) {
    return fminf(left, right);
}

#endif
