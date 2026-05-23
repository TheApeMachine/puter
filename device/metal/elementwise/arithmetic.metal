// --- elementwise_float32.metal ---
#include <metal_stdlib>

using namespace metal;

template <typename BinaryOp>
static inline void binary_float32(
    device const float4* leftVector,
    device const float4* rightVector,
    device float4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    BinaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = op(leftVector[i], rightVector[i]);
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const float* left = reinterpret_cast<device const float*>(leftVector);
            device const float* right = reinterpret_cast<device const float*>(rightVector);
            device float* out = reinterpret_cast<device float*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = op(left[scalarIndex], right[scalarIndex]);
            }
        }
    }
}

template <typename UnaryOp>
static inline void unary_float32(
    device const float4* inputVector,
    device float4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    UnaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = op(inputVector[i]);
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const float* input = reinterpret_cast<device const float*>(inputVector);
            device float* out = reinterpret_cast<device float*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = op(input[scalarIndex]);
            }
        }
    }
}

struct AddFloat32 {
    float4 operator()(float4 left, float4 right) const { return left + right; }
    float operator()(float left, float right) const { return left + right; }
};

struct SubFloat32 {
    float4 operator()(float4 left, float4 right) const { return left - right; }
    float operator()(float left, float right) const { return left - right; }
};

struct MulFloat32 {
    float4 operator()(float4 left, float4 right) const { return left * right; }
    float operator()(float left, float right) const { return left * right; }
};

struct DivFloat32 {
    float4 operator()(float4 left, float4 right) const { return left / right; }
    float operator()(float left, float right) const { return left / right; }
};

struct MaxFloat32 {
    float4 operator()(float4 left, float4 right) const { return max(left, right); }
    float operator()(float left, float right) const { return max(left, right); }
};

struct MinFloat32 {
    float4 operator()(float4 left, float4 right) const { return min(left, right); }
    float operator()(float left, float right) const { return min(left, right); }
};

struct EqFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left == right);
    }

    float operator()(float left, float right) const { return left == right ? 1.0 : 0.0; }
};

struct NeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left != right);
    }

    float operator()(float left, float right) const { return left != right ? 1.0 : 0.0; }
};

struct LtFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left < right);
    }

    float operator()(float left, float right) const { return left < right ? 1.0 : 0.0; }
};

struct LeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left <= right);
    }

    float operator()(float left, float right) const { return left <= right ? 1.0 : 0.0; }
};

struct GtFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left > right);
    }

    float operator()(float left, float right) const { return left > right ? 1.0 : 0.0; }
};

struct GeFloat32 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left >= right);
    }

    float operator()(float left, float right) const { return left >= right ? 1.0 : 0.0; }
};

struct PowFloat32 {
    float4 operator()(float4 left, float4 right) const { return precise::pow(left, right); }
    float operator()(float left, float right) const { return precise::pow(left, right); }
};

struct Atan2Float32 {
    float4 operator()(float4 left, float4 right) const { return precise::atan2(left, right); }
    float operator()(float left, float right) const { return precise::atan2(left, right); }
};

struct ModFloat32 {
    float4 operator()(float4 left, float4 right) const { return fmod(left, right); }
    float operator()(float left, float right) const { return fmod(left, right); }
};

struct ReluFloat32 {
    float4 operator()(float4 value) const { return max(value, float4(0.0)); }
    float operator()(float value) const { return max(value, 0.0); }
};

struct AbsFloat32 {
    float4 operator()(float4 value) const { return fabs(value); }
    float operator()(float value) const { return fabs(value); }
};

struct NegFloat32 {
    float4 operator()(float4 value) const { return -value; }
    float operator()(float value) const { return -value; }
};

struct SquareFloat32 {
    float4 operator()(float4 value) const { return value * value; }
    float operator()(float value) const { return value * value; }
};

struct RecipFloat32 {
    float4 operator()(float4 value) const { return float4(1.0) / value; }
    float operator()(float value) const { return 1.0 / value; }
};

struct SqrtFloat32 {
    float4 operator()(float4 value) const { return precise::sqrt(value); }
    float operator()(float value) const { return precise::sqrt(value); }
};

struct SignFloat32 {
    float4 operator()(float4 value) const {
        float4 positive = select(float4(0.0), float4(1.0), value > float4(0.0));
        return select(positive, float4(-1.0), value < float4(0.0));
    }

    float operator()(float value) const {
        if (value > 0.0) {
            return 1.0;
        }

        if (value < 0.0) {
            return -1.0;
        }

        return 0.0;
    }
};

kernel void add_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, AddFloat32{});
}

kernel void sub_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, SubFloat32{});
}

kernel void mul_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, MulFloat32{});
}

kernel void div_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, DivFloat32{});
}

kernel void max_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, MaxFloat32{});
}

kernel void min_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, MinFloat32{});
}

kernel void eq_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, EqFloat32{});
}

kernel void ne_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, NeFloat32{});
}

kernel void lt_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, LtFloat32{});
}

kernel void le_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, LeFloat32{});
}

kernel void gt_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, GtFloat32{});
}

kernel void ge_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, GeFloat32{});
}

kernel void pow_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, PowFloat32{});
}

kernel void atan2_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, Atan2Float32{});
}

kernel void mod_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    binary_float32(leftVector, rightVector, outVector, count, index, stride, ModFloat32{});
}

kernel void relu_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, ReluFloat32{});
}

kernel void abs_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, AbsFloat32{});
}

kernel void neg_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, NegFloat32{});
}

kernel void square_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, SquareFloat32{});
}

kernel void recip_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, RecipFloat32{});
}

kernel void sqrt_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, SqrtFloat32{});
}

kernel void sign_float32(
    device const float4* inputVector [[buffer(0)]],
    device float4* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    unary_float32(inputVector, outVector, count, index, stride, SignFloat32{});
}

// --- elementwise_float16.metal ---
#include <metal_stdlib>

using namespace metal;

template <typename BinaryOp>
static inline void binary_float16(
    device const half4* leftVector,
    device const half4* rightVector,
    device half4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    BinaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = op(leftVector[i], rightVector[i]);
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const half* left = reinterpret_cast<device const half*>(leftVector);
            device const half* right = reinterpret_cast<device const half*>(rightVector);
            device half* out = reinterpret_cast<device half*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = op(left[scalarIndex], right[scalarIndex]);
            }
        }
    }
}

template <typename UnaryOp>
static inline void unary_float16(
    device const half4* inputVector,
    device half4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    UnaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = op(inputVector[i]);
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const half* input = reinterpret_cast<device const half*>(inputVector);
            device half* out = reinterpret_cast<device half*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = op(input[scalarIndex]);
            }
        }
    }
}

struct AddFloat16 {
    half4 operator()(half4 left, half4 right) const { return left + right; }
    half operator()(half left, half right) const { return left + right; }
};

struct SubFloat16 {
    half4 operator()(half4 left, half4 right) const { return left - right; }
    half operator()(half left, half right) const { return left - right; }
};

struct MulFloat16 {
    half4 operator()(half4 left, half4 right) const { return left * right; }
    half operator()(half left, half right) const { return left * right; }
};

struct DivFloat16 {
    half4 operator()(half4 left, half4 right) const { return left / right; }
    half operator()(half left, half right) const { return left / right; }
};

struct MaxFloat16 {
    half4 operator()(half4 left, half4 right) const { return max(left, right); }
    half operator()(half left, half right) const { return max(left, right); }
};

struct MinFloat16 {
    half4 operator()(half4 left, half4 right) const { return min(left, right); }
    half operator()(half left, half right) const { return min(left, right); }
};

struct EqFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left == right);
    }

    half operator()(half left, half right) const { return left == right ? 1.0h : 0.0h; }
};

struct NeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left != right);
    }

    half operator()(half left, half right) const { return left != right ? 1.0h : 0.0h; }
};

struct LtFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left < right);
    }

    half operator()(half left, half right) const { return left < right ? 1.0h : 0.0h; }
};

struct LeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left <= right);
    }

    half operator()(half left, half right) const { return left <= right ? 1.0h : 0.0h; }
};

struct GtFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left > right);
    }

    half operator()(half left, half right) const { return left > right ? 1.0h : 0.0h; }
};

struct GeFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return select(half4(0.0h), half4(1.0h), left >= right);
    }

    half operator()(half left, half right) const { return left >= right ? 1.0h : 0.0h; }
};

struct PowFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return half4(pow(float4(left), float4(right)));
    }

    half operator()(half left, half right) const { return half(pow(float(left), float(right))); }
};

struct Atan2Float16 {
    half4 operator()(half4 left, half4 right) const {
        return half4(atan2(float4(left), float4(right)));
    }

    half operator()(half left, half right) const { return half(atan2(float(left), float(right))); }
};

struct ModFloat16 {
    half4 operator()(half4 left, half4 right) const {
        return half4(fmod(float4(left), float4(right)));
    }

    half operator()(half left, half right) const { return half(fmod(float(left), float(right))); }
};

struct ReluFloat16 {
    half4 operator()(half4 value) const { return max(value, half4(0.0h)); }
    half operator()(half value) const { return max(value, 0.0h); }
};

struct AbsFloat16 {
    half4 operator()(half4 value) const { return fabs(value); }
    half operator()(half value) const { return fabs(value); }
};

struct NegFloat16 {
    half4 operator()(half4 value) const { return -value; }
    half operator()(half value) const { return -value; }
};

struct SquareFloat16 {
    half4 operator()(half4 value) const { return value * value; }
    half operator()(half value) const { return value * value; }
};

struct RecipFloat16 {
    half4 operator()(half4 value) const { return half4(1.0h) / value; }
    half operator()(half value) const { return 1.0h / value; }
};

struct SqrtFloat16 {
    half4 operator()(half4 value) const { return sqrt(value); }
    half operator()(half value) const { return sqrt(value); }
};

struct SignFloat16 {
    half4 operator()(half4 value) const {
        half4 positive = select(half4(0.0h), half4(1.0h), value > half4(0.0h));
        return select(positive, half4(-1.0h), value < half4(0.0h));
    }

    half operator()(half value) const {
        if (value > 0.0h) {
            return 1.0h;
        }

        if (value < 0.0h) {
            return -1.0h;
        }

        return 0.0h;
    }
};

#define BINARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* leftVector [[buffer(0)]], \
    device const half4* rightVector [[buffer(1)]], \
    device half4* outVector [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]], \
    uint stride [[threads_per_grid]] \
) { \
    binary_float16(leftVector, rightVector, outVector, count, index, stride, op{}); \
}

#define UNARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* inputVector [[buffer(0)]], \
    device half4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]], \
    uint stride [[threads_per_grid]] \
) { \
    unary_float16(inputVector, outVector, count, index, stride, op{}); \
}

BINARY_FLOAT16_KERNEL(add, AddFloat16)
BINARY_FLOAT16_KERNEL(sub, SubFloat16)
BINARY_FLOAT16_KERNEL(mul, MulFloat16)
BINARY_FLOAT16_KERNEL(div, DivFloat16)
BINARY_FLOAT16_KERNEL(max, MaxFloat16)
BINARY_FLOAT16_KERNEL(min, MinFloat16)
BINARY_FLOAT16_KERNEL(eq, EqFloat16)
BINARY_FLOAT16_KERNEL(ne, NeFloat16)
BINARY_FLOAT16_KERNEL(lt, LtFloat16)
BINARY_FLOAT16_KERNEL(le, LeFloat16)
BINARY_FLOAT16_KERNEL(gt, GtFloat16)
BINARY_FLOAT16_KERNEL(ge, GeFloat16)
BINARY_FLOAT16_KERNEL(pow, PowFloat16)
BINARY_FLOAT16_KERNEL(atan2, Atan2Float16)
BINARY_FLOAT16_KERNEL(mod, ModFloat16)

UNARY_FLOAT16_KERNEL(relu, ReluFloat16)
UNARY_FLOAT16_KERNEL(abs, AbsFloat16)
UNARY_FLOAT16_KERNEL(neg, NegFloat16)
UNARY_FLOAT16_KERNEL(square, SquareFloat16)
UNARY_FLOAT16_KERNEL(recip, RecipFloat16)
UNARY_FLOAT16_KERNEL(sqrt, SqrtFloat16)
UNARY_FLOAT16_KERNEL(sign, SignFloat16)

// --- elementwise_bfloat16.metal ---
#include <metal_stdlib>

using namespace metal;

static inline float bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 bf16_to_float4(ushort4 value) {
    return float4(
        bf16_to_float(value.x),
        bf16_to_float(value.y),
        bf16_to_float(value.z),
        bf16_to_float(value.w)
    );
}

static inline ushort4 float4_to_bf16(float4 value) {
    return ushort4(
        float_to_bf16(value.x),
        float_to_bf16(value.y),
        float_to_bf16(value.z),
        float_to_bf16(value.w)
    );
}

template <typename BinaryOp>
static inline void binary_bfloat16(
    device const ushort4* leftVector,
    device const ushort4* rightVector,
    device ushort4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    BinaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = float4_to_bf16(op(bf16_to_float4(leftVector[i]), bf16_to_float4(rightVector[i])));
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const ushort* left = reinterpret_cast<device const ushort*>(leftVector);
            device const ushort* right = reinterpret_cast<device const ushort*>(rightVector);
            device ushort* out = reinterpret_cast<device ushort*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = float_to_bf16(op(bf16_to_float(left[scalarIndex]), bf16_to_float(right[scalarIndex])));
            }
        }
    }
}

template <typename UnaryOp>
static inline void unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index,
    uint stride,
    UnaryOp op
) {
    uint vectorCount = count / 4;
    for (uint i = index; i < vectorCount; i += stride) {
        outVector[i] = float4_to_bf16(op(bf16_to_float4(inputVector[i])));
    }

    if (index == 0) {
        uint remainder = count % 4;
        if (remainder > 0) {
            device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
            device ushort* out = reinterpret_cast<device ushort*>(outVector);
            
            for (uint offset = 0; offset < remainder; offset++) {
                uint scalarIndex = vectorCount * 4 + offset;
                out[scalarIndex] = float_to_bf16(op(bf16_to_float(input[scalarIndex])));
            }
        }
    }
}

struct AddBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left + right; }
    float operator()(float left, float right) const { return left + right; }
};

struct SubBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left - right; }
    float operator()(float left, float right) const { return left - right; }
};

struct MulBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left * right; }
    float operator()(float left, float right) const { return left * right; }
};

struct DivBFloat16 {
    float4 operator()(float4 left, float4 right) const { return left / right; }
    float operator()(float left, float right) const { return left / right; }
};

struct MaxBFloat16 {
    float4 operator()(float4 left, float4 right) const { return max(left, right); }
    float operator()(float left, float right) const { return max(left, right); }
};

struct MinBFloat16 {
    float4 operator()(float4 left, float4 right) const { return min(left, right); }
    float operator()(float left, float right) const { return min(left, right); }
};

struct EqBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left == right);
    }

    float operator()(float left, float right) const { return left == right ? 1.0 : 0.0; }
};

struct NeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left != right);
    }

    float operator()(float left, float right) const { return left != right ? 1.0 : 0.0; }
};

struct LtBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left < right);
    }

    float operator()(float left, float right) const { return left < right ? 1.0 : 0.0; }
};

struct LeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left <= right);
    }

    float operator()(float left, float right) const { return left <= right ? 1.0 : 0.0; }
};

struct GtBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left > right);
    }

    float operator()(float left, float right) const { return left > right ? 1.0 : 0.0; }
};

struct GeBFloat16 {
    float4 operator()(float4 left, float4 right) const {
        return select(float4(0.0), float4(1.0), left >= right);
    }

    float operator()(float left, float right) const { return left >= right ? 1.0 : 0.0; }
};

struct PowBFloat16 {
    float4 operator()(float4 left, float4 right) const { return pow(left, right); }
    float operator()(float left, float right) const { return pow(left, right); }
};

struct Atan2BFloat16 {
    float4 operator()(float4 left, float4 right) const { return atan2(left, right); }
    float operator()(float left, float right) const { return atan2(left, right); }
};

struct ModBFloat16 {
    float4 operator()(float4 left, float4 right) const { return fmod(left, right); }
    float operator()(float left, float right) const { return fmod(left, right); }
};

struct ReluBFloat16 {
    float4 operator()(float4 value) const { return max(value, float4(0.0)); }
    float operator()(float value) const { return max(value, 0.0); }
};

struct AbsBFloat16 {
    float4 operator()(float4 value) const { return fabs(value); }
    float operator()(float value) const { return fabs(value); }
};

struct NegBFloat16 {
    float4 operator()(float4 value) const { return -value; }
    float operator()(float value) const { return -value; }
};

struct SquareBFloat16 {
    float4 operator()(float4 value) const { return value * value; }
    float operator()(float value) const { return value * value; }
};

struct RecipBFloat16 {
    float4 operator()(float4 value) const { return float4(1.0) / value; }
    float operator()(float value) const { return 1.0 / value; }
};

struct SqrtBFloat16 {
    float4 operator()(float4 value) const { return sqrt(value); }
    float operator()(float value) const { return sqrt(value); }
};

struct SignBFloat16 {
    float4 operator()(float4 value) const {
        float4 positive = select(float4(0.0), float4(1.0), value > float4(0.0));
        return select(positive, float4(-1.0), value < float4(0.0));
    }

    float operator()(float value) const {
        if (value > 0.0) {
            return 1.0;
        }

        if (value < 0.0) {
            return -1.0;
        }

        return 0.0;
    }
};

#define BINARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* leftVector [[buffer(0)]], \
    device const ushort4* rightVector [[buffer(1)]], \
    device ushort4* outVector [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]], \
    uint stride [[threads_per_grid]] \
) { \
    binary_bfloat16(leftVector, rightVector, outVector, count, index, stride, op{}); \
}

#define UNARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* inputVector [[buffer(0)]], \
    device ushort4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]], \
    uint stride [[threads_per_grid]] \
) { \
    unary_bfloat16(inputVector, outVector, count, index, stride, op{}); \
}

BINARY_BFLOAT16_KERNEL(add, AddBFloat16)
BINARY_BFLOAT16_KERNEL(sub, SubBFloat16)
BINARY_BFLOAT16_KERNEL(mul, MulBFloat16)
BINARY_BFLOAT16_KERNEL(div, DivBFloat16)
BINARY_BFLOAT16_KERNEL(max, MaxBFloat16)
BINARY_BFLOAT16_KERNEL(min, MinBFloat16)
BINARY_BFLOAT16_KERNEL(eq, EqBFloat16)
BINARY_BFLOAT16_KERNEL(ne, NeBFloat16)
BINARY_BFLOAT16_KERNEL(lt, LtBFloat16)
BINARY_BFLOAT16_KERNEL(le, LeBFloat16)
BINARY_BFLOAT16_KERNEL(gt, GtBFloat16)
BINARY_BFLOAT16_KERNEL(ge, GeBFloat16)
BINARY_BFLOAT16_KERNEL(pow, PowBFloat16)
BINARY_BFLOAT16_KERNEL(atan2, Atan2BFloat16)
BINARY_BFLOAT16_KERNEL(mod, ModBFloat16)

UNARY_BFLOAT16_KERNEL(relu, ReluBFloat16)
UNARY_BFLOAT16_KERNEL(abs, AbsBFloat16)
UNARY_BFLOAT16_KERNEL(neg, NegBFloat16)
UNARY_BFLOAT16_KERNEL(square, SquareBFloat16)
UNARY_BFLOAT16_KERNEL(recip, RecipBFloat16)
UNARY_BFLOAT16_KERNEL(sqrt, SqrtBFloat16)
UNARY_BFLOAT16_KERNEL(sign, SignBFloat16)

// --- elementwise_float64.metal ---
#include <metal_stdlib>
#include "elementwise_f64_soft.metalinc"

using namespace metal;

kernel void add_float64(
    device const ulong* leftVector [[buffer(0)]],
    device const ulong* rightVector [[buffer(1)]],
    device ulong* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4;

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            outVector[scalarIndex] = metal_sf64_add(
                leftVector[scalarIndex],
                rightVector[scalarIndex]
            );
        }
    }
}

// --- elementwise_fused.metal ---
#include <metal_stdlib>

using namespace metal;

static inline float axpy_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort axpy_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

kernel void axpy_float32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    uint vectorCount = count / 4;
    device float4* yVector = reinterpret_cast<device float4*>(y);
    device const float4* xVector = reinterpret_cast<device const float4*>(x);

    for (uint i = index; i < vectorCount; i += stride) {
        yVector[i] += float(alpha) * xVector[i];
    }

    if (index == 0) {
        uint remainder = count % 4;
        for (uint offset = 0; offset < remainder; offset++) {
            uint scalarIndex = vectorCount * 4 + offset;
            y[scalarIndex] += float(alpha) * x[scalarIndex];
        }
    }
}

kernel void axpy_float16(
    device half4* yVector [[buffer(0)]],
    device const half4* xVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    uint vectorCount = count / 4;

    for (uint vectorIndex = index; vectorIndex < vectorCount; vectorIndex += stride) {
        half4 yPacked = yVector[vectorIndex];
        half4 xPacked = xVector[vectorIndex];
        yVector[vectorIndex] = yPacked + half(alpha) * xPacked;
    }

    if (index == 0) {
        uint remainder = count % 4;
        device half* yScalar = reinterpret_cast<device half*>(yVector);
        device const half* xScalar = reinterpret_cast<device const half*>(xVector);

        for (uint offset = 0; offset < remainder; offset++) {
            uint scalarIndex = vectorCount * 4 + offset;
            yScalar[scalarIndex] += half(alpha) * xScalar[scalarIndex];
        }
    }
}

kernel void axpy_bfloat16(
    device ushort4* yVector [[buffer(0)]],
    device const ushort4* xVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    uint vectorCount = count / 4;

    for (uint vectorIndex = index; vectorIndex < vectorCount; vectorIndex += stride) {
        ushort4 yPacked = yVector[vectorIndex];
        ushort4 xPacked = xVector[vectorIndex];
        float4 yValues = float4(
            axpy_bf16_to_float(yPacked.x),
            axpy_bf16_to_float(yPacked.y),
            axpy_bf16_to_float(yPacked.z),
            axpy_bf16_to_float(yPacked.w)
        );
        float4 xValues = float4(
            axpy_bf16_to_float(xPacked.x),
            axpy_bf16_to_float(xPacked.y),
            axpy_bf16_to_float(xPacked.z),
            axpy_bf16_to_float(xPacked.w)
        );
        float4 result = yValues + float(alpha) * xValues;
        yVector[vectorIndex] = ushort4(
            axpy_float_to_bf16(result.x),
            axpy_float_to_bf16(result.y),
            axpy_float_to_bf16(result.z),
            axpy_float_to_bf16(result.w)
        );
    }

    if (index == 0) {
        uint remainder = count % 4;
        device ushort* yScalar = reinterpret_cast<device ushort*>(yVector);
        device const ushort* xScalar = reinterpret_cast<device const ushort*>(xVector);

        for (uint offset = 0; offset < remainder; offset++) {
            uint scalarIndex = vectorCount * 4 + offset;
            float result = axpy_bf16_to_float(yScalar[scalarIndex]) +
                float(alpha) * axpy_bf16_to_float(xScalar[scalarIndex]);
            yScalar[scalarIndex] = axpy_float_to_bf16(result);
        }
    }
}

static inline float dot_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

kernel void dot_float32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    float partial = 0.0f;

    for (uint offset = index; offset < count; offset += stride) {
        partial += left[offset] * right[offset];
    }

    float simd_partial = simd_sum(partial);

    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(out, simd_partial, memory_order_relaxed);
    }
}

kernel void dot_bfloat16(
    device const ushort* left [[buffer(0)]],
    device const ushort* right [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    float partial = 0.0f;

    for (uint offset = index; offset < count; offset += stride) {
        partial += dot_bf16_to_float(left[offset]) * dot_bf16_to_float(right[offset]);
    }

    float simd_partial = simd_sum(partial);

    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(out, simd_partial, memory_order_relaxed);
    }
}

kernel void dot_float16(
    device const half* left [[buffer(0)]],
    device const half* right [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    float partial = 0.0f;

    for (uint offset = index; offset < count; offset += stride) {
        partial += float(left[offset]) * float(right[offset]);
    }

    float simd_partial = simd_sum(partial);

    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(out, simd_partial, memory_order_relaxed);
    }
}

