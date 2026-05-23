#include <metal_stdlib>

using namespace metal;

static inline float research_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort research_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32ResearchStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16ResearchStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16ResearchStorage {
    static float load(device const ushort* values, uint index) {
        return research_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = research_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline void vsa_bind_kernel(
    device const Scalar* left,
    device const Scalar* right,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(out, index, Storage::load(left, index) * Storage::load(right, index));
}

template <typename Storage, typename Scalar>
static inline void vsa_bundle_kernel(
    device const Scalar* left,
    device const Scalar* right,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(out, index, Storage::load(left, index) + Storage::load(right, index));
}

template <typename Storage, typename Scalar>
static inline void vsa_permute_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count || count == 0) {
        return;
    }

    uint target = index + 1;
    if (target == count) {
        target = 0;
    }

    Storage::store(out, target, Storage::load(input, index));
}

template <typename Storage, typename Scalar>
static inline void vsa_inverse_permute_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count || count == 0) {
        return;
    }

    uint target = index == 0 ? count - 1 : index - 1;
    Storage::store(out, target, Storage::load(input, index));
}

#define RESEARCH_BINARY_KERNEL(name, body, storage, scalar) \
kernel void name( \
    device const scalar* left [[buffer(0)]], \
    device const scalar* right [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(left, right, out, count, index); \
}

#define RESEARCH_UNARY_KERNEL(name, body, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(input, out, count, index); \
}
