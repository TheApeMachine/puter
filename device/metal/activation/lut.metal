#include <metal_stdlib>

using namespace metal;

constant uint lutGatherLaneWidth = 8;

kernel void lut_gather_float16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device const ushort* lut [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * lutGatherLaneWidth;

    if (base >= count) {
        return;
    }

    uint end = min(base + lutGatherLaneWidth, count);

    for (uint offset = base; offset < end; offset++) {
        ushort key = as_type<ushort>(input[offset]);
        output[offset] = as_type<half>(lut[key]);
    }
}

kernel void lut_gather_bfloat16(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device const ushort* lut [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * lutGatherLaneWidth;

    if (base >= count) {
        return;
    }

    uint end = min(base + lutGatherLaneWidth, count);

    for (uint offset = base; offset < end; offset++) {
        output[offset] = lut[input[offset]];
    }
}
