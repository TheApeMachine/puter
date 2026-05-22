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
