#include <metal_stdlib>
using namespace metal;

kernel void int8_dequant(
    device float* destination [[buffer(0)]],
    device const char* source [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& zeroPoint [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    destination[index] = float(int(source[index]) - zeroPoint) * scale;
}

kernel void int4_dequant(
    device float* destination [[buffer(0)]],
    device const char* source [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& zeroPoint [[buffer(3)]],
    constant uint& pairCount [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= pairCount) {
        return;
    }

    uint byteIndex = index / 2u;
    uint nibble = index & 1u;
    int packed = int(source[byteIndex]);
    int value = (nibble == 0u) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

    if (value >= 8) {
        value -= 16;
    }

    destination[index] = float(value - zeroPoint) * scale;
}
