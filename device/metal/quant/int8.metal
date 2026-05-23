#include "quant.metal"

using namespace metal;

kernel void int8_quant(
    device const float* input [[buffer(0)]],
    device char* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    float rounded = round(input[index]);
    rounded = clamp(rounded, -128.0f, 127.0f);
    out[index] = char(rounded);
}
