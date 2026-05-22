#include <metal_stdlib>

using namespace metal;

static inline float masking_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort masking_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 masking_bf16_to_float4(ushort4 value) {
    return float4(
        masking_bf16_to_float(value.x),
        masking_bf16_to_float(value.y),
        masking_bf16_to_float(value.z),
        masking_bf16_to_float(value.w)
    );
}

static inline ushort4 masking_float4_to_bf16(float4 value) {
    return ushort4(
        masking_float_to_bf16(value.x),
        masking_float_to_bf16(value.y),
        masking_float_to_bf16(value.z),
        masking_float_to_bf16(value.w)
    );
}

static inline float masking_neg_inf_float32() {
    return as_type<float>(0xFF800000u);
}

static inline half masking_neg_inf_float16() {
    return as_type<half>(ushort(0xFC00));
}

static inline ushort masking_neg_inf_bfloat16() {
    return masking_float_to_bf16(masking_neg_inf_float32());
}

kernel void apply_mask_float32(
    device const float* input [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint vectorIndex [[thread_position_in_grid]]
) {
    uint base = vectorIndex * 4;

    if (base + 3 < count) {
        float4 leftVec = *((device const float4*)(input + base));
        float4 maskVec = *((device const float4*)(mask + base));
        *((device float4*)(out + base)) = leftVec + maskVec;
        return;
    }

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = input[scalarIndex] + mask[scalarIndex];
        }
    }
}

kernel void apply_mask_float16(
    device const half* input [[buffer(0)]],
    device const half* mask [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint vectorIndex [[thread_position_in_grid]]
) {
    uint base = vectorIndex * 4;

    if (base + 3 < count) {
        half4 leftVec = *((device const half4*)(input + base));
        half4 maskVec = *((device const half4*)(mask + base));
        *((device half4*)(out + base)) = leftVec + maskVec;
        return;
    }

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = input[scalarIndex] + mask[scalarIndex];
        }
    }
}

kernel void apply_mask_bfloat16(
    device const ushort* input [[buffer(0)]],
    device const ushort* mask [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint vectorIndex [[thread_position_in_grid]]
) {
    uint base = vectorIndex * 4;

    if (base + 3 < count) {
        ushort4 leftVec = *((device const ushort4*)(input + base));
        ushort4 maskVec = *((device const ushort4*)(mask + base));
        *((device ushort4*)(out + base)) = masking_float4_to_bf16(
            masking_bf16_to_float4(leftVec) + masking_bf16_to_float4(maskVec)
        );
        return;
    }

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            float leftValue = masking_bf16_to_float(input[scalarIndex]);
            float maskValue = masking_bf16_to_float(mask[scalarIndex]);
            out[scalarIndex] = masking_float_to_bf16(leftValue + maskValue);
        }
    }
}

kernel void causal_mask_float32(
    device float* out [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    float negInf = masking_neg_inf_float32();
    float4 zero4 = float4(0.0f);
    float4 negInf4 = float4(negInf);
    float rowValue = float(row);

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            float4 colIdx = float4(float(colBase)) + float4(0.0f, 1.0f, 2.0f, 3.0f);
            bool4 masked = colIdx > float4(rowValue);
            float4 result = select(zero4, negInf4, masked);
            *((device float4*)(out + rowBase + colBase)) = result;
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            out[rowBase + col] = col > row ? negInf : 0.0f;
        }
    }
}

kernel void causal_mask_float16(
    device half* out [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    half negInf = masking_neg_inf_float16();
    half4 zero4 = half4(0.0h);
    half4 negInf4 = half4(negInf);
    half rowValue = half(row);

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            half4 colIdx = half4(half(colBase)) + half4(0.0h, 1.0h, 2.0h, 3.0h);
            bool4 masked = colIdx > half4(rowValue);
            half4 result = select(zero4, negInf4, masked);
            *((device half4*)(out + rowBase + colBase)) = result;
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            out[rowBase + col] = col > row ? negInf : half(0.0h);
        }
    }
}

kernel void causal_mask_bfloat16(
    device ushort* out [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    ushort negInf = masking_neg_inf_bfloat16();
    ushort zero = ushort(0);

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            float4 colIdx = float4(float(colBase)) + float4(0.0f, 1.0f, 2.0f, 3.0f);
            bool4 masked = colIdx > float4(float(row));
            float4 zero4 = float4(0.0f);
            float4 negInf4 = float4(masking_bf16_to_float(negInf));
            float4 result = select(zero4, negInf4, masked);
            *((device ushort4*)(out + rowBase + colBase)) = masking_float4_to_bf16(result);
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            out[rowBase + col] = col > row ? negInf : zero;
        }
    }
}

kernel void alibi_bias_float32(
    device const float* scores [[buffer(0)]],
    device const float* slope [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    float slopeValue = slope[0];
    float4 row4 = float4(float(row));

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            float4 colIdx = float4(float(colBase)) + float4(0.0f, 1.0f, 2.0f, 3.0f);
            float4 score4 = *((device const float4*)(scores + rowBase + colBase));
            float4 dist4 = row4 - colIdx;
            bool4 apply = colIdx <= row4;
            float4 bias = float4(slopeValue) * dist4;
            float4 result = select(score4, score4 - bias, apply);
            *((device float4*)(out + rowBase + colBase)) = result;
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            uint index = rowBase + col;
            float scoreValue = scores[index];
            float outputValue = scoreValue;

            if (row >= col) {
                outputValue = scoreValue - slopeValue * float(row - col);
            }

            out[index] = outputValue;
        }
    }
}

kernel void alibi_bias_float16(
    device const half* scores [[buffer(0)]],
    device const half* slope [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    float slopeValue = float(slope[0]);
    float4 row4 = float4(float(row));

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            float4 colIdx = float4(float(colBase)) + float4(0.0f, 1.0f, 2.0f, 3.0f);
            half4 scoreHalf = *((device const half4*)(scores + rowBase + colBase));
            float4 score4 = float4(float(scoreHalf.x), float(scoreHalf.y), float(scoreHalf.z), float(scoreHalf.w));
            float4 dist4 = row4 - colIdx;
            bool4 apply = colIdx <= row4;
            float4 bias = float4(slopeValue) * dist4;
            float4 result = select(score4, score4 - bias, apply);
            *((device half4*)(out + rowBase + colBase)) = half4(
                half(result.x), half(result.y), half(result.z), half(result.w)
            );
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            uint index = rowBase + col;
            float scoreValue = float(scores[index]);
            float outputValue = scoreValue;

            if (row >= col) {
                outputValue = scoreValue - slopeValue * float(row - col);
            }

            out[index] = half(outputValue);
        }
    }
}

kernel void alibi_bias_bfloat16(
    device const ushort* scores [[buffer(0)]],
    device const ushort* slope [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint rowBase = row * cols;
    float slopeValue = masking_bf16_to_float(slope[0]);
    float4 row4 = float4(float(row));

    for (uint colBase = 0; colBase < cols; colBase += 4) {
        if (colBase + 3 < cols) {
            float4 colIdx = float4(float(colBase)) + float4(0.0f, 1.0f, 2.0f, 3.0f);
            ushort4 scorePacked = *((device const ushort4*)(scores + rowBase + colBase));
            float4 score4 = masking_bf16_to_float4(scorePacked);
            float4 dist4 = row4 - colIdx;
            bool4 apply = colIdx <= row4;
            float4 bias = float4(slopeValue) * dist4;
            float4 result = select(score4, score4 - bias, apply);
            *((device ushort4*)(out + rowBase + colBase)) = masking_float4_to_bf16(result);
            continue;
        }

        for (uint offset = 0; offset < 4; offset++) {
            uint col = colBase + offset;

            if (col >= cols) {
                return;
            }

            uint index = rowBase + col;
            float scoreValue = masking_bf16_to_float(scores[index]);
            float outputValue = scoreValue;

            if (row >= col) {
                outputValue = scoreValue - slopeValue * float(row - col);
            }

            out[index] = masking_float_to_bf16(outputValue);
        }
    }
}
