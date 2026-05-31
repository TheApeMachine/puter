#include <metal_stdlib>
#include "../../elementwise/elementwise_f64_transcendental.metalinc"
#include "../../activation/activation.metal"

using namespace metal;

constant uint probeOutputWords = 10;

static inline float probe_fast_tanh_rational(float value) {
    return metal_fast_tanh_rational(value);
}

static inline void probe_write(
    device ulong* outputs,
    uint base,
    uint index,
    ulong value
) {
    outputs[base + index] = value;
}

kernel void sf64_transcendental_probe(
    device const float* inputs [[buffer(0)]],
    device const ulong* sqrtInputs [[buffer(1)]],
    device ulong* outputs [[buffer(2)]],
    constant uint& caseCount [[buffer(3)]],
    uint threadIndex [[thread_position_in_grid]]
) {
    if (threadIndex >= caseCount) {
        return;
    }

    uint inputBase = threadIndex * 4u;
    uint outputBase = threadIndex * probeOutputWords;
    float uniformFirst = inputs[inputBase + 0u];
    float uniformSecond = inputs[inputBase + 1u];
    float geluValue = inputs[inputBase + 2u];
    ulong sqrtInput64 = sqrtInputs[threadIndex];

    ulong uniformFirst64 = metal_sf64_from_float32(uniformFirst);

    if (uniformFirst == 0.0f) {
        uniformFirst64 = SF64_TRAN_SMALLEST_POSITIVE;
    }

    probe_write(outputs, outputBase, 0u, metal_sf64_log(uniformFirst64));
    probe_write(outputs, outputBase, 1u, metal_sf64_sqrt(sqrtInput64));

    ulong angle64 = metal_sf64_mul(
        SF64_TRAN_TWO_PI,
        metal_sf64_from_float32(uniformSecond)
    );
    ulong sin64;
    ulong cos64;
    metal_sf64_sincos(angle64, sin64, cos64);
    probe_write(outputs, outputBase, 2u, sin64);
    probe_write(outputs, outputBase, 3u, cos64);

    ulong geluValue64 = metal_sf64_from_float32(geluValue);
    ulong geluCube = metal_sf64_mul(
        metal_sf64_mul(geluValue64, geluValue64),
        geluValue64
    );
    ulong geluInner64 = metal_sf64_mul(
        SF64_TRAN_GELU_ALPHA,
        metal_sf64_add(geluValue64, metal_sf64_mul(SF64_TRAN_GELU_BETA, geluCube))
    );
    probe_write(outputs, outputBase, 4u, geluInner64);

    ulong sqrtVariance = metal_sf64_sqrt(sqrtInput64);
    ulong invStdDev64 = metal_sf64_div(SF64_TRAN_ONE, sqrtVariance);
    probe_write(outputs, outputBase, 5u, invStdDev64);

    ulong logFirst = metal_sf64_log(uniformFirst64);
    ulong negTwoLog = metal_sf64_mul(SF64_TRAN_NEG_TWO, logFirst);
    ulong magnitude64 = metal_sf64_sqrt(negTwoLog);
    probe_write(outputs, outputBase, 6u, magnitude64);

    float gaussianCos = as_type<float>(metal_sf64_to32(metal_sf64_mul(magnitude64, cos64)));
    float gaussianSin = as_type<float>(metal_sf64_to32(metal_sf64_mul(magnitude64, sin64)));
    probe_write(
        outputs,
        outputBase,
        7u,
        metal_sf32_to64(as_type<uint>(gaussianCos))
    );
    probe_write(
        outputs,
        outputBase,
        8u,
        metal_sf32_to64(as_type<uint>(gaussianSin))
    );

    float geluInner = as_type<float>(metal_sf64_to32(geluInner64));
    float geluTanh = probe_fast_tanh_rational(geluInner);
    ulong geluTanh64 = metal_sf64_from_float32(geluTanh);
    ulong geluProduct = metal_sf64_mul(
        SF64_TRAN_HALF,
        metal_sf64_mul(
            geluValue64,
            metal_sf64_add(SF64_TRAN_ONE, geluTanh64)
        )
    );
    probe_write(outputs, outputBase, 9u, geluProduct);
}
