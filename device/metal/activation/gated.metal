#include <metal_stdlib>
#include "activation_gated_ops_f32.metalinc"
#include "activation_gated_ops_f16.metalinc"
#include "activation_gated_ops_bf16.metalinc"
#include "activation_gated.macros.metalinc"

using namespace metal;

GATED_TENSOR_KERNEL_F32(
    swiglu,
    gated_swiglu_f4(gate[index], up[index]),
    gated_swiglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    geglu,
    gated_geglu_f4(gate[index], up[index]),
    gated_geglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    glu,
    gated_glu_f4(gate[index], up[index]),
    gated_glu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    reglu,
    gated_reglu_f4(gate[index], up[index]),
    gated_reglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    siglu,
    gated_siglu_f4(gate[index], up[index]),
    gated_siglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    seglu,
    gated_seglu_f4(gate[index], up[index]),
    gated_seglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    linglu,
    gated_linglu_f4(gate[index], up[index]),
    gated_linglu_f(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F32(
    geglu_tanh,
    gated_geglu_tanh_f4(gate[index], up[index]),
    gated_geglu_tanh_f(gateValue, upValue)
)

GATED_TENSOR_KERNEL_F16(
    swiglu,
    gated_swiglu_h4(gate[index], up[index]),
    gated_swiglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    geglu,
    gated_geglu_h4(gate[index], up[index]),
    gated_geglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    glu,
    gated_glu_h4(gate[index], up[index]),
    gated_glu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    reglu,
    gated_reglu_h4(gate[index], up[index]),
    gated_reglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    siglu,
    gated_siglu_h4(gate[index], up[index]),
    gated_siglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    seglu,
    gated_seglu_h4(gate[index], up[index]),
    gated_seglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    linglu,
    gated_linglu_h4(gate[index], up[index]),
    gated_linglu_h(gateValue, upValue)
)
GATED_TENSOR_KERNEL_F16(
    geglu_tanh,
    gated_geglu_tanh_h4(gate[index], up[index]),
    gated_geglu_tanh_h(gateValue, upValue)
)

GATED_TENSOR_KERNEL_BF16(
    swiglu,
    gated_swiglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_swiglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    geglu,
    gated_geglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_geglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    glu,
    gated_glu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_glu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    reglu,
    gated_reglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_reglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    siglu,
    gated_siglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_siglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    seglu,
    gated_seglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_seglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    linglu,
    gated_linglu_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_linglu_bf16(gateValue, upValue)
)
GATED_TENSOR_KERNEL_BF16(
    geglu_tanh,
    gated_geglu_tanh_bf164(as_type<bfloat4>(gate[index]), as_type<bfloat4>(up[index])),
    gated_geglu_tanh_bf16(gateValue, upValue)
)

GATED_PACKED_KERNEL_F32(swiglu, gated_swiglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(geglu, gated_geglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(glu, gated_glu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(reglu, gated_reglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(siglu, gated_siglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(seglu, gated_seglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(linglu, gated_linglu_f(gateValue, upValue))
GATED_PACKED_KERNEL_F32(geglu_tanh, gated_geglu_tanh_f(gateValue, upValue))

GATED_PACKED_KERNEL_F16(swiglu, gated_swiglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(geglu, gated_geglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(glu, gated_glu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(reglu, gated_reglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(siglu, gated_siglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(seglu, gated_seglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(linglu, gated_linglu_h(gateValue, upValue))
GATED_PACKED_KERNEL_F16(geglu_tanh, gated_geglu_tanh_h(gateValue, upValue))

GATED_PACKED_KERNEL_BF16(swiglu, gated_swiglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(geglu, gated_geglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(glu, gated_glu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(reglu, gated_reglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(siglu, gated_siglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(seglu, gated_seglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(linglu, gated_linglu_bf16(gateValue, upValue))
GATED_PACKED_KERNEL_BF16(geglu_tanh, gated_geglu_tanh_bf16(gateValue, upValue))
