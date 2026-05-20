package metal

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

/*
GraphKernelName maps manifest operation IDs to Metal graph dispatch names.
*/
func GraphKernelName(operationID string) string {
	if mapped, ok := graphKernelAliases[operationID]; ok {
		return mapped
	}

	text := operationID

	for index := len(text) - 1; index >= 0; index-- {
		if text[index] != '.' {
			continue
		}

		return text[index+1:]
	}

	return text
}

/*
DispatchGraphKernel runs a Metal compute kernel by name on resident tensors.
This is the graph execution path; it does not use kernels.Default lookup.
*/
func DispatchGraphKernel(name string, args ...tensor.Tensor) error {
	runner, ok := graphKernelRunners[name]

	if !ok {
		return fmt.Errorf("metal graph dispatch: unknown kernel %q", name)
	}

	return runner(args...)
}

var graphKernelAliases = map[string]string{
	"projection.linear":            "linear",
	"projection.fused_qkv":         "fused_qkv",
	"math.rmsnorm":                 "rmsnorm",
	"math.layernorm":               "layernorm",
	"math.add":                     "add",
	"math.mul":                     "mul",
	"math.matmul":                  "matmul",
	"math.softmax":                 "softmax",
	"embedding.token":              "embedding",
	"attention.gqa":                "grouped_query_attention",
	"attention.sdpa":               "attention",
	"activation.swiglu":            "swiglu",
	"activation.gelu":              "gelu",
	"activation.relu":              "relu",
	"shape.concat":                 "concat",
	"shape.view_as_heads":          "view_as_heads",
	"shape.merge_heads":            "merge_heads",
	"positional.rope":              "rope",
	"pooling.max_pool2d":           "max_pool2d",
	"pooling.avg_pool2d":           "avg_pool2d",
	"convolution.conv2d":           "conv2d",
	"convolution.conv_transpose2d": "conv_transpose2d",
	"sampling.topk_sample":         "topk_sample",
}

var graphKernelRunners = map[string]func(...tensor.Tensor) error{
	"add":                      runBinaryFloat32(metalBinaryFloat32Add),
	"sub":                      runBinaryFloat32(metalBinaryFloat32Sub),
	"mul":                      runBinaryFloat32(metalBinaryFloat32Mul),
	"div":                      runBinaryFloat32(metalBinaryFloat32Div),
	"max":                      runBinaryFloat32(metalBinaryFloat32Max),
	"min":                      runBinaryFloat32(metalBinaryFloat32Min),
	"matmul":                   runMetalMatMulKernel,
	"matmul_add":               runMetalMatMulAddKernel,
	"linear":                   runMetalLinearKernel,
	"fused_qkv":                runMetalFusedQKVKernel,
	"rmsnorm":                  runMetalRMSNormKernel,
	"layernorm":                runMetalLayerNormKernel,
	"softmax":                  runMetalSoftmaxKernel,
	"gelu":                     runExtendedUnaryElementwise(metalUnaryFloat32Gelu),
	"relu":                     runUnaryFloat32(metalUnaryFloat32Relu),
	"tanh":                     runExtendedUnaryElementwise(metalUnaryFloat32Tanh),
	"sigmoid":                  runExtendedUnaryElementwise(metalUnaryFloat32Sigmoid),
	"silu":                     runExtendedUnaryElementwise(metalUnaryFloat32Silu),
	"swish":                    runExtendedUnaryElementwise(metalUnaryFloat32Swish),
	"swiglu":                   runMetalSwiGLUKernel,
	"geglu":                    runMetalGeGLUKernel,
	"glu":                      runMetalGLUKernel,
	"reglu":                    runMetalReGLUKernel,
	"siglu":                    runMetalSiGLUKernel,
	"seglu":                    runMetalSeGLUKernel,
	"linglu":                   runMetalLinGLUKernel,
	"geglu_tanh":               runMetalGeGLUTanhKernel,
	"embedding":                runMetalEmbeddingLookupKernel,
	"attention":                runMetalAttentionKernel,
	"flash_attention":          runMetalFlashAttentionKernel,
	"multi_head_attention":     runMetalMultiHeadAttentionKernel,
	"grouped_query_attention":  runMetalGroupedQueryAttentionKernel,
	"sliding_window_attention": runMetalSlidingWindowAttentionKernel,
	"rope":                     runMetalRoPEKernel,
	"concat":                   runBinaryShape(runMetalConcat),
	"view_as_heads":            runViewAsHeadsShape(runMetalViewAsHeads),
	"merge_heads":              runUnaryShape(runMetalMergeHeads),
	"max_pool2d":               runMetalMaxPool2DKernel,
	"avg_pool2d":               runMetalAvgPool2DKernel,
	"conv2d":                   runMetalConv2DKernel,
	"conv_transpose2d":         runMetalConvTranspose2DKernel,
	"topk_sample":              runMetalSamplingKernel(metalSamplingTopK),
	"greedy_sample":            runMetalSamplingKernel(metalSamplingGreedy),
	"topp_sample":              runMetalSamplingKernel(metalSamplingTopP),
}
