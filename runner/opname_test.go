package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
)

func TestKernelName(testingObject *testing.T) {
	convey.Convey("Given manifest operation IDs", testingObject, func() {
		cases := map[ir.OpID]string{
			"math.matmul":                  "matmul",
			"math.add":                     "add",
			"math.rmsnorm":                 "rmsnorm",
			"math.softmax":                 "softmax",
			"activation.relu":              "relu",
			"activation.swiglu":            "swiglu",
			"projection.linear":            "linear",
			"projection.fused_qkv":         "fused_qkv",
			"embedding.token":              "embedding_lookup",
			"positional.rope":              "rope",
			"positional.alibi":             "alibi_bias",
			"shape.view_as_heads":          "view_as_heads",
			"shape.merge_heads":            "merge_heads",
			"shape.split":                  "split2",
			"convolution.conv2d":           "conv2d",
			"pooling.max_pool2d":           "max_pool2d",
			"attention.gqa":                "grouped_query_attention",
			"attention.sdpa":               "flash_attention",
			"attention.mqa":                "flash_attention",
			"masking.causal":               "causal_mask",
			"causal.backdoor_adjustment":   "backdoor_adjustment",
			"causal.do_calculus":           "do_intervene",
			"hawkes.intensity":             "hawkes_intensity",
			"markov_blanket.flow_active":   "markov_flow_active",
			"vsa.bind":                     "vsa_bind",
			"active_inference.free_energy": "free_energy",
			"predictive_coding.prediction": "pc_prediction",
			"train.optimizer.adam":         "adam_step",
			"train.optimizer.hebbian":      "hebbian_step",
			"train.loss.mse":               "mse_loss",
			"train.loss.cross_entropy":     "cross_entropy",
			"stencil.laplacian":            "laplacian",
			"model.graft":                  "weight_graft_add",
			"state.page_write":             "page_write",
			"state.page_gather":            "page_gather",
			"state.page_alloc":             "page_alloc",
			"state.page_table_append":      "page_table_append",
		}

		for operationID, expectedKernel := range cases {
			convey.So(
				kernelName(operationID),
				convey.ShouldEqual,
				expectedKernel,
			)
		}
	})
}

func BenchmarkKernelName(benchmark *testing.B) {
	operationID := ir.OpID("math.matmul")

	for benchmark.Loop() {
		_ = kernelName(operationID)
	}
}
