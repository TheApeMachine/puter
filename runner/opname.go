package runner

import (
	"strings"

	"github.com/theapemachine/manifesto/ir"
)

/*
kernelName maps manifest operation IDs to registered kernel names.

Manifest ops use dotted IDs (math.matmul, attention.gqa). Kernel registries
use flat names (matmul, grouped_query_attention). Most categories share the
suffix after the first dot; nested categories (train.optimizer.adam) and a
few renamed ops use explicit rules below.

Surfaces on device.Backend that never reach graph dispatch — runtime I/O,
control flow, checkpoints, metrics — are omitted here on purpose.
*/
func kernelName(operationID ir.OpID) string {
	identifier := string(operationID)

	if identifier == "" {
		return ""
	}

	if kernel, ok := manifestKernelNames[identifier]; ok {
		return kernel
	}

	for _, rule := range manifestCategoryRules {
		suffix, ok := strings.CutPrefix(identifier, rule.prefix)

		if !ok {
			continue
		}

		if kernel, ok := rule.aliases[suffix]; ok {
			return kernel
		}

		if rule.transform != nil {
			return rule.transform(suffix)
		}

		return suffix
	}

	return identifier
}

/*
manifestKernelNames holds manifest ops whose kernel name is not derivable
from category prefix rules alone.
*/
var manifestKernelNames = map[string]string{
	"projection.linear":        "linear",
	"projection.fused_qkv":     "fused_qkv",
	"embedding.lookup":         "embedding_lookup",
	"embedding.token":          "embedding_lookup",
	"attention.multi_head":     "multi_head_attention",
	"attention.flash":          "flash_attention",
	"attention.sdpa":           "flash_attention",
	"attention.mqa":            "flash_attention",
	"attention.grouped_query":  "grouped_query_attention",
	"attention.gqa":            "grouped_query_attention",
	"attention.sliding_window": "sliding_window_attention",
	"masking.causal":           "causal_mask",
	"masking.apply":            "apply_mask",
	"positional.alibi":         "alibi_bias",
	"causal.do_calculus":       "do_intervene",
	"shape.split":              "split2",
	"model.graft":              "weight_graft_add",
	"model.freeze":             "weight_freeze_mask",
	"model.lora":               "lora_apply",
	"state.page_write":         "page_write",
	"state.page_gather":        "page_gather",
	"state.page_alloc":         "page_alloc",
	"state.page_table_append":  "page_table_append",
}

type manifestCategoryRule struct {
	prefix    string
	aliases   map[string]string
	transform func(suffix string) string
}

/*
manifestCategoryRules is ordered longest-prefix-first so train.optimizer.*
matches before train.*.
*/
var manifestCategoryRules = []manifestCategoryRule{
	{
		prefix:    "train.optimizer.",
		transform: optimizerKernelName,
	},
	{
		prefix: "train.loss.",
		aliases: map[string]string{
			"mse":                  "mse_loss",
			"cross_entropy":        "cross_entropy",
			"binary_cross_entropy": "binary_cross_entropy",
			"huber":                "huber_loss",
			"kl_divergence":        "kl_divergence",
		},
	},
	{
		prefix:    "predictive_coding.",
		transform: predictiveCodingKernelName,
	},
	{
		prefix:    "markov_blanket.",
		transform: markovBlanketKernelName,
	},
	{
		prefix:    "active_inference.",
		transform: identityKernelSuffix,
	},
	{
		prefix:    "hawkes.",
		transform: hawkesKernelName,
	},
	{
		prefix:    "vsa.",
		transform: vsaKernelName,
	},
	{prefix: "convolution."},
	{prefix: "pooling."},
	{prefix: "causal."},
	{prefix: "positional."},
	{prefix: "shape."},
	{prefix: "attention."},
	{prefix: "masking."},
	{prefix: "embedding."},
	{prefix: "projection."},
	{prefix: "stencil."},
	{prefix: "activation."},
	{prefix: "math."},
}

func identityKernelSuffix(suffix string) string {
	return suffix
}

func optimizerKernelName(suffix string) string {
	return suffix + "_step"
}

func predictiveCodingKernelName(suffix string) string {
	return "pc_" + suffix
}

func markovBlanketKernelName(suffix string) string {
	return "markov_" + suffix
}

func hawkesKernelName(suffix string) string {
	return "hawkes_" + suffix
}

func vsaKernelName(suffix string) string {
	return "vsa_" + suffix
}
