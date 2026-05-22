package runner

import (
	"strings"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	hfconfig "github.com/theapemachine/hf/config"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/expand"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/lower"
	"github.com/theapemachine/manifesto/parse"
)

func TestLlamaCompiledGraphRoPEAttributes(testingObject *testing.T) {
	yamlText, err := hfconfig.GenerateYAML(&hfconfig.Config{
		Architectures:     []string{"LlamaForCausalLM"},
		ModelType:         "llama",
		VocabSize:         128256,
		HiddenSize:        2048,
		IntermediateSize:  8192,
		NumHiddenLayers:   16,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		RMSNormEps:        1e-5,
		RopeTheta:         500000,
		TieWordEmbeddings: true,
	}, "meta-llama/Llama-3.2-1B-Instruct")

	convey.Convey("Given a generated Llama topology", testingObject, func() {
		convey.So(err, convey.ShouldBeNil)

		block, parseErr := parse.BlockModelFromYAML([]byte(yamlText))
		convey.So(parseErr, convey.ShouldBeNil)

		topology, topologyErr := block.TopologyAST()
		convey.So(topologyErr, convey.ShouldBeNil)

		topology, expandErr := expand.NewRecipe(nil).ExpandTopology(topology)
		convey.So(expandErr, convey.ShouldBeNil)

		manifestGraph, lowerErr := lower.NewLowerer().Topology(topology, dtype.BFloat16)
		convey.So(lowerErr, convey.ShouldBeNil)

		computeGraph, irErr := ir.NewLowerer().Graph(manifestGraph)
		convey.So(irErr, convey.ShouldBeNil)

		ropeNode := findComputeNodeByPrefix(computeGraph, "rope_q_0")
		convey.So(ropeNode, convey.ShouldNotBeNil)

		config := ropeConfigFromNode(ropeNode)

		convey.Convey("RoPE nodes should preserve Llama 3 config in IR attributes", func() {
			baseAttr := ropeNode.Attribute("base")
			convey.So(baseAttr.Kind, convey.ShouldEqual, ir.AttributeInt)
			convey.So(baseAttr.Value, convey.ShouldEqual, "500000")
			convey.So(config.Base, convey.ShouldEqual, float32(500000))
			convey.So(config.Type, convey.ShouldEqual, "llama3")
			convey.So(config.Factor, convey.ShouldEqual, float32(32))
			convey.So(config.OriginalContext, convey.ShouldEqual, uint32(8192))
		})

		normNode := findComputeNodeByPrefix(computeGraph, "input_layernorm_0")
		convey.So(normNode, convey.ShouldNotBeNil)

		convey.Convey("RMSNorm nodes should preserve eps in IR attributes", func() {
			convey.So(rmsNormEpsilonFromNode(normNode), convey.ShouldEqual, float32(1e-5))
		})
	})
}

func findComputeNodeByPrefix(computeGraph *ir.Graph, prefix string) *ir.Node {
	for _, node := range computeGraph.Nodes() {
		if strings.HasPrefix(node.ID(), prefix) {
			return node
		}
	}

	return nil
}
