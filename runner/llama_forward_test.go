//go:build darwin && cgo

package runner

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	hfconfig "github.com/theapemachine/hf/config"
	"github.com/theapemachine/hf/tokenizer"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/expand"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/lower"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/weights"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/qpool"
)

func TestLlamaHelloForwardTopLogits(testingObject *testing.T) {
	weightPath := llama32InstructWeightPath(testingObject)
	tokenizerPath := llama32InstructTokenizerPath(testingObject)

	convey.Convey("Given a Llama 3.2 instruct hello prompt", testingObject, func() {
		artifact, err := tokenizer.Read(tokenizerPath)
		convey.So(err, convey.ShouldBeNil)

		metadata, err := tokenizer.LoadMetadata(
			context.Background(),
			tokenizer.Source{Source: llama32SnapshotDir(testingObject)},
		)
		convey.So(err, convey.ShouldBeNil)

		prompt, err := metadata.ApplyChatTemplate("Hello")
		convey.So(err, convey.ShouldBeNil)

		tokenIDs, err := artifact.Tokenizer.Encode(prompt)
		convey.So(err, convey.ShouldBeNil)

		manifestGraph, computeGraph, err := compileLlamaGraph(testingObject)
		convey.So(err, convey.ShouldBeNil)

		manifestGraph.Metadata = map[string]any{"weights_path": weightPath}
		bindManifestWeights(testingObject, manifestGraph, weightPath)

		workerPool := qpool.NewQ(context.Background(), 1, 2, qpool.NewConfig())
		devicePool, err := pool.New(context.Background(), workerPool)
		convey.So(err, convey.ShouldBeNil)
		defer devicePool.Close()

		result, err := New(devicePool).CallGraph(context.Background(), runtime.GraphCallRequest{
			GraphName: "model",
			Graph:     manifestGraph,
			Compute:   computeGraph,
			Inputs: map[string]any{
				"input_ids": tokenIDs,
			},
		})
		convey.So(err, convey.ShouldBeNil)

		logits, ok := result.Outputs["logits"].([]float32)
		convey.So(ok, convey.ShouldBeTrue)

		vocabSize := 128256
		lastLogits := logits[len(logits)-vocabSize:]
		top := topLogitIndices(lastLogits, 5)

		convey.Convey("It should rank a natural greeting continuation highest", func() {
			for _, entry := range top {
				text, _ := artifact.Tokenizer.Decode([]int{entry.index}, false)
				testingObject.Logf("top token %d (%q) logit=%f", entry.index, text, entry.value)
			}

			firstText, err := artifact.Tokenizer.Decode([]int{top[0].index}, false)
			convey.So(err, convey.ShouldBeNil)
			convey.So(firstText, convey.ShouldEqual, "How")
		})

		convey.Convey("Greedy decode steps should stay coherent", func() {
			sequence := append([]int(nil), tokenIDs...)
			graphRunner := New(devicePool)
			pieces := make([]string, 0, 3)

			for step := 0; step < 3; step++ {
				stepResult, stepErr := graphRunner.CallGraph(context.Background(), runtime.GraphCallRequest{
					GraphName: "model",
					Graph:     manifestGraph,
					Compute:   computeGraph,
					Inputs: map[string]any{
						"input_ids": sequence,
					},
				})
				convey.So(stepErr, convey.ShouldBeNil)

				stepLogits, ok := stepResult.Outputs["logits"].([]float32)
				convey.So(ok, convey.ShouldBeTrue)

				last := stepLogits[len(stepLogits)-vocabSize:]
				nextToken := topLogitIndices(last, 1)[0].index
				nextText, decodeErr := artifact.Tokenizer.Decode([]int{nextToken}, false)
				convey.So(decodeErr, convey.ShouldBeNil)

				pieces = append(pieces, nextText)
				sequence = append(sequence, nextToken)
			}

			convey.So(pieces, convey.ShouldResemble, []string{"How", " can", " I"})
		})
	})
}

func compileLlamaGraph(testingObject *testing.T) (*ast.Graph, *ir.Graph, error) {
	testingObject.Helper()

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

	if err != nil {
		return nil, nil, err
	}

	block, err := parse.BlockModelFromYAML([]byte(yamlText))

	if err != nil {
		return nil, nil, err
	}

	topology, err := block.TopologyAST()

	if err != nil {
		return nil, nil, err
	}

	topology, err = expand.NewRecipe(nil).ExpandTopology(topology)

	if err != nil {
		return nil, nil, err
	}

	manifestGraph, err := lower.NewLowerer().Topology(topology, dtype.BFloat16)

	if err != nil {
		return nil, nil, err
	}

	computeGraph, err := ir.NewLowerer().Graph(manifestGraph)

	return manifestGraph, computeGraph, err
}

func bindManifestWeights(testingObject *testing.T, manifestGraph *ast.Graph, weightPath string) {
	testingObject.Helper()

	file, err := os.Open(weightPath)

	if err != nil {
		testingObject.Fatal(err)
	}

	defer file.Close()

	index, err := weights.NewBinder().Index(file)

	if err != nil {
		testingObject.Fatal(err)
	}

	if err := weights.NewBinder().Bind(manifestGraph, index, nil); err != nil {
		testingObject.Fatal(err)
	}
}

type logitEntry struct {
	index int
	value float32
}

func topLogitIndices(logits []float32, count int) []logitEntry {
	entries := make([]logitEntry, len(logits))

	for index, value := range logits {
		entries[index] = logitEntry{index: index, value: value}
	}

	for left := 0; left < count && left < len(entries); left++ {
		best := left

		for right := left + 1; right < len(entries); right++ {
			if entries[right].value > entries[best].value {
				best = right
			}
		}

		entries[left], entries[best] = entries[best], entries[left]
	}

	if count > len(entries) {
		count = len(entries)
	}

	return entries[:count]
}

func llama32InstructTokenizerPath(testingObject *testing.T) string {
	testingObject.Helper()

	return filepath.Join(llama32SnapshotDir(testingObject), "tokenizer.json")
}

func llama32SnapshotDir(testingObject *testing.T) string {
	testingObject.Helper()

	home, err := os.UserHomeDir()

	if err != nil {
		testingObject.Skip("home directory is unavailable")
	}

	repoDir := filepath.Join(
		home,
		".cache",
		"huggingface",
		"hub",
		"models--meta-llama--Llama-3.2-1B-Instruct",
		"snapshots",
	)

	entries, err := os.ReadDir(repoDir)

	if err != nil {
		testingObject.Skip("Llama 3.2 instruct snapshot is not cached")
	}

	for _, entry := range entries {
		if entry.IsDir() {
			return filepath.Join(repoDir, entry.Name())
		}
	}

	testingObject.Skip("Llama 3.2 instruct snapshot is not cached")
	return ""
}
