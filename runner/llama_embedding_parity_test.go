//go:build darwin && cgo

package runner

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/weights"
	"github.com/theapemachine/puter/kernels"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/qpool"
)

func TestLlamaEmbeddingLookupParity(testingObject *testing.T) {
	weightPath := llama32InstructWeightPath(testingObject)
	tokenID := 128000

	convey.Convey("Given Llama 3.2 instruct embed_tokens weights", testingObject, func() {
		raw, meta, err := weights.ReadTensor(weightPath, "model.embed_tokens.weight")
		convey.So(err, convey.ShouldBeNil)
		convey.So(meta.DType, convey.ShouldEqual, "BF16")

		storageDType, err := dtype.Parse(meta.DType)
		convey.So(err, convey.ShouldBeNil)

		hiddenSize := int(meta.Shape[1])
		rowRaw := weightRowBytes(raw, hiddenSize*2, tokenID)
		expected, err := convert.BytesToFloat32(storageDType, rowRaw)
		convey.So(err, convey.ShouldBeNil)

		memory := newMetalMemoryForRunnerTest(testingObject)

		tableShape, err := tensor.NewShape([]int{int(meta.Shape[0]), hiddenSize})
		convey.So(err, convey.ShouldBeNil)

		table, err := memory.Upload(tableShape, storageDType, raw)
		convey.So(err, convey.ShouldBeNil)
		defer table.Close()

		indices, err := uploadInt32Indices(memory, []int{tokenID})
		convey.So(err, convey.ShouldBeNil)
		defer indices.Close()

		outShape, err := tensor.NewShape([]int{1, 1, hiddenSize})
		convey.So(err, convey.ShouldBeNil)

		outBytes, err := storageDType.BytesFor(outShape.Len())
		convey.So(err, convey.ShouldBeNil)

		out, err := memory.Upload(outShape, storageDType, make([]byte, outBytes))
		convey.So(err, convey.ShouldBeNil)
		defer out.Close()

		kernel, ok := kernels.Default.LookupLocation(
			"embedding_lookup",
			kernels.Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{storageDType, dtype.Int32},
				Outputs: []dtype.DType{storageDType},
			},
			tensor.Metal,
		)
		convey.So(ok, convey.ShouldBeTrue)

		convey.Convey("Metal embedding lookup should match the weight row", func() {
			err := kernel.Run(table, indices, out)
			convey.So(err, convey.ShouldBeNil)

			got, err := downloadFloat32Vector(memory, out)
			convey.So(err, convey.ShouldBeNil)
			convey.So(len(got), convey.ShouldEqual, hiddenSize)

			maxULP := float32(0)

			for index := range got {
				distance := bf16ULPDistance(got[index], expected[index])

				if distance > maxULP {
					maxULP = distance
				}
			}

			convey.So(maxULP, convey.ShouldBeLessThanOrEqualTo, 1)
		})
	})
}

func llama32InstructWeightPath(testingObject *testing.T) string {
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
		testingObject.Skip("Llama 3.2 instruct weights are not cached")
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		candidate := filepath.Join(repoDir, entry.Name(), "model.safetensors")

		if _, statErr := os.Stat(candidate); statErr == nil {
			return candidate
		}
	}

	testingObject.Skip("Llama 3.2 instruct weights are not cached")
	return ""
}

func newMetalMemoryForRunnerTest(testingObject *testing.T) tensor.Backend {
	testingObject.Helper()

	workerPool := qpool.NewQ(context.Background(), 1, 2, qpool.NewConfig())
	devicePool, err := pool.New(context.Background(), workerPool)

	if err != nil {
		testingObject.Fatal(err)
	}

	memory, _, err := devicePool.ComputeMemory()

	if err != nil {
		testingObject.Fatal(err)
	}

	return memory
}

func weightRowBytes(raw []byte, rowBytes int, rowIndex int) []byte {
	start := rowIndex * rowBytes
	end := start + rowBytes

	return raw[start:end]
}

func bf16ULPDistance(got float32, want float32) float32 {
	if got == want {
		return 0
	}

	const ulpScale = float32(1 << 16)

	diff := got - want

	if diff < 0 {
		diff = -diff
	}

	return diff * ulpScale
}
