//go:build xla

package embedding_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuembedding "github.com/theapemachine/puter/device/cpu/embedding"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceEmbedding = cpuembedding.New()

func randomEmbeddingTable(vocab, hidden int, seed int64) []float32 {
	table := make([]float32, vocab*hidden)
	state := uint64(seed)

	for index := range table {
		state = state*6364136223846793005 + 1
		table[index] = float32(int32(state>>33)%1000) / 1000.0
	}

	return table
}

func randomEmbeddingIndices(indexCount, vocab int, seed int64) []int32 {
	indices := make([]int32, indexCount)
	state := uint64(seed)

	for index := range indices {
		state = state*6364136223846793005 + 1
		indices[index] = int32(state % uint64(vocab))
	}

	return indices
}

func TestEmbeddingXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA embedding lookup", t, func() {
		const vocab = 128

		for _, hidden := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("hidden=%d", hidden), func() {
				for _, indexCount := range xlaparity.Lengths {
					convey.Convey(fmt.Sprintf("indexCount=%d", indexCount), func() {
						table := randomEmbeddingTable(vocab, hidden, 0xE220+int64(hidden)+int64(indexCount))
						indices := randomEmbeddingIndices(indexCount, vocab, 0xE221+int64(hidden))
						want := make([]float32, indexCount*hidden)
						referenceEmbedding.Lookup(
							unsafe.Pointer(&table[0]),
							unsafe.Pointer(&indices[0]),
							unsafe.Pointer(&want[0]),
							vocab, hidden, indexCount,
							dtype.Float32,
						)

						tableTensor := harness.UploadMatrix(table, vocab, hidden, dtype.Float32)
						indicesTensor := harness.UploadInt32Vector(indices)
						outputTensor := harness.UploadMatrix(
							make([]float32, indexCount*hidden), indexCount, hidden, dtype.Float32,
						)
						defer tableTensor.Close()
						defer indicesTensor.Close()
						defer outputTensor.Close()

						harness.Backend().Lookup(
							xla.ResidentPointer(tableTensor),
							xla.ResidentPointer(indicesTensor),
							xla.ResidentPointer(outputTensor),
							vocab, hidden, indexCount,
							dtype.Float32,
						)

						got := harness.DownloadFloat32(outputTensor, dtype.Float32)
						xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 0)
					})
				}
			})
		}
	})

	convey.Convey("Given XLA embedding bag", t, func() {
		const vocab = 64
		const bagCount = 3
		const indexCount = 11

		for _, hidden := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("hidden=%d", hidden), func() {
				table := randomEmbeddingTable(vocab, hidden, 0xE230+int64(hidden))
				indices := randomEmbeddingIndices(indexCount, vocab, 0xE231+int64(hidden))
				offsets := []int32{0, 4, 7}
				want := make([]float32, bagCount*hidden)
				referenceEmbedding.Bag(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&offsets[0]),
					unsafe.Pointer(&want[0]),
					vocab, hidden, bagCount, indexCount,
					dtype.Float32,
				)

				tableTensor := harness.UploadMatrix(table, vocab, hidden, dtype.Float32)
				indicesTensor := harness.UploadInt32Vector(indices)
				offsetsTensor := harness.UploadInt32Vector(offsets)
				outputTensor := harness.UploadMatrix(
					make([]float32, bagCount*hidden), bagCount, hidden, dtype.Float32,
				)
				defer tableTensor.Close()
				defer indicesTensor.Close()
				defer offsetsTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Bag(
					xla.ResidentPointer(tableTensor),
					xla.ResidentPointer(indicesTensor),
					xla.ResidentPointer(offsetsTensor),
					xla.ResidentPointer(outputTensor),
					vocab, hidden, bagCount, indexCount,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}
