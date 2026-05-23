//go:build cuda

package embedding

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func lookupReference(table []float32, indices []uint32, vocab, hidden uint32) []float32 {
	output := make([]float32, len(indices)*hidden)

	for indexPosition, vocabIndex := range indices {
		for hiddenIndex := uint32(0); hiddenIndex < hidden; hiddenIndex++ {
			tableOffset := vocabIndex*hidden + hiddenIndex
			output[indexPosition*hidden+hiddenIndex] = table[tableOffset]
		}
	}

	return output
}

func TestEmbeddingCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA embedding lookup", t, func() {
		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				vocab := uint32(16)
				hidden := uint32(8)
				table := parity.RandomUnaryInput(int(vocab*hidden), 0xE100+int64(count))
				indices := make([]uint32, count)

				for index := range indices {
					indices[index] = uint32(index % int(vocab))
				}

				want := lookupReference(table, indices, vocab, hidden)
				indicesBytes := uint32SliceToBytes(indices)

				tableTensor := harness.UploadVector(table, dtype.Float32)
				indicesTensor := harness.UploadBytes(indicesBytes)
				outputTensor := harness.UploadVector(make([]float32, len(want)), dtype.Float32)
				errorFlagTensor := harness.UploadBytes(make([]byte, 4))
				defer tableTensor.Close()
				defer indicesTensor.Close()
				defer outputTensor.Close()
				defer errorFlagTensor.Close()

				if err := DispatchLookupRefs(
					harness.ContextRef(),
					tableTensor.Ref(),
					indicesTensor.Ref(),
					outputTensor.Ref(),
					errorFlagTensor.Ref(),
					dtype.Float32,
					vocab,
					hidden,
					uint32(count),
				); err != nil {
					t.Fatalf("dispatch Lookup: %v", err)
				}

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func uint32SliceToBytes(values []uint32) []byte {
	bytesOut := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(bytesOut[index*4:], value)
	}

	return bytesOut
}
