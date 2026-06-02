//go:build darwin && cgo

package activation

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGeluTanhMetalF32Parity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	for _, count := range parity.Lengths {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			source := parity.RandomUnaryInput(count, 0x4D00+int64(count))
			wantBytes := parity.ComputeUnaryReferenceBytes(
				source,
				dtype.Float32,
				parity.ReferenceGeluTanh(dtype.Float32),
			)

			sourceTensor := harness.UploadVector(source, dtype.Float32)
			destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
			defer sourceTensor.Close()
			defer destinationTensor.Close()

			if err := DispatchStandardUnaryRefs(
				harness.ContextRef(),
				destinationTensor.Ref(),
				sourceTensor.Ref(),
				dtype.Float32,
				StandardGeluTanh,
				uint32(count),
			); err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			got := harness.DownloadFloat32(destinationTensor, dtype.Float32)
			want := parity.DecodeFloat32Vector(wantBytes, dtype.Float32)
			parity.AssertFloat32SlicesWithinULP(t, got, want, 8)
		})
	}
}

func BenchmarkGeluTanhMetalF32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	count := 8192
	source := make([]float32, count)
	rng := rand.New(rand.NewSource(0x4D00 + int64(count)))

	for index := range source {
		source[index] = rng.Float32()*4 - 2
	}

	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	for b.Loop() {
		if err := DispatchStandardUnaryRefs(
			harness.ContextRef(),
			destinationTensor.Ref(),
			sourceTensor.Ref(),
			dtype.Float32,
			StandardGeluTanh,
			uint32(count),
		); err != nil {
			b.Fatal(err)
		}
	}
}
