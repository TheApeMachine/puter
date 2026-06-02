//go:build darwin && cgo

package activation

import (
	"math/rand"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGeluTanhLane80N1024Probe(t *testing.T) {
	count := 1024
	rng := rand.New(rand.NewSource(0x4D00 + int64(count)))
	source := make([]float32, count)

	for index := range source {
		source[index] = rng.Float32()*4 - 2
	}

	wantBytes := parity.ComputeUnaryReferenceBytes(
		source,
		dtype.Float32,
		parity.ReferenceGeluTanh(dtype.Float32),
	)
	want := parity.DecodeFloat32Vector(wantBytes, dtype.Float32)

	harness := parity.NewHarness(t)
	defer harness.Close()

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

	laneIndex := 80
	inner := geluInner(source[laneIndex])

	t.Logf(
		"lane %d source=%.9g inner=%.9g want=%.9g metal=%.9g metalUlp=%d",
		laneIndex,
		source[laneIndex],
		inner,
		want[laneIndex],
		got[laneIndex],
		cpuparity.Float32ULPDistance(got[laneIndex], want[laneIndex]),
	)
}
