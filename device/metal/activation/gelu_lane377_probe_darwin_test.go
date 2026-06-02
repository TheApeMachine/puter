//go:build darwin && cgo

package activation

import (
	"math/rand"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGeluTanhLane377N1024Probe(t *testing.T) {
	count := 1024
	rng := rand.New(rand.NewSource(0x4D00 + int64(count)))
	source := make([]float32, count)

	for index := range source {
		source[index] = rng.Float32()*4 - 2
	}

	want := parity.ComputeUnaryReferenceBytes(
		source,
		dtype.Float32,
		parity.ReferenceGeluTanh(dtype.Float32),
	)
	wantValues := parity.DecodeFloat32Vector(want, dtype.Float32)

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

	laneIndex := 377
	t.Logf(
		"lane %d source=%.9g got=%.9g want=%.9g ulp=%d",
		laneIndex,
		source[laneIndex],
		got[laneIndex],
		wantValues[laneIndex],
		cpuparity.Float32ULPDistance(got[laneIndex], wantValues[laneIndex]),
	)
}
