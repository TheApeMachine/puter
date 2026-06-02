//go:build darwin && cgo

package activation

import (
	"math/rand"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func geluInnerValue(value float32) float32 {
	valueCubed := value * value * value
	innerArg := valueCubed*cpumath.GeluTanhBeta + value

	return float32(cpumath.GeluTanhAlpha * float64(innerArg))
}

func TestGeluTanhLane5610Probe(t *testing.T) {
	count := 8192
	rng := rand.New(rand.NewSource(0x4D00 + int64(count)))
	source := make([]float32, count)

	for index := range source {
		source[index] = rng.Float32()*4 - 2
	}

	for _, laneIndex := range []int{377, 5610, 3621} {
		t.Logf(
			"lane %d source=%.9g inner=%.9g",
			laneIndex,
			source[laneIndex],
			geluInnerValue(source[laneIndex]),
		)
	}

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
	want := parity.ComputeUnaryReferenceBytes(
		source,
		dtype.Float32,
		parity.ReferenceGeluTanh(dtype.Float32),
	)
	wantValues := parity.DecodeFloat32Vector(want, dtype.Float32)

	for _, laneIndex := range []int{377, 5610, 3621} {
		t.Logf(
			"lane %d got=%.9g want=%.9g",
			laneIndex,
			got[laneIndex],
			wantValues[laneIndex],
		)
	}
}
