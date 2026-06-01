//go:build darwin && cgo

package activation

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGeluTanhLane26Probe(t *testing.T) {
	count := 64
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

	cpuOut := make([]float32, count)
	cpuactivation.New().GeluTanh(
		unsafe.Pointer(&cpuOut[0]),
		unsafe.Pointer(&source[0]),
		count,
		dtype.Float32,
	)

	genericOut := make([]float32, count)
	cpuactivation.GeluTanhF32Generic(&genericOut[0], &source[0], count)

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

	for _, lane := range []int{24, 25, 26, 27} {
		t.Logf(
			"lane %d source=%.9g cpu=%.9g generic=%.9g fast=%.9g metal=%.9g want=%.9g cpuUlp=%d genericUlp=%d fastUlp=%d metalUlp=%d",
			lane,
			source[lane],
			cpuOut[lane],
			genericOut[lane],
			cpumath.FastGeluTanh32(source[lane]),
			got[lane],
			want[lane],
			cpuparity.Float32ULPDistance(cpuOut[lane], want[lane]),
			cpuparity.Float32ULPDistance(genericOut[lane], want[lane]),
			cpuparity.Float32ULPDistance(cpumath.FastGeluTanh32(source[lane]), want[lane]),
			cpuparity.Float32ULPDistance(got[lane], want[lane]),
		)
	}
}
