//go:build darwin && cgo

package activation

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
	cpumath "github.com/theapemachine/puter/device/cpu/math"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func neonStyleExp32(value float32) float32 {
	scaled := value * float32(1.4426950408889634)
	roundedK := float32(math.RoundToEven(float64(scaled)))
	fraction := value - roundedK*float32(0.6931471805599453)
	poly := float32(0.00019841270)
	poly = fraction*poly + 0.0013888889
	poly = fraction*poly + 0.008333334
	poly = fraction*poly + 0.041666667
	poly = fraction*poly + 0.16666667
	poly = fraction*poly + 0.5
	poly = fraction*poly + 1.0
	poly = fraction*poly + 1.0
	exponentInt := int32(roundedK)
	scaleBits := uint32(exponentInt+127) << 23

	return math.Float32frombits(scaleBits) * poly
}

func geluInner(value float32) float32 {
	valueCubed := value * value * value
	innerArg := valueCubed*cpumath.GeluTanhBeta + value

	return float32(cpumath.GeluTanhAlpha * float64(innerArg))
}

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

	for _, lane := range []int{1, 26, 41} {
		inner := geluInner(source[lane])
		expTwo := neonStyleExp32(2 * inner)
		tanhHorner := (expTwo - 1) / (expTwo + 1)
		tanhPrecise := float32(math.Tanh(float64(inner)))
		t.Logf(
			"lane %d source=%.9g inner=%.9g exp2=%.9g tanhHorner=%.9g tanhPrecise=%.9g cpu=%.9g generic=%.9g fast=%.9g metal=%.9g want=%.9g cpuUlp=%d genericUlp=%d fastUlp=%d metalUlp=%d",
			lane,
			source[lane],
			inner,
			expTwo,
			tanhHorner,
			tanhPrecise,
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
