//go:build darwin && cgo

package parity

import (
	"math"
	"testing"

	cpumath "github.com/theapemachine/puter/device/cpu/math"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

const (
	sf64ProbeLog = iota
	sf64ProbeSqrt
	sf64ProbeSin
	sf64ProbeCos
	sf64ProbeGeluInner
	sf64ProbeInvStdDev
	sf64ProbeBoxMagnitude
	sf64ProbeGaussianCos
	sf64ProbeGaussianSin
	sf64ProbeGeluProduct
)

type sf64ProbeCase struct {
	name          string
	uniformFirst  float32
	uniformSecond float32
	geluValue     float32
	sqrtInput64   uint64
}

func TestSF64TranscendentalProbeMatchesGoMath(t *testing.T) {
	harness := NewHarness(t)
	defer harness.Close()

	cases := []sf64ProbeCase{
		{
			name:          "philox_lane0_uniforms",
			uniformFirst:  0.39904642105102539,
			uniformSecond: 0.88052010536193848,
			geluValue:     1.9228818416595459,
			sqrtInput64:   math.Float64bits(0.001 + 1e-5),
		},
		{
			name:          "random_lane149_uniforms",
			uniformFirst:  0,
			uniformSecond: 0,
			geluValue:     -0.098532654,
			sqrtInput64:   math.Float64bits(0.0025 + 1e-5),
		},
		{
			name:          "gelu_tanh_repro",
			uniformFirst:  0.5,
			uniformSecond: 0.25,
			geluValue:     1.9228818416595459,
			sqrtInput64:   math.Float64bits(1e-5),
		},
		{
			name:          "layernorm_variance_band",
			uniformFirst:  0.25,
			uniformSecond: 0.125,
			geluValue:     0.026889682,
			sqrtInput64:   math.Float64bits(0.001548387 + 1e-5),
		},
	}

	first149, second149 := philoxUniformPair(149)
	cases[1].uniformFirst = first149
	cases[1].uniformSecond = second149

	inputs := make([]float32, 0, len(cases)*MetalSF64ProbeInputFloats)
	sqrtInputs := make([]uint64, 0, len(cases))

	for _, probeCase := range cases {
		inputs = append(
			inputs,
			probeCase.uniformFirst,
			probeCase.uniformSecond,
			probeCase.geluValue,
			0,
		)
		sqrtInputs = append(sqrtInputs, probeCase.sqrtInput64)
	}

	outputs, err := harness.DispatchSF64TranscendentalProbe(inputs, sqrtInputs)

	if err != nil {
		t.Fatalf("dispatch sf64 probe: %v", err)
	}

	for caseIndex, probeCase := range cases {
		base := caseIndex * MetalSF64ProbeOutputWords
		got := outputs[base : base+MetalSF64ProbeOutputWords]
		want := referenceSF64ProbeOutputs(probeCase)

		assertF64BitsEqual(t, probeCase.name+"/log", got[sf64ProbeLog], want[sf64ProbeLog])
		assertF64WithinULP(t, probeCase.name+"/sqrt", got[sf64ProbeSqrt], want[sf64ProbeSqrt], 1)
		assertF64WithinULP(t, probeCase.name+"/sin", got[sf64ProbeSin], want[sf64ProbeSin], 1)
		assertF64WithinULP(t, probeCase.name+"/cos", got[sf64ProbeCos], want[sf64ProbeCos], 1)
		assertF64BitsEqual(t, probeCase.name+"/gelu_inner", got[sf64ProbeGeluInner], want[sf64ProbeGeluInner])
		assertF64WithinULP(t, probeCase.name+"/inv_std_dev", got[sf64ProbeInvStdDev], want[sf64ProbeInvStdDev], 1)
		assertF64WithinULP(t, probeCase.name+"/box_magnitude", got[sf64ProbeBoxMagnitude], want[sf64ProbeBoxMagnitude], 1)
		assertF64BitsEqual(t, probeCase.name+"/gelu_product", got[sf64ProbeGeluProduct], want[sf64ProbeGeluProduct])
		assertStoredF32WithinULP(t, probeCase.name+"/gaussian_cos", got[sf64ProbeGaussianCos], want[sf64ProbeGaussianCos], 128)
		assertStoredF32WithinULP(t, probeCase.name+"/gaussian_sin", got[sf64ProbeGaussianSin], want[sf64ProbeGaussianSin], 128)
	}
}

func referenceSF64ProbeOutputs(probeCase sf64ProbeCase) []uint64 {
	uniformFirst := probeCase.uniformFirst

	if uniformFirst == 0 {
		uniformFirst = math.Float32frombits(0x34000000)
	}

	uniformFirst64 := float64(uniformFirst)
	logValue := math.Log(uniformFirst64)
	sqrtValue := math.Sqrt(math.Float64frombits(probeCase.sqrtInput64))
	angle := 2.0 * math.Pi * float64(probeCase.uniformSecond)
	sinValue, cosValue := math.Sincos(angle)

	geluValue64 := float64(probeCase.geluValue)
	geluInner := cpumath.GeluTanhAlpha * (geluValue64 + cpumath.GeluTanhBeta*geluValue64*geluValue64*geluValue64)
	invStdDev := 1.0 / sqrtValue

	magnitude := math.Sqrt(-2.0 * logValue)
	gaussianCos := float32(magnitude * cosValue)
	gaussianSin := float32(magnitude * sinValue)

	innerF32 := float32(geluInner)
	geluTanh := cpumath.FastTanh32(innerF32)
	geluProduct := 0.5 * geluValue64 * (1.0 + float64(geluTanh))

	return []uint64{
		math.Float64bits(logValue),
		math.Float64bits(sqrtValue),
		math.Float64bits(sinValue),
		math.Float64bits(cosValue),
		math.Float64bits(geluInner),
		math.Float64bits(invStdDev),
		math.Float64bits(magnitude),
		math.Float64bits(float64(gaussianCos)),
		math.Float64bits(float64(gaussianSin)),
		math.Float64bits(geluProduct),
	}
}

func assertStoredF32WithinULP(testingObject *testing.T, label string, got, want uint64, maxULP int) {
	testingObject.Helper()

	gotValue := float32(math.Float64frombits(got))
	wantValue := float32(math.Float64frombits(want))
	distance := cpuparity.Float32ULPDistance(gotValue, wantValue)

	if distance <= maxULP {
		return
	}

	testingObject.Fatalf(
		"%s f32 ULP=%d max=%d got=%g want=%g",
		label,
		distance,
		maxULP,
		gotValue,
		wantValue,
	)
}

func assertF64BitsEqual(testingObject *testing.T, label string, got, want uint64) {
	testingObject.Helper()

	if got == want {
		return
	}

	testingObject.Fatalf(
		"%s f64 bits got=%016x (%g) want=%016x (%g)",
		label,
		got,
		math.Float64frombits(got),
		want,
		math.Float64frombits(want),
	)
}

func assertF64WithinULP(testingObject *testing.T, label string, got, want uint64, maxULP int) {
	testingObject.Helper()

	distance := float64ULPDistance(math.Float64frombits(got), math.Float64frombits(want))

	if distance <= maxULP {
		return
	}

	testingObject.Fatalf(
		"%s f64 ULP=%d max=%d got=%016x (%g) want=%016x (%g)",
		label,
		distance,
		maxULP,
		got,
		math.Float64frombits(got),
		want,
		math.Float64frombits(want),
	)
}

func float64ULPDistance(left, right float64) int {
	leftBits := float64BitsOrdered(left)
	rightBits := float64BitsOrdered(right)

	if leftBits > rightBits {
		leftBits, rightBits = rightBits, leftBits
	}

	return int(rightBits - leftBits)
}

func float64BitsOrdered(value float64) uint64 {
	bits := math.Float64bits(value)

	const signBit = uint64(1) << 63

	if bits&signBit != 0 {
		return signBit - bits
	}

	return bits
}

func philoxUniformPair(laneIndex int) (float32, float32) {
	threadIndex := uint32(laneIndex / 4)
	pairIndex := laneIndex % 4

	w0, w1, w2, w3 := philoxWords(0, 0, threadIndex, 0)
	words := []uint32{w0, w1, w2, w3}

	switch pairIndex {
	case 0, 1:
		return uniformFromBits(words[0]), uniformFromBits(words[1])
	default:
		return uniformFromBits(words[2]), uniformFromBits(words[3])
	}
}

func uniformFromBits(bits uint32) float32 {
	const oneAsBits = uint32(0x3F800000)
	mantissa := bits >> 9

	return math.Float32frombits(oneAsBits|mantissa) - 1.0
}

func philoxWords(seedLo, seedHi, ctrLo, ctrHi uint32) (uint32, uint32, uint32, uint32) {
	const (
		philoxM0 = uint32(0xD2511F53)
		philoxM1 = uint32(0xCD9E8D57)
		philoxW0 = uint32(0x9E3779B9)
		philoxW1 = uint32(0xBB67AE85)
	)

	c0 := ctrLo
	c1 := ctrHi
	c2 := uint32(0)
	c3 := uint32(0)
	key0 := seedLo
	key1 := seedHi

	for round := 0; round < 10; round++ {
		product0 := uint64(philoxM0) * uint64(c0)
		product1 := uint64(philoxM1) * uint64(c2)
		hi0 := uint32(product0 >> 32)
		lo0 := uint32(product0)
		hi1 := uint32(product1 >> 32)
		lo1 := uint32(product1)

		newC0 := hi1 ^ c1 ^ key0
		newC2 := hi0 ^ c3 ^ key1
		c0 = newC0
		c1 = lo1
		c2 = newC2
		c3 = lo0
		key0 += philoxW0
		key1 += philoxW1
	}

	return c0, c1, c2, c3
}
