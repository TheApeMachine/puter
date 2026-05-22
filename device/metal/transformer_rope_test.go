//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func TestKernelRegistry_MetalRoPEDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalRoPEDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalRoPEDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, headDimCase := range parityElementCounts {
		headDimCase := headDimCase

		testingObject.Run(testNameForElementCount(headDimCase), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" RoPE tensors", testingObject, func() {
				runRoPEParityCase(testingObject, backend, storageDType, headDimCase)
			})
		})
	}
}

func runRoPEParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	headDimCase int,
) {
	seqLen, numHeads, headDim := 2, 3, ropeHeadDimForTest(headDimCase)
	fixture := ropeFixtureForTest(seqLen, numHeads, headDim, storageDType)
	input, out := ropeTensorsForTest(
		testingObject, backend, seqLen, numHeads, headDim, storageDType, fixture,
	)
	defer closeBenchmarkTensors(input, out)

	err := lookupRoPEKernel(testingObject, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertRoPEBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func TestMetalFlux2RoPE(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given text and latent tokens with FLUX2 4D RoPE IDs", t, func() {
		seqLen, numHeads, headDim := 6, 1, 128
		latentSeqLen, latentSide := 4, 2
		fixture := flux2RoPEFixtureForTest(seqLen, numHeads, headDim, latentSeqLen, latentSide)
		input, out := ropeTensorsForTest(
			t, backend, seqLen, numHeads, headDim, dtype.Float32, fixture,
		)
		defer closeBenchmarkTensors(input, out)

		convey.Convey("It should match HF axis-wise text and latent positions", func() {
			err := runMetalFlux2RoPE(input, out, latentSeqLen, latentSide, 2000)

			convey.So(err, convey.ShouldBeNil)
			assertRoPEBytesForTest(t, backend, out, dtype.Float32, fixture)
		})
	})
}

func flux2RoPEFixtureForTest(
	seqLen int,
	numHeads int,
	headDim int,
	latentSeqLen int,
	latentSide int,
) ropeFixture {
	inputBytes := encodeProjectionValuesAsDType(
		ropeInputValues(seqLen*numHeads*headDim), dtype.Float32,
	)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, dtype.Float32)
	expected := flux2RoPEExpectedValues(inputStored, seqLen, numHeads, headDim, latentSeqLen, latentSide)

	return ropeFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeProjectionValuesAsDType(expected, dtype.Float32),
		expectedFloat32: expected,
	}
}

func flux2RoPEExpectedValues(
	input []float32,
	seqLen int,
	numHeads int,
	headDim int,
	latentSeqLen int,
	latentSide int,
) []float32 {
	out := make([]float32, len(input))
	halfDim := headDim / 2

	for seqIndex := range seqLen {
		for headIndex := range numHeads {
			for pairIndex := range halfDim {
				flux2RoPEExpectedPair(
					input, out, seqIndex, headIndex, pairIndex,
					numHeads, headDim, latentSeqLen, latentSide,
				)
			}
		}
	}

	return out
}

func flux2RoPEExpectedPair(
	input []float32,
	out []float32,
	seqIndex int,
	headIndex int,
	pairIndex int,
	numHeads int,
	headDim int,
	latentSeqLen int,
	latentSide int,
) {
	inputIndex := (seqIndex*numHeads+headIndex)*headDim + pairIndex*2
	axisPairCount := headDim / 8
	axisIndex := pairIndex / axisPairCount
	localPair := pairIndex - axisIndex*axisPairCount
	textLen := seqLenFromValues(input, numHeads, headDim) - latentSeqLen
	position := flux2RoPEPosition(seqIndex, textLen, latentSide, axisIndex)
	axisDim := axisPairCount * 2
	exponent := -2 * float64(localPair) / float64(axisDim)
	theta := float64(position) * math.Pow(2000, exponent)
	cosTheta := float32(math.Cos(theta))
	sinTheta := float32(math.Sin(theta))
	even := input[inputIndex]
	odd := input[inputIndex+1]

	out[inputIndex] = even*cosTheta - odd*sinTheta
	out[inputIndex+1] = even*sinTheta + odd*cosTheta
}

func flux2RoPEPosition(seqIndex int, textLen int, latentSide int, axisIndex int) int {
	if seqIndex < textLen {
		if axisIndex == 3 {
			return seqIndex
		}

		return 0
	}

	imageIndex := seqIndex - textLen

	if axisIndex == 1 {
		return imageIndex / latentSide
	}

	if axisIndex == 2 {
		return imageIndex % latentSide
	}

	return 0
}

func seqLenFromValues(values []float32, numHeads int, headDim int) int {
	return len(values) / (numHeads * headDim)
}

func ropeHeadDimForTest(headDimCase int) int {
	if headDimCase <= 1 {
		return 2
	}

	if headDimCase%2 == 0 {
		return headDimCase
	}

	return headDimCase + 1
}

func lookupRoPEKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("rope", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s rope kernel", storageDType.Name())
	}

	return kernel
}

type ropeFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func ropeTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
	fixture ropeFixture,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{seqLen, numHeads, headDim})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)

	return input, out
}

func ropeFixtureForTest(
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
) ropeFixture {
	inputBytes := encodeProjectionValuesAsDType(
		ropeInputValues(seqLen*numHeads*headDim), storageDType,
	)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := ropeExpectedValues(inputStored, seqLen, numHeads, headDim)

	return ropeFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeProjectionValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func ropeInputValues(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		if index%2 == 0 {
			values[index] = 0.5 + float32((index*17+11)%31)/128
			continue
		}

		values[index] = 0.03125 + float32((index*19+7)%17)/512
	}

	return values
}

func ropeExpectedValues(
	input []float32,
	seqLen int,
	numHeads int,
	headDim int,
) []float32 {
	out := make([]float32, len(input))
	halfDim := headDim / 2

	for seqIndex := range seqLen {
		for headIndex := range numHeads {
			for pairIndex := range halfDim {
				ropeExpectedPair(input, out, seqIndex, headIndex, pairIndex, numHeads, headDim)
			}
		}
	}

	return out
}

func ropeExpectedPair(
	input []float32,
	out []float32,
	seqIndex int,
	headIndex int,
	pairIndex int,
	numHeads int,
	headDim int,
) {
	inputIndex := (seqIndex*numHeads+headIndex)*headDim + pairIndex*2
	exponent := -2 * float64(pairIndex) / float64(headDim)
	theta := float64(seqIndex) * math.Pow(10000, exponent)
	cosTheta := float32(math.Cos(theta))
	sinTheta := float32(math.Sin(theta))
	even := input[inputIndex]
	odd := input[inputIndex+1]

	out[inputIndex] = even*cosTheta - odd*sinTheta
	out[inputIndex+1] = even*sinTheta + odd*cosTheta
}

func TestMetalLlama3RoPE(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given Llama 3 scaled RoPE parameters", t, func() {
		seqLen, numHeads, headDim := 4, 2, 64
		fixture := llama3RoPEFixtureForTest(seqLen, numHeads, headDim, dtype.Float32)
		input, out := ropeTensorsForTest(
			t, backend, seqLen, numHeads, headDim, dtype.Float32, fixture,
		)
		defer closeBenchmarkTensors(input, out)

		convey.Convey("It should match llama3 frequency scaling", func() {
			config := RoPEConfig{
				Base:            500000,
				Type:            "llama3",
				Mode:            "half",
				Factor:          32,
				LowFreqFactor:   1,
				HighFreqFactor:  4,
				OriginalContext: 8192,
			}
			err := RunRoPE(input, out, config)

			convey.So(err, convey.ShouldBeNil)
			assertFloat32TensorForTest(t, backend, out, fixture.expectedFloat32, 1024)
		})
	})
}

func llama3RoPEFixtureForTest(
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
) ropeFixture {
	inputBytes := encodeProjectionValuesAsDType(
		ropeInputValues(seqLen*numHeads*headDim), storageDType,
	)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := llama3RoPEExpectedValues(inputStored, seqLen, numHeads, headDim)

	return ropeFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeProjectionValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func llama3RoPEExpectedValues(
	input []float32,
	seqLen int,
	numHeads int,
	headDim int,
) []float32 {
	out := make([]float32, len(input))
	halfDim := headDim / 2
	const (
		base            = 500000.0
		factor          = 32.0
		lowFreqFactor   = 1.0
		highFreqFactor  = 4.0
		originalContext = 8192.0
	)

	for seqIndex := range seqLen {
		for headIndex := range numHeads {
			for pairIndex := range halfDim {
				llama3RoPEExpectedHalfPair(
					input, out, seqIndex, headIndex, pairIndex, numHeads, headDim,
					base, factor, lowFreqFactor, highFreqFactor, originalContext,
				)
			}
		}
	}

	return out
}

func llama3ScaledInvFreq(
	invFreq float64,
	originalContext float64,
	factor float64,
	lowFreqFactor float64,
	highFreqFactor float64,
) float64 {
	wavelen := (2 * math.Pi) / invFreq
	lowFreqWavelen := originalContext / lowFreqFactor
	highFreqWavelen := originalContext / highFreqFactor

	if wavelen > lowFreqWavelen {
		return invFreq / factor
	}

	if wavelen < highFreqWavelen {
		return invFreq
	}

	smooth := (originalContext/wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)

	return (1-smooth)*(invFreq/factor) + smooth*invFreq
}

func llama3RoPEExpectedPair(
	input []float32,
	out []float32,
	seqIndex int,
	headIndex int,
	pairIndex int,
	numHeads int,
	headDim int,
	base float64,
	factor float64,
	lowFreqFactor float64,
	highFreqFactor float64,
	originalContext float64,
) {
	inputIndex := (seqIndex*numHeads+headIndex)*headDim + pairIndex*2
	exponent := -2 * float64(pairIndex) / float64(headDim)
	invFreq := math.Pow(base, exponent)
	invFreq = llama3ScaledInvFreq(invFreq, originalContext, factor, lowFreqFactor, highFreqFactor)
	angle := float32(seqIndex) * float32(invFreq)
	cosTheta := float32(math.Cos(float64(angle)))
	sinTheta := float32(math.Sin(float64(angle)))
	even := input[inputIndex]
	odd := input[inputIndex+1]

	out[inputIndex] = even*cosTheta - odd*sinTheta
	out[inputIndex+1] = even*sinTheta + odd*cosTheta
}

func llama3RoPEExpectedHalfPair(
	input []float32,
	out []float32,
	seqIndex int,
	headIndex int,
	pairIndex int,
	numHeads int,
	headDim int,
	base float64,
	factor float64,
	lowFreqFactor float64,
	highFreqFactor float64,
	originalContext float64,
) {
	headOffset := (seqIndex*numHeads + headIndex) * headDim
	halfDim := headDim / 2
	evenIndex := headOffset + pairIndex
	oddIndex := headOffset + halfDim + pairIndex
	exponent := -2 * float64(pairIndex) / float64(headDim)
	invFreq := math.Pow(base, exponent)
	invFreq = llama3ScaledInvFreq(invFreq, originalContext, factor, lowFreqFactor, highFreqFactor)
	angle := float32(seqIndex) * float32(invFreq)
	cosTheta := float32(math.Cos(float64(angle)))
	sinTheta := float32(math.Sin(float64(angle)))
	even := input[evenIndex]
	odd := input[oddIndex]

	out[evenIndex] = even*cosTheta - odd*sinTheta
	out[oddIndex] = even*sinTheta + odd*cosTheta
}

func assertRoPEBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	out tensor.Tensor,
	storageDType dtype.DType,
	fixture ropeFixture,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, 2)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, out, fixture.expectedFloat32, 32)
}
