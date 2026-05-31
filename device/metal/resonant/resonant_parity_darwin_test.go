//go:build darwin && cgo

package resonant

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpuresonant "github.com/theapemachine/puter/device/cpu/resonant"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

const (
	resonantForwardMetalMaxULPF32   = 16
	resonantForwardMetalMaxULPF16   = 2
	resonantForwardMetalMaxULPBF16  = 2
	resonantBackwardMetalMaxULPF32  = 20
	resonantBackwardMetalMaxULPF16  = 1
	resonantBackwardMetalMaxULPBF16 = 1
)

func resonantForwardMetalMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return resonantForwardMetalMaxULPF32
	case dtype.Float16:
		return resonantForwardMetalMaxULPF16
	case dtype.BFloat16:
		return resonantForwardMetalMaxULPBF16
	default:
		panic("resonant parity: unsupported dtype")
	}
}

func resonantBackwardMetalMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return resonantBackwardMetalMaxULPF32
	case dtype.Float16:
		return resonantBackwardMetalMaxULPF16
	case dtype.BFloat16:
		return resonantBackwardMetalMaxULPBF16
	default:
		panic("resonant parity: unsupported dtype")
	}
}

var resonantParityFormats = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func TestResonantUpdateForwardMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	config := device.ResonantUpdateConfig{
		Scale:    0.25,
		Damping:  0.1,
		ZeroDiag: true,
	}

	for _, format := range resonantParityFormats {
		format := format

		testingObject.Run(fmt.Sprintf("format=%s", format), func(formatTest *testing.T) {
			maxULP := resonantForwardMetalMaxULP(format)

			for _, shape := range resonantParityShapes {
				shape := shape

				formatTest.Run(
					fmt.Sprintf("BT=%d_H=%d_D=%d", shape.batchTime, shape.headCount, shape.headDim),
					func(shapeTest *testing.T) {
						elementCount := shape.batchTime * shape.headCount * shape.headDim
						diagCount := shape.headCount * shape.headDim
						seed := int64(0x5400 + int64(elementCount) + int64(format)*0x100)

						x, y, vr, vi, diag := randomResonantInputs(elementCount, diagCount, seed)
						wantXOut, wantYOut, wantAOut, wantBOut, wantInvROut := resonantForwardReference(
							x, y, vr, vi, diag,
							shape.headCount,
							shape.headDim,
							config,
							format,
						)

						xTensor := harness.UploadVector(x, format)
						yTensor := harness.UploadVector(y, format)
						vrTensor := harness.UploadVector(vr, format)
						viTensor := harness.UploadVector(vi, format)
						diagTensor := harness.UploadVector(diag, format)
						xOutTensor := harness.UploadVector(make([]float32, elementCount), format)
						yOutTensor := harness.UploadVector(make([]float32, elementCount), format)
						aOutTensor := harness.UploadVector(make([]float32, elementCount), format)
						bOutTensor := harness.UploadVector(make([]float32, elementCount), format)
						invROutTensor := harness.UploadVector(make([]float32, elementCount), format)
						defer xTensor.Close()
						defer yTensor.Close()
						defer vrTensor.Close()
						defer viTensor.Close()
						defer diagTensor.Close()
						defer xOutTensor.Close()
						defer yOutTensor.Close()
						defer aOutTensor.Close()
						defer bOutTensor.Close()
						defer invROutTensor.Close()

						dispatchErr := DispatchResonantUpdateForwardRefs(
							harness.ContextRef(),
							xTensor.Ref(),
							yTensor.Ref(),
							vrTensor.Ref(),
							viTensor.Ref(),
							diagTensor.Ref(),
							xOutTensor.Ref(),
							yOutTensor.Ref(),
							aOutTensor.Ref(),
							bOutTensor.Ref(),
							invROutTensor.Ref(),
							shape.batchTime,
							shape.headCount,
							shape.headDim,
							config,
							format,
						)

						if dispatchErr != nil {
							shapeTest.Fatalf("dispatch forward: %v", dispatchErr)
						}

						gotXOut := harness.DownloadFloat32(xOutTensor, format)
						gotYOut := harness.DownloadFloat32(yOutTensor, format)
						gotAOut := harness.DownloadFloat32(aOutTensor, format)
						gotBOut := harness.DownloadFloat32(bOutTensor, format)
						gotInvROut := harness.DownloadFloat32(invROutTensor, format)

						parity.AssertDecodedSlicesMatch(shapeTest, gotXOut, wantXOut, format, maxULP)
						parity.AssertDecodedSlicesMatch(shapeTest, gotYOut, wantYOut, format, maxULP)
						parity.AssertDecodedSlicesMatch(shapeTest, gotAOut, wantAOut, format, maxULP)
						parity.AssertDecodedSlicesMatch(shapeTest, gotBOut, wantBOut, format, maxULP)
						parity.AssertDecodedSlicesMatch(shapeTest, gotInvROut, wantInvROut, format, maxULP)
					},
				)
			}
		})
	}
}

func TestResonantUpdateBackwardMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	config := device.ResonantUpdateConfig{
		Scale:    0.25,
		Damping:  0.1,
		ZeroDiag: false,
	}
	elementCount := 64
	headCount := 4
	headDim := 16
	diagCount := headCount * headDim

	for _, format := range resonantParityFormats {
		format := format

		testingObject.Run(fmt.Sprintf("format=%s", format), func(formatTest *testing.T) {
			maxULP := resonantBackwardMetalMaxULP(format)

			x, y, vr, vi, diag := randomResonantInputs(elementCount, diagCount, 0x5500+int64(format))
			_, _, aOut, bOut, invROut := resonantForwardReference(
				x, y, vr, vi, diag,
				headCount,
				headDim,
				config,
				format,
			)
			gradXOut := randomResonantVector(elementCount, 0x5501+int64(format))
			gradYOut := randomResonantVector(elementCount, 0x5502+int64(format))
			wantGradX, wantGradY, wantGradVR, wantGradVI := resonantBackwardReference(
				gradXOut, gradYOut,
				x, y, diag, aOut, bOut, invROut,
				headCount,
				headDim,
				config,
				format,
			)

			gradXOutTensor := harness.UploadVector(gradXOut, format)
			gradYOutTensor := harness.UploadVector(gradYOut, format)
			xTensor := harness.UploadVector(x, format)
			yTensor := harness.UploadVector(y, format)
			diagTensor := harness.UploadVector(diag, format)
			aTensor := harness.UploadVector(aOut, format)
			bTensor := harness.UploadVector(bOut, format)
			invRTensor := harness.UploadVector(invROut, format)
			gradXTensor := harness.UploadVector(make([]float32, elementCount), format)
			gradYTensor := harness.UploadVector(make([]float32, elementCount), format)
			gradVRTensor := harness.UploadVector(make([]float32, elementCount), format)
			gradVITensor := harness.UploadVector(make([]float32, elementCount), format)
			defer gradXOutTensor.Close()
			defer gradYOutTensor.Close()
			defer xTensor.Close()
			defer yTensor.Close()
			defer diagTensor.Close()
			defer aTensor.Close()
			defer bTensor.Close()
			defer invRTensor.Close()
			defer gradXTensor.Close()
			defer gradYTensor.Close()
			defer gradVRTensor.Close()
			defer gradVITensor.Close()

			dispatchErr := DispatchResonantUpdateBackwardRefs(
				harness.ContextRef(),
				gradXOutTensor.Ref(),
				gradYOutTensor.Ref(),
				xTensor.Ref(),
				yTensor.Ref(),
				diagTensor.Ref(),
				aTensor.Ref(),
				bTensor.Ref(),
				invRTensor.Ref(),
				gradXTensor.Ref(),
				gradYTensor.Ref(),
				gradVRTensor.Ref(),
				gradVITensor.Ref(),
				elementCount/headCount/headDim,
				headCount,
				headDim,
				config,
				format,
			)

			if dispatchErr != nil {
				formatTest.Fatalf("dispatch backward: %v", dispatchErr)
			}

			gotGradX := harness.DownloadFloat32(gradXTensor, format)
			gotGradY := harness.DownloadFloat32(gradYTensor, format)
			gotGradVR := harness.DownloadFloat32(gradVRTensor, format)
			gotGradVI := harness.DownloadFloat32(gradVITensor, format)

			parity.AssertDecodedSlicesMatch(formatTest, gotGradX, wantGradX, format, maxULP)
			parity.AssertDecodedSlicesMatch(formatTest, gotGradY, wantGradY, format, maxULP)
			parity.AssertDecodedSlicesMatch(formatTest, gotGradVR, wantGradVR, format, maxULP)
			parity.AssertDecodedSlicesMatch(formatTest, gotGradVI, wantGradVI, format, maxULP)
		})
	}
}

type resonantParityShape struct {
	batchTime int
	headCount int
	headDim   int
}

var resonantParityShapes = []resonantParityShape{
	{batchTime: 1, headCount: 1, headDim: 1},
	{batchTime: 1, headCount: 1, headDim: 7},
	{batchTime: 1, headCount: 64, headDim: 1},
	{batchTime: 4, headCount: 4, headDim: 4},
	{batchTime: 8, headCount: 8, headDim: 16},
}

func resonantForwardReference(
	x, y, vr, vi, diag []float32,
	headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) ([]float32, []float32, []float32, []float32, []float32) {
	elementCount := len(x)
	xOut := make([]float32, elementCount)
	yOut := make([]float32, elementCount)
	aOut := make([]float32, elementCount)
	bOut := make([]float32, elementCount)
	invROut := make([]float32, elementCount)

	switch format {
	case dtype.Float32:
		cpuresonant.ResonantUpdateForwardGeneric(
			x, y, vr, vi, diag,
			xOut, yOut, aOut, bOut, invROut,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.Float16:
		encodedXOut := make([]uint16, elementCount)
		encodedYOut := make([]uint16, elementCount)
		encodedAOut := make([]uint16, elementCount)
		encodedBOut := make([]uint16, elementCount)
		encodedInvROut := make([]uint16, elementCount)

		cpuresonant.ResonantUpdateForwardFloat16(
			encodeFloat16Slice(x),
			encodeFloat16Slice(y),
			encodeFloat16Slice(vr),
			encodeFloat16Slice(vi),
			encodeFloat16Slice(diag),
			encodedXOut,
			encodedYOut,
			encodedAOut,
			encodedBOut,
			encodedInvROut,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)

		xOut = decodeFloat16Slice(encodedXOut)
		yOut = decodeFloat16Slice(encodedYOut)
		aOut = decodeFloat16Slice(encodedAOut)
		bOut = decodeFloat16Slice(encodedBOut)
		invROut = decodeFloat16Slice(encodedInvROut)
	case dtype.BFloat16:
		encodedXOut := make([]uint16, elementCount)
		encodedYOut := make([]uint16, elementCount)
		encodedAOut := make([]uint16, elementCount)
		encodedBOut := make([]uint16, elementCount)
		encodedInvROut := make([]uint16, elementCount)

		cpuresonant.ResonantUpdateForwardBFloat16(
			encodeBFloat16Slice(x),
			encodeBFloat16Slice(y),
			encodeBFloat16Slice(vr),
			encodeBFloat16Slice(vi),
			encodeBFloat16Slice(diag),
			encodedXOut,
			encodedYOut,
			encodedAOut,
			encodedBOut,
			encodedInvROut,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)

		xOut = decodeBFloat16Slice(encodedXOut)
		yOut = decodeBFloat16Slice(encodedYOut)
		aOut = decodeBFloat16Slice(encodedAOut)
		bOut = decodeBFloat16Slice(encodedBOut)
		invROut = decodeBFloat16Slice(encodedInvROut)
	default:
		panic("resonant parity: unsupported dtype")
	}

	return xOut, yOut, aOut, bOut, invROut
}

func resonantBackwardReference(
	gradXOut, gradYOut, x, y, diag, a, b, invR []float32,
	headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) ([]float32, []float32, []float32, []float32) {
	elementCount := len(gradXOut)
	gradX := make([]float32, elementCount)
	gradY := make([]float32, elementCount)
	gradVR := make([]float32, elementCount)
	gradVI := make([]float32, elementCount)

	switch format {
	case dtype.Float32:
		cpuresonant.ResonantUpdateBackwardGeneric(
			gradXOut, gradYOut,
			x, y, diag, a, b, invR,
			gradX, gradY, gradVR, gradVI,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)
	case dtype.Float16:
		encodedGradX := make([]uint16, elementCount)
		encodedGradY := make([]uint16, elementCount)
		encodedGradVR := make([]uint16, elementCount)
		encodedGradVI := make([]uint16, elementCount)

		cpuresonant.ResonantUpdateBackwardFloat16(
			encodeFloat16Slice(gradXOut),
			encodeFloat16Slice(gradYOut),
			encodeFloat16Slice(x),
			encodeFloat16Slice(y),
			encodeFloat16Slice(diag),
			encodeFloat16Slice(a),
			encodeFloat16Slice(b),
			encodeFloat16Slice(invR),
			encodedGradX,
			encodedGradY,
			encodedGradVR,
			encodedGradVI,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)

		gradX = decodeFloat16Slice(encodedGradX)
		gradY = decodeFloat16Slice(encodedGradY)
		gradVR = decodeFloat16Slice(encodedGradVR)
		gradVI = decodeFloat16Slice(encodedGradVI)
	case dtype.BFloat16:
		encodedGradX := make([]uint16, elementCount)
		encodedGradY := make([]uint16, elementCount)
		encodedGradVR := make([]uint16, elementCount)
		encodedGradVI := make([]uint16, elementCount)

		cpuresonant.ResonantUpdateBackwardBFloat16(
			encodeBFloat16Slice(gradXOut),
			encodeBFloat16Slice(gradYOut),
			encodeBFloat16Slice(x),
			encodeBFloat16Slice(y),
			encodeBFloat16Slice(diag),
			encodeBFloat16Slice(a),
			encodeBFloat16Slice(b),
			encodeBFloat16Slice(invR),
			encodedGradX,
			encodedGradY,
			encodedGradVR,
			encodedGradVI,
			headCount,
			headDim,
			config.Scale,
			config.Damping,
			config.ZeroDiag,
		)

		gradX = decodeBFloat16Slice(encodedGradX)
		gradY = decodeBFloat16Slice(encodedGradY)
		gradVR = decodeBFloat16Slice(encodedGradVR)
		gradVI = decodeBFloat16Slice(encodedGradVI)
	default:
		panic("resonant parity: unsupported dtype")
	}

	return gradX, gradY, gradVR, gradVI
}

func encodeFloat16Slice(values []float32) []uint16 {
	encoded := make([]uint16, len(values))

	for index, value := range values {
		encoded[index] = uint16(dtype.Fromfloat32(value))
	}

	return encoded
}

func decodeFloat16Slice(values []uint16) []float32 {
	decoded := make([]float32, len(values))

	for index, value := range values {
		decoded[index] = dtype.F16(value).Float32()
	}

	return decoded
}

func encodeBFloat16Slice(values []float32) []uint16 {
	encoded := make([]uint16, len(values))

	for index, value := range values {
		encoded[index] = uint16(dtype.NewBfloat16FromFloat32(value))
	}

	return encoded
}

func decodeBFloat16Slice(values []uint16) []float32 {
	decoded := make([]float32, len(values))

	for index, value := range values {
		decoded[index] = dtype.BF16(value).Float32()
	}

	return decoded
}

func randomResonantInputs(elementCount, diagCount int, seed int64) ([]float32, []float32, []float32, []float32, []float32) {
	return randomResonantVector(elementCount, seed),
		randomResonantVector(elementCount, seed+1),
		randomResonantVector(elementCount, seed+2),
		randomResonantVector(elementCount, seed+3),
		randomResonantVector(diagCount, seed+4)
}

func randomResonantVector(length int, seed int64) []float32 {
	return parity.RandomUnaryInput(length, seed)
}
