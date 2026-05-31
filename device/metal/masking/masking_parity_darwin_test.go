//go:build darwin && cgo

package masking

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	cpumasking "github.com/theapemachine/puter/device/cpu/masking"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

const maskingMetalMaxULP = 0

func maskingMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return maskingMetalMaxULP
	case dtype.Float16, dtype.BFloat16:
		return 1
	default:
		return maskingMetalMaxULP
	}
}

func TestApplyMaskMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal ApplyMask kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						input := parity.RandomUnaryInput(count, 0x5100+int64(count))
						mask := parity.RandomUnaryInput(count, 0x5101+int64(count))
						want := maskingReference(input, mask, storageDType)

						inputTensor := harness.UploadVector(input, storageDType)
						maskTensor := harness.UploadVector(mask, storageDType)
						outputTensor := harness.UploadVector(make([]float32, count), storageDType)
						defer inputTensor.Close()
						defer maskTensor.Close()
						defer outputTensor.Close()

						dispatchErr := DispatchApplyMaskRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							maskTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(count),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, maskingMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func TestCausalMaskMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal CausalMask kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, length := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", length), func() {
						side := maskingSquareSide(length)
						want := causalMaskReference(side, storageDType)

						outputTensor := harness.UploadVector(make([]float32, side*side), storageDType)
						defer outputTensor.Close()

						dispatchErr := DispatchCausalMaskRefs(
							harness.ContextRef(),
							outputTensor.Ref(),
							storageDType,
							uint32(side),
							uint32(side),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, maskingMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func TestALiBiBiasMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal ALiBiBias kernels", testingObject, func() {
		for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, length := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", length), func() {
						side := maskingSquareSide(length)
						scores := randomMaskingScores(side, side, 0x5120+int64(length))
						slope := []float32{0.125}
						want := alibiReference(scores, slope, side, storageDType)

						scoresTensor := harness.UploadVector(scores, storageDType)
						slopeTensor := harness.UploadVector(slope, storageDType)
						outputTensor := harness.UploadVector(make([]float32, side*side), storageDType)
						defer scoresTensor.Close()
						defer slopeTensor.Close()
						defer outputTensor.Close()

						dispatchErr := DispatchALiBiBiasRefs(
							harness.ContextRef(),
							scoresTensor.Ref(),
							slopeTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(side),
							uint32(side),
						)
						convey.So(dispatchErr, convey.ShouldBeNil)

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertDecodedSlicesMatch(testingObject, got, want, storageDType, alibiMaxULP(storageDType))
					})
				}
			})
		}
	})
}

func BenchmarkApplyMaskMetal(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	count := 8192
	input := parity.RandomUnaryInput(count, 0x5130)
	mask := parity.RandomUnaryInput(count, 0x5131)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	maskTensor := harness.UploadVector(mask, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer maskTensor.Close()
	defer outputTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = DispatchApplyMaskRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			maskTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(count),
		)
	}

	harness.Sync()
}

func maskingSquareSide(length int) int {
	side := 1

	for side*side < length {
		side++
	}

	return side
}

func randomMaskingScores(rows, cols int, seed int64) []float32 {
	generator := rand.New(rand.NewSource(seed))
	values := make([]float32, rows*cols)

	for index := range values {
		values[index] = generator.Float32()*2.0 - 1.0
	}

	return values
}

func alibiMaxULP(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return 0
	case dtype.Float16:
		return 1
	case dtype.BFloat16:
		return 1
	default:
		return 0
	}
}

func maskingReference(input, mask []float32, format dtype.DType) []float32 {
	count := len(input)
	outputBytes := float32SliceToNativeBytes(input, format)
	maskBytes := float32SliceToNativeBytes(mask, format)
	wantBytes := make([]byte, len(outputBytes))

	cpumasking.Default.ApplyMask(
		unsafe.Pointer(&outputBytes[0]),
		unsafe.Pointer(&maskBytes[0]),
		unsafe.Pointer(&wantBytes[0]),
		count,
		format,
	)

	return nativeBytesToFloat32Slice(wantBytes, format)
}

func causalMaskReference(side int, format dtype.DType) []float32 {
	wantBytes := make([]byte, side*side*elementByteSize(format))
	cpumasking.Default.CausalMask(unsafe.Pointer(&wantBytes[0]), side, side, format)

	return nativeBytesToFloat32Slice(wantBytes, format)
}

func alibiReference(scores, slope []float32, side int, format dtype.DType) []float32 {
	scoreBytes := float32SliceToNativeBytes(scores, format)
	slopeBytes := float32SliceToNativeBytes(slope, format)
	wantBytes := make([]byte, len(scoreBytes))

	cpumasking.Default.ALiBiBias(
		unsafe.Pointer(&scoreBytes[0]),
		unsafe.Pointer(&slopeBytes[0]),
		unsafe.Pointer(&wantBytes[0]),
		side,
		side,
		format,
	)

	return nativeBytesToFloat32Slice(wantBytes, format)
}

func float32SliceToNativeBytes(values []float32, format dtype.DType) []byte {
	switch format {
	case dtype.Float32:
		return convert.Float32ToBytes(values)
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for index, value := range values {
			encoded[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded)
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded)
	default:
		panic(fmt.Sprintf("unsupported dtype %v", format))
	}
}

func nativeBytesToFloat32Slice(bytesIn []byte, format dtype.DType) []float32 {
	decoded, err := convert.BytesToFloat32(format, bytesIn)

	if err != nil {
		panic(err)
	}

	return decoded
}

func elementByteSize(format dtype.DType) int {
	switch format {
	case dtype.Float32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 0
	}
}
