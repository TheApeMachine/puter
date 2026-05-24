//go:build xla

package dequant_test

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpudequant "github.com/theapemachine/puter/device/cpu/dequant"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestDequantXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA int8 dequant", t, func() {
		scale := float32(0.125)
		zeroPoint := int8(4)
		config := device.DequantInt8Config{Scale: scale, ZeroPoint: zeroPoint}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := make([]int8, count)

				for index := range source {
					source[index] = int8((index*17 + 3) % 251)
				}

				want := make([]float32, count)
				cpudequant.DequantInt8Native(want, source, scale, zeroPoint)

				sourceTensor := uploadInt8Tensor(harness, source)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer sourceTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Dequant(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(sourceTensor),
					count,
					config,
					dtype.Float32,
					dtype.Int8,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}

func uploadInt8Tensor(harness *xla.ParityHarness, values []int8) *xla.DeviceTensor {
	bytesIn := make([]byte, len(values))

	for index, value := range values {
		bytesIn[index] = byte(value)
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		panic(err)
	}

	deviceTensor, err := harness.Backend().Upload(shape, dtype.Int8, bytesIn)

	if err != nil {
		panic(err)
	}

	residentTensor, ok := deviceTensor.(*xla.DeviceTensor)

	if !ok {
		panic("xla parity: upload did not return DeviceTensor")
	}

	return residentTensor
}

func TestDequant4XLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA int4 dequant", t, func() {
		scale := float32(0.0625)
		zeroPoint := int8(3)
		config := device.DequantInt4Config{Scale: scale, ZeroPoint: zeroPoint}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := int4PackedBytes(count, 0x4d00+int64(count))
				want := make([]float32, count)
				cpudequant.DequantInt4Native(want, int4VectorFromBytes(source, count), scale, zeroPoint)

				sourceTensor := uploadPackedInt4Tensor(harness, source)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer sourceTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Dequant4(
					xla.ResidentPointer(outputTensor),
					xla.ResidentPointer(sourceTensor),
					count,
					config,
					dtype.Float32,
					dtype.Int4,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}

func int4PackedBytes(length int, seed int64) []byte {
	byteCount := (length + 1) / 2
	bytesIn := make([]byte, byteCount)

	for index := range bytesIn {
		bytesIn[index] = byte((seed + int64(index)*17) % 256)
	}

	return bytesIn
}

func int4VectorFromBytes(bytesIn []byte, length int) tensor.Int4Vector {
	pairs := make([]dtype.Int4Pair, len(bytesIn))

	for index, value := range bytesIn {
		pairs[index] = dtype.Int4Pair(value)
	}

	return tensor.NewInt4Vector(pairs, length)
}

func uploadPackedInt4Tensor(harness *xla.ParityHarness, bytesIn []byte) *xla.DeviceTensor {
	shape, err := tensor.NewShape([]int{len(bytesIn)})

	if err != nil {
		panic(err)
	}

	deviceTensor, err := harness.Backend().Upload(shape, dtype.Int4, bytesIn)

	if err != nil {
		panic(err)
	}

	residentTensor, ok := deviceTensor.(*xla.DeviceTensor)

	if !ok {
		panic("xla parity: upload did not return DeviceTensor")
	}

	return residentTensor
}

func BenchmarkDequantXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	source := make([]int8, count)
	config := device.DequantInt8Config{Scale: 0.05, ZeroPoint: 0}
	sourceTensor := uploadInt8Tensor(harness, source)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer sourceTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Dequant(
			xla.ResidentPointer(outputTensor),
			xla.ResidentPointer(sourceTensor),
			count,
			config,
			dtype.Float32,
			dtype.Int8,
		)
	}
}
