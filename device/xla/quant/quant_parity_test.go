//go:build xla

package quant_test

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpuquant "github.com/theapemachine/puter/device/cpu/quant"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestQuantXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA int8 quant", t, func() {
		scale := float32(0.0875)
		zeroPoint := int8(-13)
		config := device.DequantInt8Config{Scale: scale, ZeroPoint: zeroPoint}

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				source := xlaparity.RandomUnaryInput(count, 0x7100+int64(count))
				want := make([]int8, count)
				cpuquant.QuantInt8Native(want, source, scale, zeroPoint)

				sourceTensor := harness.UploadVector(source, dtype.Float32)
				destinationTensor := uploadInt8Tensor(harness, count)
				defer sourceTensor.Close()
				defer destinationTensor.Close()

				harness.Backend().Quant(
					xla.ResidentPointer(destinationTensor),
					xla.ResidentPointer(sourceTensor),
					count,
					config,
					dtype.Int8,
					dtype.Float32,
				)

				got := downloadInt8Tensor(harness, destinationTensor, count)

				for index := range got {
					if got[index] != want[index] {
						t.Fatalf("lane %d got=%d want=%d", index, got[index], want[index])
					}
				}
			})
		}
	})
}

func uploadInt8Tensor(harness *xla.ParityHarness, count int) *xla.DeviceTensor {
	bytesIn := make([]byte, count)
	shape, err := tensor.NewShape([]int{count})

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

func downloadInt8Tensor(harness *xla.ParityHarness, deviceTensor *xla.DeviceTensor, count int) []int8 {
	bytesOut := harness.DownloadBytes(deviceTensor)
	values := make([]int8, count)

	for index := 0; index < count; index++ {
		values[index] = int8(bytesOut[index])
	}

	return values
}

func BenchmarkQuantXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	source := xlaparity.RandomUnaryInput(count, 0x7200)
	config := device.DequantInt8Config{Scale: 0.05, ZeroPoint: 0}
	sourceTensor := harness.UploadVector(source, dtype.Float32)
	destinationTensor := uploadInt8Tensor(harness, count)
	defer sourceTensor.Close()
	defer destinationTensor.Close()

	for b.Loop() {
		harness.Backend().Quant(
			xla.ResidentPointer(destinationTensor),
			xla.ResidentPointer(sourceTensor),
			count,
			config,
			dtype.Int8,
			dtype.Float32,
		)
	}
}
