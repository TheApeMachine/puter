//go:build xla

package pool_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	cpupool "github.com/theapemachine/puter/device/cpu/pool"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

func TestPoolXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA max pool2d", t, func() {
		for _, storageDType := range xlaparity.FloatParityDTypes {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, count := range xlaparity.Lengths {
					if count < 4 {
						continue
					}

					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						batch := 1
						channels := 1
						inHeight := count
						inWidth := count
						outHeight := count / 2
						outWidth := count / 2
						config := device.PoolConfig{
							KernelH: inHeight / outHeight,
							KernelW: inWidth / outWidth,
							StrideH: inHeight / outHeight,
							StrideW: inWidth / outWidth,
						}

						source := xlaparity.RandomUnaryInput(batch*channels*inHeight*inWidth, 0xA100+int64(count))
						wantBytes := poolReferenceBytes(source, config, batch, channels, inHeight, inWidth, outHeight, outWidth, storageDType, true)

						sourceTensor := harness.UploadVector(source, storageDType)
						destinationTensor := harness.UploadVector(make([]float32, batch*channels*outHeight*outWidth), storageDType)
						defer sourceTensor.Close()
						defer destinationTensor.Close()

						harness.Backend().MaxPool2D(
							config,
							unsafe.Pointer(sourceTensor),
							unsafe.Pointer(destinationTensor),
							batch, channels, inHeight, inWidth, outHeight, outWidth,
							storageDType,
						)

						assertPoolOutput(t, harness, destinationTensor, storageDType, wantBytes)
					})
				}
			})
		}
	})
}

func poolReferenceBytes(
	source []float32,
	config device.PoolConfig,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
) []byte {
	sourceBytes, err := xlaparity.EncodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, batch*channels*outHeight*outWidth*elementByteWidth(format))

	if useMax {
		cpupool.New().MaxPool2D(
			cpupool.PoolConfig{
				KernelH:  config.KernelH,
				KernelW:  config.KernelW,
				StrideH:  config.StrideH,
				StrideW:  config.StrideW,
				PaddingH: config.PaddingH,
				PaddingW: config.PaddingW,
			},
			unsafe.Pointer(&sourceBytes[0]),
			unsafe.Pointer(&destinationBytes[0]),
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			format,
		)
	}

	return destinationBytes
}

func elementByteWidth(format dtype.DType) int {
	width, err := format.Size()

	if err != nil {
		panic(err)
	}

	return width
}

func assertPoolOutput(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	deviceTensor *xla.DeviceTensor,
	storageDType dtype.DType,
	wantBytes []byte,
) {
	testingTB.Helper()

	if storageDType == dtype.Float32 || storageDType == dtype.Float64 {
		got := harness.DownloadFloat32(deviceTensor, storageDType)
		want := xlaparity.DecodeFloat32Vector(wantBytes, storageDType)
		xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 2)
	}

	if storageDType != dtype.Float32 && storageDType != dtype.Float64 {
		gotBytes := harness.DownloadBytes(deviceTensor)
		xlaparity.AssertEncodedSlicesEqual(testingTB, gotBytes, wantBytes)
	}
}
