//go:build cuda

package dequant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	cpudequant "github.com/theapemachine/puter/device/cpu/dequant"
	"github.com/theapemachine/puter/device/cuda/internal/parity"
)

func randomInt8Slice(count int, seed int64) []int8 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]int8, count)

	for index := range values {
		values[index] = int8(rng.Intn(255) - 128)
	}

	return values
}

func TestDequantCUDAParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given CUDA int8 dequant", t, func() {
		scale := float32(0.0875)
		zeroPoint := int8(-13)

		for _, count := range parity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						source := randomInt8Slice(count, 0xDE00+int64(count))
						wantF32 := make([]float32, count)
						cpudequant.DequantInt8Native(wantF32, source, scale, zeroPoint)
						wantBytes := convertFromFloat32(wantF32, storageDType)

						sourceTensor := harness.UploadBytes(int8ToBytes(source))
						destinationTensor := harness.UploadVector(make([]float32, count), storageDType)
						defer sourceTensor.Close()
						defer destinationTensor.Close()

						if err := DispatchDequantRefs(
							harness.ContextRef(),
							sourceTensor.Ref(),
							destinationTensor.Ref(),
							storageDType,
							scale,
							zeroPoint,
							uint32(count),
						); err != nil {
							t.Fatalf("dispatch Dequant: %v", err)
						}

						if storageDType == dtype.Float32 {
							got := harness.DownloadFloat32(destinationTensor, storageDType)
							parity.AssertFloat32SlicesWithinULP(t, got, wantF32, 1)
						}

						if storageDType != dtype.Float32 {
							harness.Sync()
							parity.AssertEncodedSlicesEqual(t, destinationTensor.ReadBytes(), wantBytes)
						}
					})
				}
			})
		}
	})
}

func int8ToBytes(values []int8) []byte {
	bytesOut := make([]byte, len(values))

	for index, value := range values {
		bytesOut[index] = byte(value)
	}

	return bytesOut
}

func convertFromFloat32(values []float32, format dtype.DType) []byte {
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
