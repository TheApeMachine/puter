//go:build darwin && cgo

package layernorm

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestAdaptiveRMSNormMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal AdaptiveRMSNorm kernels", testingObject, func() {
		config := device.RMSNormConfig{Epsilon: 1e-6}

		for _, cols := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match CPU for N=%d", cols), func() {
				rowsPerBatch := 3
				batches := 2
				rows := batches * rowsPerBatch
				modulationCols := 2 * cols

				for _, storageDType := range []dtype.DType{
					dtype.Float32,
					dtype.Float16,
					dtype.BFloat16,
				} {
					convey.Convey(storageDType.Name(), func() {
						elementCount := rows * cols
						input := randomAdaptiveRMSNormVector(elementCount, int64(3000+cols))
						modulation := randomAdaptiveRMSNormModulation(
							batches,
							modulationCols,
							cols,
							int64(4000+cols),
						)
						want := parity.AdaptiveRMSNormReference(
							config,
							input,
							modulation,
							rows,
							cols,
							rowsPerBatch,
							modulationCols,
							storageDType,
						)
						wantBytes := parity.AdaptiveRMSNormReferenceBytes(
							config,
							input,
							modulation,
							rows,
							cols,
							rowsPerBatch,
							modulationCols,
							storageDType,
						)

						inputTensor := harness.UploadVector(input, storageDType)
						modulationTensor := harness.UploadVector(modulation, storageDType)
						outputTensor := harness.UploadVector(make([]float32, elementCount), storageDType)
						defer inputTensor.Close()
						defer modulationTensor.Close()
						defer outputTensor.Close()

						if err := DispatchAdaptiveRMSNormRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							modulationTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(rows),
							uint32(cols),
							uint32(rowsPerBatch),
							uint32(modulationCols),
							float32(config.Epsilon),
						); err != nil {
							testingObject.Fatalf("dispatch AdaptiveRMSNorm: %v", err)
						}

						if storageDType != dtype.Float32 {
							harness.Sync()
							assertAdaptiveRMSNormStorageParity(
								testingObject,
								outputTensor.ReadBytes(),
								wantBytes,
								storageDType,
							)
							return
						}

						got := harness.DownloadFloat32(outputTensor, storageDType)
						parity.AssertFloat32SlicesWithinULP(
							testingObject,
							got,
							want,
							adaptiveRMSNormMetalMaxULP(storageDType),
						)
					})
				}
			})
		}
	})
}

func BenchmarkAdaptiveRMSNormMetalFloat32(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	config := device.RMSNormConfig{Epsilon: 1e-6}
	rowsPerBatch := 16
	batches := 2
	rows := batches * rowsPerBatch
	cols := 8192
	modulationCols := 2 * cols
	input := randomAdaptiveRMSNormVector(rows*cols, 1)
	modulation := randomAdaptiveRMSNormModulation(batches, modulationCols, cols, 2)

	inputTensor := harness.UploadVector(input, dtype.Float32)
	modulationTensor := harness.UploadVector(modulation, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, rows*cols), dtype.Float32)
	defer inputTensor.Close()
	defer modulationTensor.Close()
	defer outputTensor.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := DispatchAdaptiveRMSNormRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			modulationTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(rows),
			uint32(cols),
			uint32(rowsPerBatch),
			uint32(modulationCols),
			float32(config.Epsilon),
		); err != nil {
			benchmark.Fatal(err)
		}
	}

	harness.Sync()
}

func randomAdaptiveRMSNormVector(length int, seed int64) []float32 {
	generator := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = generator.Float32()*2.0 - 1.0
	}

	return values
}

func randomAdaptiveRMSNormModulation(
	batches int,
	modulationCols int,
	cols int,
	seed int64,
) []float32 {
	generator := rand.New(rand.NewSource(seed))
	values := make([]float32, batches*modulationCols)

	for batchIndex := range batches {
		offset := batchIndex * modulationCols

		for columnIndex := range cols {
			values[offset+columnIndex] = generator.Float32()*0.2 - 0.1
			values[offset+cols+columnIndex] = generator.Float32()*0.2 - 0.1
		}
	}

	return values
}

func assertAdaptiveRMSNormStorageParity(
	testingObject *testing.T,
	got []byte,
	want []byte,
	format dtype.DType,
) {
	testingObject.Helper()

	if len(got) != len(want) {
		testingObject.Fatalf("byte length mismatch got=%d want=%d", len(got), len(want))
	}

	for byteIndex := 0; byteIndex < len(got); byteIndex += 2 {
		gotBits := binary.LittleEndian.Uint16(got[byteIndex : byteIndex+2])
		wantBits := binary.LittleEndian.Uint16(want[byteIndex : byteIndex+2])

		if adaptiveRMSNormUint16Distance(gotBits, wantBits) <= adaptiveRMSNormStorageMaxDistance(format) {
			continue
		}

		testingObject.Fatalf(
			"%s lane %d got_bits=%d want_bits=%d distance=%d max=%d",
			format.Name(),
			byteIndex/2,
			gotBits,
			wantBits,
			adaptiveRMSNormUint16Distance(gotBits, wantBits),
			adaptiveRMSNormStorageMaxDistance(format),
		)
	}
}

func adaptiveRMSNormUint16Distance(left uint16, right uint16) uint16 {
	if left > right {
		left, right = right, left
	}

	return right - left
}

func adaptiveRMSNormStorageMaxDistance(format dtype.DType) uint16 {
	if format == dtype.BFloat16 {
		return 128
	}

	return 64
}

func adaptiveRMSNormMetalMaxULP(format dtype.DType) int {
	if format == dtype.Float32 {
		return 512
	}

	return 24
}
