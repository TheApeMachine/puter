//go:build darwin && cgo

package normalization

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGroupNormMetalParity(t *testing.T) {
	harness := parity.NewHarness(t)
	defer harness.Close()

	convey.Convey("Given Metal GroupNorm kernels", t, func() {
		configs := []struct {
			batch    int
			channels int
			groups   int
		}{
			{batch: 2, channels: 8, groups: 2},
			{batch: 1, channels: 4, groups: 2},
			{batch: 3, channels: 6, groups: 3},
		}

		for _, config := range configs {
			config := config

			convey.Convey(fmt.Sprintf("batch=%d channels=%d groups=%d", config.batch, config.channels, config.groups), func() {
				for _, storageDType := range []dtype.DType{
					dtype.Float32,
					dtype.Float16,
					dtype.BFloat16,
				} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, spatial := range parity.Lengths {
							convey.Convey(fmt.Sprintf("spatial=%d", spatial), func() {
								elementCount := config.batch * config.channels * spatial
								seedBase := int64(0x4E00 + config.batch*100 + config.channels*10 + config.groups + spatial)

								input := randomGroupNormVector(elementCount, seedBase)
								scale := randomGroupNormVector(config.channels, seedBase+1)
								bias := randomGroupNormVector(config.channels, seedBase+2)
								want := parity.GroupNormReference(
									input,
									scale,
									bias,
									config.batch,
									config.channels,
									spatial,
									config.groups,
									storageDType,
								)

								inputTensor := harness.UploadVector(input, storageDType)
								scaleTensor := harness.UploadVector(scale, storageDType)
								biasTensor := harness.UploadVector(bias, storageDType)
								outputTensor := harness.UploadVector(make([]float32, elementCount), storageDType)
								defer inputTensor.Close()
								defer scaleTensor.Close()
								defer biasTensor.Close()
								defer outputTensor.Close()

								if err := DispatchGroupNormRefs(
									harness.ContextRef(),
									inputTensor.Ref(),
									scaleTensor.Ref(),
									biasTensor.Ref(),
									outputTensor.Ref(),
									storageDType,
									uint32(config.batch),
									uint32(config.channels),
									uint32(spatial),
									uint32(config.groups),
								); err != nil {
									t.Fatalf("dispatch GroupNorm: %v", err)
								}

								got := harness.DownloadFloat32(outputTensor, storageDType)
								maxULP := groupNormMaxULP(storageDType)
								parity.AssertFloat32SlicesWithinULP(t, got, want, maxULP)
							})
						}
					})
				}
			})
		}
	})
}

func TestBatchNormDenormMetalParity(testingObject *testing.T) {
	harness := parity.NewHarness(testingObject)
	defer harness.Close()

	convey.Convey("Given Metal BatchNormDenorm kernels", testingObject, func() {
		batch := 2
		channels := 3

		for _, storageDType := range []dtype.DType{
			dtype.Float32,
			dtype.Float16,
			dtype.BFloat16,
		} {
			storageDType := storageDType

			convey.Convey(storageDType.Name(), func() {
				for _, spatial := range parity.Lengths {
					convey.Convey(fmt.Sprintf("spatial=%d", spatial), func() {
						elementCount := batch * channels * spatial
						seedBase := int64(0x7E00 + spatial)
						input := randomGroupNormVector(elementCount, seedBase)
						mean := randomGroupNormVector(channels, seedBase+1)
						variance := positiveGroupNormVector(channels, seedBase+2)
						want := parity.BatchNormDenormReference(
							input,
							mean,
							variance,
							batch,
							channels,
							spatial,
							storageDType,
						)

						inputTensor := harness.UploadVector(input, storageDType)
						meanTensor := harness.UploadVector(mean, storageDType)
						varianceTensor := harness.UploadVector(variance, storageDType)
						outputTensor := harness.UploadVector(make([]float32, elementCount), storageDType)
						defer inputTensor.Close()
						defer meanTensor.Close()
						defer varianceTensor.Close()
						defer outputTensor.Close()

						if err := DispatchBatchNormDenormRefs(
							harness.ContextRef(),
							inputTensor.Ref(),
							meanTensor.Ref(),
							varianceTensor.Ref(),
							outputTensor.Ref(),
							storageDType,
							uint32(batch*channels),
							uint32(channels),
							uint32(spatial),
						); err != nil {
							testingObject.Fatalf("dispatch BatchNormDenorm: %v", err)
						}

						got := harness.DownloadFloat32(outputTensor, storageDType)
						maxULP := batchNormDenormMaxULP(storageDType)
						parity.AssertFloat32SlicesWithinULP(testingObject, got, want, maxULP)
					})
				}
			})
		}
	})
}

func BenchmarkGroupNormMetalFloat32(b *testing.B) {
	harness := parity.NewHarness(b)
	defer harness.Close()

	batch := 2
	channels := 8
	groups := 2
	spatial := 8192
	elementCount := batch * channels * spatial

	input := randomGroupNormVector(elementCount, 1)
	scale := randomGroupNormVector(channels, 2)
	bias := randomGroupNormVector(channels, 3)

	inputTensor := harness.UploadVector(input, dtype.Float32)
	scaleTensor := harness.UploadVector(scale, dtype.Float32)
	biasTensor := harness.UploadVector(bias, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	b.ResetTimer()

	for b.Loop() {
		if err := DispatchGroupNormRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			scaleTensor.Ref(),
			biasTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(batch),
			uint32(channels),
			uint32(spatial),
			uint32(groups),
		); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchNormDenormMetalFloat32(benchmark *testing.B) {
	harness := parity.NewHarness(benchmark)
	defer harness.Close()

	batch := 2
	channels := 128
	spatial := 8192
	elementCount := batch * channels * spatial
	input := randomGroupNormVector(elementCount, 0x7E10)
	mean := randomGroupNormVector(channels, 0x7E11)
	variance := positiveGroupNormVector(channels, 0x7E12)

	inputTensor := harness.UploadVector(input, dtype.Float32)
	meanTensor := harness.UploadVector(mean, dtype.Float32)
	varianceTensor := harness.UploadVector(variance, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	defer inputTensor.Close()
	defer meanTensor.Close()
	defer varianceTensor.Close()
	defer outputTensor.Close()

	benchmark.SetBytes(int64(elementCount * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := DispatchBatchNormDenormRefs(
			harness.ContextRef(),
			inputTensor.Ref(),
			meanTensor.Ref(),
			varianceTensor.Ref(),
			outputTensor.Ref(),
			dtype.Float32,
			uint32(batch*channels),
			uint32(channels),
			uint32(spatial),
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func randomGroupNormVector(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = rng.Float32()*4.0 - 2.0
	}

	return values
}

func positiveGroupNormVector(length int, seed int64) []float32 {
	values := randomGroupNormVector(length, seed)

	for index := range values {
		if values[index] < 0 {
			values[index] = -values[index]
		}

		values[index] += 0.01
	}

	return values
}

func groupNormMaxULP(format dtype.DType) int {
	if format == dtype.Float32 {
		return 3
	}

	return 8
}

func batchNormDenormMaxULP(format dtype.DType) int {
	if format == dtype.Float32 {
		return 3
	}

	return 8
}
