//go:build xla

package hawkes_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuhawkes "github.com/theapemachine/puter/device/cpu/hawkes"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceHawkes = cpuhawkes.New()

func TestHawkesXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA HawkesIntensity", t, func() {
		eventCount := 64
		queryCount := 7
		mu := float32(0.2)
		alpha := float32(0.5)
		beta := float32(1.1)

		eventTimes := xlaparity.RandomUnaryInput(eventCount, 0xa100)
		queryTimes := xlaparity.RandomUnaryInput(queryCount, 0xa200)
		want := make([]float32, queryCount)
		referenceHawkes.HawkesIntensity(
			unsafe.Pointer(&eventTimes[0]),
			unsafe.Pointer(&queryTimes[0]),
			unsafe.Pointer(&want[0]),
			eventCount, queryCount,
			mu, alpha, beta,
			dtype.Float32,
		)

		eventTensor := harness.UploadVector(eventTimes, dtype.Float32)
		queryTensor := harness.UploadVector(queryTimes, dtype.Float32)
		outputTensor := harness.UploadVector(make([]float32, queryCount), dtype.Float32)
		defer eventTensor.Close()
		defer queryTensor.Close()
		defer outputTensor.Close()

		harness.Backend().HawkesIntensity(
			xla.ResidentPointer(eventTensor),
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(outputTensor),
			eventCount, queryCount,
			mu, alpha, beta,
			dtype.Float32,
		)

		got := harness.DownloadFloat32(outputTensor, dtype.Float32)
		xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
	})

	convey.Convey("Given XLA HawkesKernelMatrix", t, func() {
		for _, eventCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("E=%d", eventCount), func() {
				alpha := float32(0.4)
				beta := float32(0.9)
				eventTimes := xlaparity.RandomUnaryInput(eventCount, 0xa300+int64(eventCount))
				want := make([]float32, eventCount*eventCount)
				referenceHawkes.HawkesKernelMatrix(
					unsafe.Pointer(&eventTimes[0]),
					unsafe.Pointer(&want[0]),
					eventCount,
					alpha, beta,
					dtype.Float32,
				)

				eventTensor := harness.UploadVector(eventTimes, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, eventCount*eventCount), eventCount, eventCount, dtype.Float32)
				defer eventTensor.Close()
				defer outputTensor.Close()

				harness.Backend().HawkesKernelMatrix(
					xla.ResidentPointer(eventTensor),
					xla.ResidentPointer(outputTensor),
					eventCount,
					alpha, beta,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})
}

func TestMarkovBlanketPartitionXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA MarkovBlanketPartition", t, func() {
		for _, nodeCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("N=%d", nodeCount), func() {
				adjacency := make([]float32, nodeCount*nodeCount)

				for row := 0; row < nodeCount; row++ {
					for col := 0; col < nodeCount; col++ {
						if row != col {
							adjacency[row*nodeCount+col] = float32((row*3+col*5)%3) / 4
						}
					}
				}

				internalCount := nodeCount / 4

				if internalCount == 0 && nodeCount > 1 {
					internalCount = 1
				}

				internal := make([]int32, internalCount)

				for index := range internal {
					internal[index] = int32((index*2 + 1) % nodeCount)
				}

				want := make([]int32, nodeCount)
				referenceHawkes.MarkovBlanketPartition(
					unsafe.Pointer(&adjacency[0]),
					unsafe.Pointer(&internal[0]),
					unsafe.Pointer(&want[0]),
					nodeCount, internalCount,
					dtype.Float32,
				)

				adjacencyTensor := harness.UploadMatrix(adjacency, nodeCount, nodeCount, dtype.Float32)
				internalTensor := harness.UploadInt32Vector(internal)
				outputTensor := harness.UploadInt32Vector(make([]int32, nodeCount))
				defer adjacencyTensor.Close()
				defer internalTensor.Close()
				defer outputTensor.Close()

				harness.Backend().MarkovBlanketPartition(
					xla.ResidentPointer(adjacencyTensor),
					xla.ResidentPointer(internalTensor),
					xla.ResidentPointer(outputTensor),
					nodeCount, internalCount,
					dtype.Float32,
				)

				got := decodeInt32Vector(harness.DownloadBytes(outputTensor), nodeCount)
				convey.So(got, convey.ShouldResemble, want)
			})
		}
	})
}

func decodeInt32Vector(bytesIn []byte, count int) []int32 {
	values := make([]int32, count)

	for index := range values {
		offset := index * 4
		values[index] = int32(bytesIn[offset]) |
			int32(bytesIn[offset+1])<<8 |
			int32(bytesIn[offset+2])<<16 |
			int32(bytesIn[offset+3])<<24
	}

	return values
}

func BenchmarkHawkesIntensityXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	eventCount := 1024
	queryCount := 64
	eventTimes := xlaparity.RandomUnaryInput(eventCount, 0xa400)
	queryTimes := xlaparity.RandomUnaryInput(queryCount, 0xa500)
	eventTensor := harness.UploadVector(eventTimes, dtype.Float32)
	queryTensor := harness.UploadVector(queryTimes, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, queryCount), dtype.Float32)
	defer eventTensor.Close()
	defer queryTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().HawkesIntensity(
			xla.ResidentPointer(eventTensor),
			xla.ResidentPointer(queryTensor),
			xla.ResidentPointer(outputTensor),
			eventCount, queryCount,
			0.2, 0.5, 1.1,
			dtype.Float32,
		)
	}
}
