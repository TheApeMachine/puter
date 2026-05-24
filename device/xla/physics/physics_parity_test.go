//go:build xla

package physics_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpuphysics "github.com/theapemachine/puter/device/cpu/physics"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referencePhysics = cpuphysics.New()

func TestPhysicsXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	spacing := float32(0.25)

	convey.Convey("Given XLA Grad1D", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0xc100+int64(count))
				want := make([]float32, count)
				referencePhysics.Grad1D(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Grad1D(
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-1", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				runLaplacianParity(t, harness, []int{count}, spacing, 0xd100+int64(count))
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-2", t, func() {
		for _, side := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("%dx%d", side, side), func() {
				runLaplacianParity(t, harness, []int{side, side}, spacing, 0xd200+int64(side))
			})
		}
	})

	convey.Convey("Given XLA Laplacian rank-3", t, func() {
		for _, depth := range []int{1, 4, 7} {
			for _, side := range []int{1, 7, 64} {
				convey.Convey(fmt.Sprintf("%dx%dx%d", depth, side, side), func() {
					runLaplacianParity(
						t,
						harness,
						[]int{depth, side, side},
						spacing,
						0xd300+int64(depth*100+side),
					)
				})
			}
		}
	})

	convey.Convey("Given XLA Laplacian4", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				input := xlaparity.RandomUnaryInput(count, 0xd400+int64(count))
				want := make([]float32, count)
				referencePhysics.Laplacian4(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				inputTensor := harness.UploadVector(input, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Laplacian4(
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA QuantumPotential", t, func() {
		spacing := float32(0.25)

		for _, count := range []int{7, 64, 1024} {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				density := xlaparity.RandomUnaryInput(count, 0xd600+int64(count))

				for index := range density {
					if density[index] <= 0 {
						density[index] = float32(index%11+1) * 0.05
					}
				}

				want := make([]float32, count)
				referencePhysics.QuantumPotential(
					unsafe.Pointer(&density[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				densityTensor := harness.UploadVector(density, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer densityTensor.Close()
				defer outputTensor.Close()

				harness.Backend().QuantumPotential(
					xla.ResidentPointer(densityTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA BohmianVelocity", t, func() {
		spacing := float32(0.25)

		for _, count := range []int{7, 64, 1024} {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				phase := xlaparity.RandomUnaryInput(count, 0xd700+int64(count))
				want := make([]float32, count)
				referencePhysics.BohmianVelocity(
					unsafe.Pointer(&phase[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				phaseTensor := harness.UploadVector(phase, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer phaseTensor.Close()
				defer outputTensor.Close()

				harness.Backend().BohmianVelocity(
					xla.ResidentPointer(phaseTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA MadelungContinuity", t, func() {
		spacing := float32(0.25)

		for _, count := range []int{7, 64, 1024} {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				density := xlaparity.RandomUnaryInput(count, 0xd800+int64(count))
				velocity := xlaparity.RandomUnaryInput(count, 0xd900+int64(count))
				want := make([]float32, count)
				referencePhysics.MadelungContinuity(
					unsafe.Pointer(&density[0]),
					unsafe.Pointer(&velocity[0]),
					unsafe.Pointer(&want[0]),
					count,
					spacing,
					dtype.Float32,
				)

				densityTensor := harness.UploadVector(density, dtype.Float32)
				velocityTensor := harness.UploadVector(velocity, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer densityTensor.Close()
				defer velocityTensor.Close()
				defer outputTensor.Close()

				harness.Backend().MadelungContinuity(
					xla.ResidentPointer(densityTensor),
					xla.ResidentPointer(velocityTensor),
					xla.ResidentPointer(outputTensor),
					count,
					spacing,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA FFT1D at power-of-two lengths", t, func() {
		for _, count := range fftPowerOfTwoLengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				runFFT1DParity(t, harness, count, 0xe100+int64(count), false)
			})
		}
	})

	convey.Convey("Given XLA IFFT1D at power-of-two lengths", t, func() {
		for _, count := range fftPowerOfTwoLengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				runFFT1DParity(t, harness, count, 0xe200+int64(count), true)
			})
		}
	})
}

var fftPowerOfTwoLengths = []int{1, 64, 1024, 8192}

func runFFT1DParity(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	count int,
	seed int64,
	inverse bool,
) {
	testingTB.Helper()

	realIn := xlaparity.RandomUnaryInput(count, seed)
	imagIn := xlaparity.RandomUnaryInput(count, seed+0x10)
	wantReal := make([]float32, count)
	wantImag := make([]float32, count)

	if inverse {
		referencePhysics.IFFT1D(
			unsafe.Pointer(&realIn[0]),
			unsafe.Pointer(&imagIn[0]),
			unsafe.Pointer(&wantReal[0]),
			unsafe.Pointer(&wantImag[0]),
			count,
			dtype.Float32,
		)
	} else {
		referencePhysics.FFT1D(
			unsafe.Pointer(&realIn[0]),
			unsafe.Pointer(&imagIn[0]),
			unsafe.Pointer(&wantReal[0]),
			unsafe.Pointer(&wantImag[0]),
			count,
			dtype.Float32,
		)
	}

	realInTensor := harness.UploadVector(realIn, dtype.Float32)
	imagInTensor := harness.UploadVector(imagIn, dtype.Float32)
	realOutTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	imagOutTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer realInTensor.Close()
	defer imagInTensor.Close()
	defer realOutTensor.Close()
	defer imagOutTensor.Close()

	if inverse {
		harness.Backend().IFFT1D(
			xla.ResidentPointer(realInTensor),
			xla.ResidentPointer(imagInTensor),
			xla.ResidentPointer(realOutTensor),
			xla.ResidentPointer(imagOutTensor),
			count,
			dtype.Float32,
		)
	} else {
		harness.Backend().FFT1D(
			xla.ResidentPointer(realInTensor),
			xla.ResidentPointer(imagInTensor),
			xla.ResidentPointer(realOutTensor),
			xla.ResidentPointer(imagOutTensor),
			count,
			dtype.Float32,
		)
	}

	gotReal := harness.DownloadFloat32(realOutTensor, dtype.Float32)
	gotImag := harness.DownloadFloat32(imagOutTensor, dtype.Float32)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, gotReal, wantReal, 8)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, gotImag, wantImag, 8)
}

func runLaplacianParity(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	dims []int,
	spacing float32,
	seed int64,
) {
	testingTB.Helper()

	elementCount := 1

	for _, dim := range dims {
		elementCount *= dim
	}

	input := xlaparity.RandomUnaryInput(elementCount, seed)
	want := make([]float32, elementCount)
	referencePhysics.Laplacian(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&want[0]),
		dims,
		spacing,
		dtype.Float32,
	)

	var inputTensor *xla.DeviceTensor
	var outputTensor *xla.DeviceTensor

	switch len(dims) {
	case 1:
		inputTensor = harness.UploadVector(input, dtype.Float32)
		outputTensor = harness.UploadVector(make([]float32, elementCount), dtype.Float32)
	case 2:
		inputTensor = harness.UploadMatrix(input, dims[0], dims[1], dtype.Float32)
		outputTensor = harness.UploadMatrix(make([]float32, elementCount), dims[0], dims[1], dtype.Float32)
	case 3:
		inputTensor = harness.UploadVolume(input, dims[0], dims[1], dims[2], dtype.Float32)
		outputTensor = harness.UploadVolume(make([]float32, elementCount), dims[0], dims[1], dims[2], dtype.Float32)
	default:
		testingTB.Fatalf("unsupported laplacian rank %d", len(dims))
	}

	defer inputTensor.Close()
	defer outputTensor.Close()

	harness.Backend().Laplacian(
		xla.ResidentPointer(inputTensor),
		xla.ResidentPointer(outputTensor),
		dims,
		spacing,
		dtype.Float32,
	)

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func BenchmarkFFT1DXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	realIn := xlaparity.RandomUnaryInput(count, 0xe300)
	imagIn := xlaparity.RandomUnaryInput(count, 0xe400)
	realInTensor := harness.UploadVector(realIn, dtype.Float32)
	imagInTensor := harness.UploadVector(imagIn, dtype.Float32)
	realOutTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	imagOutTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer realInTensor.Close()
	defer imagInTensor.Close()
	defer realOutTensor.Close()
	defer imagOutTensor.Close()

	for b.Loop() {
		harness.Backend().FFT1D(
			xla.ResidentPointer(realInTensor),
			xla.ResidentPointer(imagInTensor),
			xla.ResidentPointer(realOutTensor),
			xla.ResidentPointer(imagOutTensor),
			count,
			dtype.Float32,
		)
	}
}

func BenchmarkGrad1DXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	spacing := float32(0.25)
	input := xlaparity.RandomUnaryInput(count, 0xc200)
	inputTensor := harness.UploadVector(input, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Grad1D(
			xla.ResidentPointer(inputTensor),
			xla.ResidentPointer(outputTensor),
			count,
			spacing,
			dtype.Float32,
		)
	}
}

func BenchmarkLaplacian2DXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	side := 64
	spacing := float32(0.25)
	elementCount := side * side
	input := xlaparity.RandomUnaryInput(elementCount, 0xd500)
	inputTensor := harness.UploadMatrix(input, side, side, dtype.Float32)
	outputTensor := harness.UploadMatrix(make([]float32, elementCount), side, side, dtype.Float32)
	defer inputTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().Laplacian(
			xla.ResidentPointer(inputTensor),
			xla.ResidentPointer(outputTensor),
			[]int{side, side},
			spacing,
			dtype.Float32,
		)
	}
}
