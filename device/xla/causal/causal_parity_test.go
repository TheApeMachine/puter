//go:build xla

package causal_test

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpucausal "github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceCausal = cpucausal.New()

func TestCausalXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA CATE", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				runCATEParity(t, harness, count, 0xb100+int64(count))
			})
		}
	})

	convey.Convey("Given XLA Counterfactual", t, func() {
		const slope = float32(0.75)

		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				observedY := xlaparity.RandomUnaryInput(count, 0xb210+int64(count))
				observedX := xlaparity.RandomUnaryInput(count, 0xb220+int64(count))
				counterfactualX := xlaparity.RandomUnaryInput(count, 0xb230+int64(count))
				want := make([]float32, count)
				referenceCausal.Counterfactual(
					unsafe.Pointer(&observedY[0]),
					unsafe.Pointer(&observedX[0]),
					unsafe.Pointer(&counterfactualX[0]),
					unsafe.Pointer(&want[0]),
					count,
					slope,
					dtype.Float32,
				)

				observedYTensor := harness.UploadVector(observedY, dtype.Float32)
				observedXTensor := harness.UploadVector(observedX, dtype.Float32)
				counterfactualXTensor := harness.UploadVector(counterfactualX, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer observedYTensor.Close()
				defer observedXTensor.Close()
				defer counterfactualXTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Counterfactual(
					xla.ResidentPointer(observedYTensor),
					xla.ResidentPointer(observedXTensor),
					xla.ResidentPointer(counterfactualXTensor),
					xla.ResidentPointer(outputTensor),
					count,
					slope,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA IVEstimate", t, func() {
		for _, count := range xlaparity.Lengths {
			if count < 2 {
				continue
			}

			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				instrument := xlaparity.RandomUnaryInput(count, 0xb310+int64(count))
				treatment := xlaparity.RandomUnaryInput(count, 0xb320+int64(count))
				outcome := xlaparity.RandomUnaryInput(count, 0xb330+int64(count))
				want := make([]float32, 1)
				referenceCausal.IVEstimate(
					unsafe.Pointer(&instrument[0]),
					unsafe.Pointer(&treatment[0]),
					unsafe.Pointer(&outcome[0]),
					count,
					unsafe.Pointer(&want[0]),
					dtype.Float32,
				)

				instrumentTensor := harness.UploadVector(instrument, dtype.Float32)
				treatmentTensor := harness.UploadVector(treatment, dtype.Float32)
				outcomeTensor := harness.UploadVector(outcome, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer instrumentTensor.Close()
				defer treatmentTensor.Close()
				defer outcomeTensor.Close()
				defer outputTensor.Close()

				harness.Backend().IVEstimate(
					xla.ResidentPointer(instrumentTensor),
					xla.ResidentPointer(treatmentTensor),
					xla.ResidentPointer(outcomeTensor),
					count,
					xla.ResidentPointer(outputTensor),
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA BackdoorAdjustment", t, func() {
		for _, xCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("X=%d", xCount), func() {
				const zCount = 5
				const yCount = 4
				conditional := xlaparity.RandomUnaryInput(xCount*zCount*yCount, 0xb400+int64(xCount))
				marginalZ := xlaparity.RandomUnaryInput(zCount, 0xb410+int64(xCount))
				want := make([]float32, xCount*yCount)
				referenceCausal.BackdoorAdjustment(
					unsafe.Pointer(&conditional[0]),
					unsafe.Pointer(&marginalZ[0]),
					unsafe.Pointer(&want[0]),
					xCount, zCount, yCount,
					dtype.Float32,
				)

				conditionalTensor := harness.UploadVolume(conditional, xCount, zCount, yCount, dtype.Float32)
				marginalTensor := harness.UploadVector(marginalZ, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, xCount*yCount), xCount, yCount, dtype.Float32)
				defer conditionalTensor.Close()
				defer marginalTensor.Close()
				defer outputTensor.Close()

				harness.Backend().BackdoorAdjustment(
					xla.ResidentPointer(conditionalTensor),
					xla.ResidentPointer(marginalTensor),
					xla.ResidentPointer(outputTensor),
					xCount, zCount, yCount,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA FrontdoorAdjustment", t, func() {
		for _, xCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("X=%d", xCount), func() {
				const mediatorCount = 5
				const yCount = 4
				mediatorGivenX := xlaparity.RandomUnaryInput(xCount*mediatorCount, 0xb420+int64(xCount))
				outcomeGivenXM := xlaparity.RandomUnaryInput(xCount*mediatorCount*yCount, 0xb430+int64(xCount))
				marginalX := xlaparity.RandomUnaryInput(xCount, 0xb440+int64(xCount))
				want := make([]float32, xCount*yCount)
				referenceCausal.FrontdoorAdjustment(
					unsafe.Pointer(&mediatorGivenX[0]),
					unsafe.Pointer(&outcomeGivenXM[0]),
					unsafe.Pointer(&marginalX[0]),
					unsafe.Pointer(&want[0]),
					xCount, mediatorCount, yCount,
					dtype.Float32,
				)

				mediatorTensor := harness.UploadMatrix(mediatorGivenX, xCount, mediatorCount, dtype.Float32)
				outcomeTensor := harness.UploadVolume(outcomeGivenXM, xCount, mediatorCount, yCount, dtype.Float32)
				marginalTensor := harness.UploadVector(marginalX, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, xCount*yCount), xCount, yCount, dtype.Float32)
				defer mediatorTensor.Close()
				defer outcomeTensor.Close()
				defer marginalTensor.Close()
				defer outputTensor.Close()

				harness.Backend().FrontdoorAdjustment(
					xla.ResidentPointer(mediatorTensor),
					xla.ResidentPointer(outcomeTensor),
					xla.ResidentPointer(marginalTensor),
					xla.ResidentPointer(outputTensor),
					xCount, mediatorCount, yCount,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA Cholesky", t, func() {
		for _, order := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("N=%d", order), func() {
				input := spdMatrix(order, 0xb500+int64(order))
				want := make([]float32, order*order)
				referenceCausal.Cholesky(
					unsafe.Pointer(&input[0]),
					unsafe.Pointer(&want[0]),
					order,
					dtype.Float32,
				)

				inputTensor := harness.UploadMatrix(input, order, order, dtype.Float32)
				outputTensor := harness.UploadMatrix(make([]float32, order*order), order, order, dtype.Float32)
				defer inputTensor.Close()
				defer outputTensor.Close()

				harness.Backend().Cholesky(
					xla.ResidentPointer(inputTensor),
					xla.ResidentPointer(outputTensor),
					order,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA DAGMarkovFactorization", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				conditionals := xlaparity.RandomUnaryInput(count, 0xb600+int64(count))

				for index := range conditionals {
					if conditionals[index] <= 0 {
						conditionals[index] = float32(index%7+1) * 0.125
					}
				}

				want := make([]float32, 1)
				referenceCausal.DAGMarkovFactorization(
					unsafe.Pointer(&conditionals[0]),
					count,
					unsafe.Pointer(&want[0]),
					dtype.Float32,
				)

				conditionalsTensor := harness.UploadVector(conditionals, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, 1), dtype.Float32)
				defer conditionalsTensor.Close()
				defer outputTensor.Close()

				harness.Backend().DAGMarkovFactorization(
					xla.ResidentPointer(conditionalsTensor),
					count,
					xla.ResidentPointer(outputTensor),
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 8)
			})
		}
	})

	convey.Convey("Given XLA DoIntervene", t, func() {
		for _, nodeCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("N=%d", nodeCount), func() {
				adjacency := xlaparity.RandomUnaryInput(nodeCount*nodeCount, 0xb700+int64(nodeCount))
				intervenedCount := nodeCount / 3

				if intervenedCount == 0 && nodeCount > 0 {
					intervenedCount = 1
				}

				intervened := make([]int32, intervenedCount)

				for index := range intervened {
					intervened[index] = int32((index*2 + 1) % nodeCount)
				}

				want := make([]float32, nodeCount*nodeCount)
				referenceCausal.DoIntervene(
					unsafe.Pointer(&adjacency[0]),
					unsafe.Pointer(&intervened[0]),
					unsafe.Pointer(&want[0]),
					nodeCount, intervenedCount,
					dtype.Float32,
				)

				adjacencyTensor := harness.UploadMatrix(adjacency, nodeCount, nodeCount, dtype.Float32)
				intervenedTensor := harness.UploadInt32Vector(intervened)
				outputTensor := harness.UploadMatrix(make([]float32, nodeCount*nodeCount), nodeCount, nodeCount, dtype.Float32)
				defer adjacencyTensor.Close()
				defer intervenedTensor.Close()
				defer outputTensor.Close()

				harness.Backend().DoIntervene(
					xla.ResidentPointer(adjacencyTensor),
					xla.ResidentPointer(intervenedTensor),
					xla.ResidentPointer(outputTensor),
					nodeCount, intervenedCount,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 4)
			})
		}
	})

	convey.Convey("Given XLA MarkovFlowActive", t, func() {
		for _, nodeCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("N=%d", nodeCount), func() {
				runMarkovFlowParity(t, harness, nodeCount, true, 0xb800+int64(nodeCount))
			})
		}
	})

	convey.Convey("Given XLA MarkovFlowInternal", t, func() {
		for _, nodeCount := range []int{1, 7, 64} {
			convey.Convey(fmt.Sprintf("N=%d", nodeCount), func() {
				runMarkovFlowParity(t, harness, nodeCount, false, 0xb900+int64(nodeCount))
			})
		}
	})
}

func runCATEParity(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	count int,
	seed int64,
) {
	testingTB.Helper()

	treated := xlaparity.RandomUnaryInput(count, seed)
	control := xlaparity.RandomUnaryInput(count, seed+0x100)
	want := make([]float32, count)
	referenceCausal.CATE(
		unsafe.Pointer(&treated[0]),
		unsafe.Pointer(&control[0]),
		unsafe.Pointer(&want[0]),
		count,
		dtype.Float32,
	)

	treatedTensor := harness.UploadVector(treated, dtype.Float32)
	controlTensor := harness.UploadVector(control, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer treatedTensor.Close()
	defer controlTensor.Close()
	defer outputTensor.Close()

	harness.Backend().CATE(
		xla.ResidentPointer(treatedTensor),
		xla.ResidentPointer(controlTensor),
		xla.ResidentPointer(outputTensor),
		count,
		dtype.Float32,
	)

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 2)
}

func runMarkovFlowParity(
	testingTB testing.TB,
	harness *xla.ParityHarness,
	nodeCount int,
	active bool,
	seed int64,
) {
	testingTB.Helper()

	mutualInformation := xlaparity.RandomUnaryInput(nodeCount*nodeCount, seed)
	partition := make([]int32, nodeCount)

	for index := range partition {
		partition[index] = int32(index % 4)
	}

	want := make([]float32, nodeCount)

	if active {
		referenceCausal.MarkovFlowActive(
			unsafe.Pointer(&mutualInformation[0]),
			unsafe.Pointer(&partition[0]),
			unsafe.Pointer(&want[0]),
			nodeCount,
			dtype.Float32,
		)
	}

	if !active {
		referenceCausal.MarkovFlowInternal(
			unsafe.Pointer(&mutualInformation[0]),
			unsafe.Pointer(&partition[0]),
			unsafe.Pointer(&want[0]),
			nodeCount,
			dtype.Float32,
		)
	}

	miTensor := harness.UploadMatrix(mutualInformation, nodeCount, nodeCount, dtype.Float32)
	partitionTensor := harness.UploadInt32Vector(partition)
	outputTensor := harness.UploadVector(make([]float32, nodeCount), dtype.Float32)
	defer miTensor.Close()
	defer partitionTensor.Close()
	defer outputTensor.Close()

	if active {
		harness.Backend().MarkovFlowActive(
			xla.ResidentPointer(miTensor),
			xla.ResidentPointer(partitionTensor),
			xla.ResidentPointer(outputTensor),
			nodeCount,
			dtype.Float32,
		)
	}

	if !active {
		harness.Backend().MarkovFlowInternal(
			xla.ResidentPointer(miTensor),
			xla.ResidentPointer(partitionTensor),
			xla.ResidentPointer(outputTensor),
			nodeCount,
			dtype.Float32,
		)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float32)
	xlaparity.AssertFloat32SlicesWithinULP(testingTB, got, want, 4)
}

func spdMatrix(order int, seed int64) []float32 {
	lower := xlaparity.RandomUnaryInput(order*order, seed)
	matrix := make([]float32, order*order)

	for row := 0; row < order; row++ {
		for col := 0; col <= row; col++ {
			value := lower[row*order+col]

			if row == col {
				value = float32(math.Abs(float64(value))) + float32(order)
			}

			matrix[row*order+col] = value
		}
	}

	output := make([]float32, order*order)

	for row := 0; row < order; row++ {
		for col := 0; col < order; col++ {
			sum := float32(0)

			for inner := 0; inner < order; inner++ {
				left := float32(0)
				right := float32(0)

				if inner <= row {
					left = matrix[row*order+inner]
				}

				if inner <= col {
					right = matrix[col*order+inner]
				}

				sum += left * right
			}

			output[row*order+col] = sum
		}
	}

	return output
}

func BenchmarkCATEXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	runCATEParity(b, harness, 8192, 0xb300)
}
