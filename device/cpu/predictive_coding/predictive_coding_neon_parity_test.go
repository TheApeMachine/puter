//go:build arm64

package predictive_coding

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestPCPredictionF32NEONParity(t *testing.T) {
	convey.Convey("Given PCPredictionF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xC311+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xC312+int64(length))
				got := make([]float32, outDim)
				want := make([]float32, outDim)

				PCPredictionF32NEON(
					&weights[0], &representation[0], &got[0], outDim, inDim,
				)
				PredictionFloat32Scalar(weights, representation, want, outDim, inDim)

				assertPredictionParity(t, got, want, outDim)
			})
		}
	})
}

func TestPCPredictionErrorF32NEONParity(t *testing.T) {
	convey.Convey("Given PCPredictionErrorF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionErrorFloat32Scalar for N=%d", length), func() {
				observed, predicted, _ := randomPredictiveCodingVectors(length, 0xC313+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PCPredictionErrorF32NEON(
					&observed[0], &predicted[0], &got[0], length,
				)
				PredictionErrorFloat32Scalar(observed, predicted, want)

				assertSliceParity(t, got, want)
			})
		}
	})
}

func TestPCUpdateRepresentationF32NEONParity(t *testing.T) {
	convey.Convey("Given PCUpdateRepresentationF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateRepresentationFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xC314+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xC315+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xC315+int64(length)+1)
				got := make([]float32, inDim)
				want := make([]float32, inDim)

				PCUpdateRepresentationF32NEON(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateRepresentationFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				assertSliceParity(t, got, want)
			})
		}
	})
}

func TestPCUpdateWeightsF32NEONParity(t *testing.T) {
	convey.Convey("Given PCUpdateWeightsF32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateWeightsFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xC316+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xC317+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xC317+int64(length)+1)
				got := make([]float32, outDim*inDim)
				want := make([]float32, outDim*inDim)

				PCUpdateWeightsF32NEON(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateWeightsFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				assertSliceParity(t, got, want)
			})
		}
	})
}
