//go:build amd64

package predictive_coding

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const predictiveCodingAVX2SSE2MaxULP = 0

func avx2PredictiveCodingAvailable() bool {
	return cpu.X86.HasAVX2 && cpu.X86.HasFMA
}

func sse2PredictiveCodingAvailable() bool {
	return cpu.X86.HasSSE2
}

func TestPCPredictionF32AVX2Parity(t *testing.T) {
	if !avx2PredictiveCodingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given PCPredictionF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB321+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB322+int64(length))
				got := make([]float32, outDim)
				want := make([]float32, outDim)

				PCPredictionF32AVX2(
					&weights[0], &representation[0], &got[0], outDim, inDim,
				)
				PredictionFloat32Scalar(weights, representation, want, outDim, inDim)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCPredictionF32SSE2Parity(t *testing.T) {
	if !sse2PredictiveCodingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given PCPredictionF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB323+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB324+int64(length))
				got := make([]float32, outDim)
				want := make([]float32, outDim)

				PCPredictionF32SSE2(
					&weights[0], &representation[0], &got[0], outDim, inDim,
				)
				PredictionFloat32Scalar(weights, representation, want, outDim, inDim)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCPredictionErrorF32AVX2Parity(t *testing.T) {
	if !avx2PredictiveCodingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given PCPredictionErrorF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionErrorFloat32Scalar for N=%d", length), func() {
				observed, predicted, _ := randomPredictiveCodingVectors(length, 0xB325+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PCPredictionErrorF32AVX2(
					&observed[0], &predicted[0], &got[0], length,
				)
				PredictionErrorFloat32Scalar(observed, predicted, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCPredictionErrorF32SSE2Parity(t *testing.T) {
	if !sse2PredictiveCodingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given PCPredictionErrorF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionErrorFloat32Scalar for N=%d", length), func() {
				observed, predicted, _ := randomPredictiveCodingVectors(length, 0xB326+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PCPredictionErrorF32SSE2(
					&observed[0], &predicted[0], &got[0], length,
				)
				PredictionErrorFloat32Scalar(observed, predicted, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCUpdateRepresentationF32AVX2Parity(t *testing.T) {
	if !avx2PredictiveCodingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given PCUpdateRepresentationF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateRepresentationFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB327+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB328+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB329+int64(length))
				got := make([]float32, inDim)
				want := make([]float32, inDim)

				PCUpdateRepresentationF32AVX2(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateRepresentationFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCUpdateRepresentationF32SSE2Parity(t *testing.T) {
	if !sse2PredictiveCodingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given PCUpdateRepresentationF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateRepresentationFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB32A+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB32B+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB32C+int64(length))
				got := make([]float32, inDim)
				want := make([]float32, inDim)

				PCUpdateRepresentationF32SSE2(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateRepresentationFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCUpdateWeightsF32AVX2Parity(t *testing.T) {
	if !avx2PredictiveCodingAvailable() {
		t.Skip("AVX2+FMA required")
	}

	convey.Convey("Given PCUpdateWeightsF32AVX2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateWeightsFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB32D+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB32E+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB32F+int64(length))
				got := make([]float32, outDim*inDim)
				want := make([]float32, outDim*inDim)

				PCUpdateWeightsF32AVX2(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateWeightsFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}

func TestPCUpdateWeightsF32SSE2Parity(t *testing.T) {
	if !sse2PredictiveCodingAvailable() {
		t.Skip("SSE2 required")
	}

	convey.Convey("Given PCUpdateWeightsF32SSE2", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateWeightsFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB330+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB331+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB332+int64(length))
				got := make([]float32, outDim*inDim)
				want := make([]float32, outDim*inDim)

				PCUpdateWeightsF32SSE2(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateWeightsFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, predictiveCodingAVX2SSE2MaxULP)
			})
		}
	})
}
