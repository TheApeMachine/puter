//go:build arm64

package predictive_coding

import "testing"

func BenchmarkPCPredictionF32NEON(b *testing.B) {
	const benchDim = 1024
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xCC11)
	_, representation, _ := randomPredictiveCodingVectors(benchDim, 0xCC12)
	output := make([]float32, benchDim)

	for b.Loop() {
		PCPredictionF32NEON(
			&weights[0], &representation[0], &output[0], benchDim, benchDim,
		)
	}
}

func BenchmarkPCPredictionErrorF32NEON(b *testing.B) {
	const benchLen = 8192
	observed, predicted, _ := randomPredictiveCodingVectors(benchLen, 0xCC13)
	output := make([]float32, benchLen)

	for b.Loop() {
		PCPredictionErrorF32NEON(
			&observed[0], &predicted[0], &output[0], benchLen,
		)
	}
}

func BenchmarkPCUpdateRepresentationF32NEON(b *testing.B) {
	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xCC14)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xCC15)
	output := make([]float32, benchDim)

	for b.Loop() {
		PCUpdateRepresentationF32NEON(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}

func BenchmarkPCUpdateWeightsF32NEON(b *testing.B) {
	const benchDim = 64
	weights := randomPredictiveCodingWeights(benchDim, benchDim, 0xCC16)
	_, representation, predictionError := randomPredictiveCodingVectors(benchDim, 0xCC17)
	output := make([]float32, benchDim*benchDim)

	for b.Loop() {
		PCUpdateWeightsF32NEON(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			predictiveCodingTestLR, benchDim, benchDim,
		)
	}
}
