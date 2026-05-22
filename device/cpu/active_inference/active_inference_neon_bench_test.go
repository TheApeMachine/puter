//go:build arm64

package active_inference

import "testing"

func BenchmarkPrecisionWeightF32NEON(b *testing.B) {
	errors, precision, _ := randomActiveInferenceVectors(8192, 0xA400)
	output := make([]float32, 8192)

	for b.Loop() {
		PrecisionWeightF32NEON(&errors[0], &precision[0], &output[0], len(errors))
	}
}

func BenchmarkBeliefUpdateF32NEON(b *testing.B) {
	likelihood, prior, _ := randomActiveInferenceVectors(8192, 0xA401)
	output := make([]float32, 8192)

	for b.Loop() {
		BeliefUpdateF32NEON(&likelihood[0], &prior[0], &output[0], len(likelihood))
	}
}

func BenchmarkFreeEnergyF32NEON(b *testing.B) {
	likelihood, posterior, prior := randomActiveInferenceVectors(8192, 0xA402)

	for b.Loop() {
		_ = FreeEnergyF32NEON(&likelihood[0], &posterior[0], &prior[0], len(likelihood))
	}
}

func BenchmarkExpectedFreeEnergyF32NEON(b *testing.B) {
	predictedObs, preferredObs, predictedState := randomActiveInferenceVectors(8192, 0xA403)

	for b.Loop() {
		_ = ExpectedFreeEnergyF32NEON(
			&predictedObs[0], &preferredObs[0], &predictedState[0],
			len(predictedObs), len(predictedState),
		)
	}
}

func BenchmarkFreeEnergyFloat32Scalar(b *testing.B) {
	likelihood, posterior, prior := randomActiveInferenceVectors(8192, 0xA404)

	for b.Loop() {
		_ = FreeEnergyFloat32Scalar(likelihood, posterior, prior)
	}
}
