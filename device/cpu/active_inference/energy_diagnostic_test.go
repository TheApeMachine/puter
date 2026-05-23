//go:build arm64

package active_inference

import (
	"fmt"
	"math"
	"testing"

	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
)

func TestFreeEnergyEnergyDiagnostic(t *testing.T) {
	length := 1
	likelihood, posterior, prior := randomActiveInferenceVectors(length, 0xA301)

	scalar := FreeEnergyFloat32Scalar(likelihood, posterior, prior)
	asm := FreeEnergyFloat32NEONAsm(&likelihood[0], &posterior[0], &prior[0], length)

	clampedLike := float32(math.Max(activeInferenceEps, float64(likelihood[0])))
	clampedPost := float32(math.Max(activeInferenceEps, float64(posterior[0])))
	clampedPrior := float32(math.Max(activeInferenceEps, float64(prior[0])))

	aiLogLike := probeAiLog(clampedLike)
	loopLogLike := float32(0)
	AiNeonFeLogLikeProbeAsm(&likelihood[0], &loopLogLike)

	ceAcc := float64(0)
	klAcc := float64(0)
	_ = ceAcc
	_ = klAcc
	aiLogPost := probeAiLog(clampedPost)
	aiLogPrior := probeAiLog(clampedPrior)

	logLike := make([]float32, 1)
	logPost := make([]float32, 1)
	logPrior := make([]float32, 1)

	cpuactivation.LogF32NEON(&logLike[0], &clampedLike, 1)
	cpuactivation.LogF32NEON(&logPost[0], &clampedPost, 1)
	cpuactivation.LogF32NEON(&logPrior[0], &clampedPrior, 1)

	fromAiLog := freeEnergyFromLogs(aiLogLike, aiLogPost, aiLogPrior, posterior[0])
	fromF32Log := freeEnergyFromLogs(logLike[0], logPost[0], logPrior[0], posterior[0])

	wantCE := -float64(posterior[0]) * float64(loopLogLike)
	wantKL := float64(posterior[0]) * float64(logPost[0]-logPrior[0])

	fmt.Printf(
		"like=%g post=%g prior=%g\nscalar=%g asm=%g aiLog=%g f32log=%g\n"+
			"aiLogLike=%g loopLogLike=%g neonLogLike=%g mathLogLike=%g\n"+
			"wantCE=%g wantKL=%g\n"+
			"aiLogPost=%g neonLogPost=%g aiLogPrior=%g neonLogPrior=%g\n",
		likelihood[0], posterior[0], prior[0],
		scalar, asm, fromAiLog, fromF32Log,
		aiLogLike, loopLogLike, logLike[0], math.Log(float64(clampedLike)),
		wantCE, wantKL,
		aiLogPost, logPost[0], aiLogPrior, logPrior[0],
	)
}

func probeAiLog(value float32) float32 {
	out := float32(0)
	AiNeonLogProbeAsm(value, &out)

	return out
}

func freeEnergyFromLogs(
	logLike, logPost, logPrior float32,
	posterior float32,
) float32 {
	crossEntropy := -float64(posterior) * float64(logLike)
	kl := float64(posterior) * float64(logPost-logPrior)

	return float32(crossEntropy + kl)
}
