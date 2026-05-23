//go:build arm64

package active_inference

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestBeliefUpdateBF16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		likelihood := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.1)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.08)
		}
		BeliefUpdateBFloat16Scalar(likelihood, prior, want)
		BeliefUpdateBF16NEON(likelihood, prior, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestPrecisionWeightBF16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		errorsVec := make([]dtype.BF16, length)
		precision := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range errorsVec {
			errorsVec[index] = dtype.NewBfloat16FromFloat32(float32(index%7+1) * 0.2)
			precision[index] = dtype.NewBfloat16FromFloat32(float32(index%5+1) * 0.3)
		}
		PrecisionWeightBFloat16Scalar(errorsVec, precision, want)
		PrecisionWeightBF16NEON(errorsVec, precision, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestFreeEnergyBF16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		likelihood := make([]dtype.BF16, length)
		posterior := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%13+1) * 0.05)
			posterior[index] = dtype.NewBfloat16FromFloat32(float32(index%9+1) * 0.07)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.06)
		}
		want := FreeEnergyBFloat16Scalar(likelihood, posterior, prior)
		got := FreeEnergyBF16NEON(likelihood, posterior, prior)
		if got != want {
			t.Fatalf("N=%d got=%v want=%v", length, got, want)
		}
	}
}

func TestExpectedFreeEnergyBF16NEONParity(t *testing.T) {
	for _, obsCount := range parity.Lengths {
		stateCount := obsCount/2 + 1
		predictedObs := make([]dtype.BF16, obsCount)
		preferredObs := make([]dtype.BF16, obsCount)
		predictedState := make([]dtype.BF16, stateCount)
		for index := range predictedObs {
			predictedObs[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.04)
			preferredObs[index] = dtype.NewBfloat16FromFloat32(float32(index%13+1) * 0.05)
		}
		for index := range predictedState {
			predictedState[index] = dtype.NewBfloat16FromFloat32(float32(index%7+1) * 0.06)
		}
		want := ExpectedFreeEnergyBFloat16Scalar(predictedObs, preferredObs, predictedState)
		got := ExpectedFreeEnergyBF16NEON(predictedObs, preferredObs, predictedState)
		if got != want {
			t.Fatalf("obs=%d state=%d got=%v want=%v", obsCount, stateCount, got, want)
		}
	}
}
