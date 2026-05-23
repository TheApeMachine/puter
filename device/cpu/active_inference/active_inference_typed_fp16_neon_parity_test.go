//go:build arm64

package active_inference

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestBeliefUpdateFP16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		likelihood := make([]dtype.F16, length)
		prior := make([]dtype.F16, length)
		want := make([]dtype.F16, length)
		got := make([]dtype.F16, length)
		for index := range likelihood {
			likelihood[index] = dtype.Fromfloat32(float32(index%17+1) * 0.1)
			prior[index] = dtype.Fromfloat32(float32(index%11+1) * 0.08)
		}
		BeliefUpdateFloat16Scalar(likelihood, prior, want)
		BeliefUpdateFP16NEON(likelihood, prior, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestPrecisionWeightFP16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		errorsVec := make([]dtype.F16, length)
		precision := make([]dtype.F16, length)
		want := make([]dtype.F16, length)
		got := make([]dtype.F16, length)
		for index := range errorsVec {
			errorsVec[index] = dtype.Fromfloat32(float32(index%7+1) * 0.2)
			precision[index] = dtype.Fromfloat32(float32(index%5+1) * 0.3)
		}
		PrecisionWeightFloat16Scalar(errorsVec, precision, want)
		PrecisionWeightFP16NEON(errorsVec, precision, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestFreeEnergyFP16NEONParity(t *testing.T) {
	for _, length := range parity.Lengths {
		likelihood := make([]dtype.F16, length)
		posterior := make([]dtype.F16, length)
		prior := make([]dtype.F16, length)
		for index := range likelihood {
			likelihood[index] = dtype.Fromfloat32(float32(index%13+1) * 0.05)
			posterior[index] = dtype.Fromfloat32(float32(index%9+1) * 0.07)
			prior[index] = dtype.Fromfloat32(float32(index%11+1) * 0.06)
		}
		want := FreeEnergyFloat16Scalar(likelihood, posterior, prior)
		got := FreeEnergyFP16NEON(likelihood, posterior, prior)
		if got != want {
			t.Fatalf("N=%d got=%v want=%v", length, got, want)
		}
	}
}

func TestExpectedFreeEnergyFP16NEONParity(t *testing.T) {
	for _, obsCount := range parity.Lengths {
		stateCount := obsCount/2 + 1
		predictedObs := make([]dtype.F16, obsCount)
		preferredObs := make([]dtype.F16, obsCount)
		predictedState := make([]dtype.F16, stateCount)
		for index := range predictedObs {
			predictedObs[index] = dtype.Fromfloat32(float32(index%17+1) * 0.04)
			preferredObs[index] = dtype.Fromfloat32(float32(index%13+1) * 0.05)
		}
		for index := range predictedState {
			predictedState[index] = dtype.Fromfloat32(float32(index%7+1) * 0.06)
		}
		want := ExpectedFreeEnergyFloat16Scalar(predictedObs, preferredObs, predictedState)
		got := ExpectedFreeEnergyFP16NEON(predictedObs, preferredObs, predictedState)
		if got != want {
			t.Fatalf("obs=%d state=%d got=%v want=%v", obsCount, stateCount, got, want)
		}
	}
}
