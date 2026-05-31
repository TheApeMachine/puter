package resonant

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestResonantUpdateForwardGeneric(t *testing.T) {
	convey.Convey("Given ResonantUpdateForwardGeneric", t, func() {
		configs := []struct {
			batchTime int
			headCount int
			headDim   int
		}{
			{batchTime: 1, headCount: 1, headDim: 1},
			{batchTime: 1, headCount: 1, headDim: 7},
			{batchTime: 4, headCount: 4, headDim: 4},
			{batchTime: 8, headCount: 8, headDim: 16},
			{batchTime: 8, headCount: 32, headDim: 32},
		}

		for _, config := range configs {
			config := config

			convey.Convey(
				fmt.Sprintf("BT=%d H=%d D=%d", config.batchTime, config.headCount, config.headDim),
				func() {
					elementCount := config.batchTime * config.headCount * config.headDim
					seed := int64(0x5200 + int64(elementCount))

					x, y, vr, vi, diag := randomResonantInputs(
						elementCount,
						config.headCount*config.headDim,
						seed,
					)
					xOut := make([]float32, elementCount)
					yOut := make([]float32, elementCount)
					aOut := make([]float32, elementCount)
					bOut := make([]float32, elementCount)
					invROut := make([]float32, elementCount)

					ResonantUpdateForwardGeneric(
						x, y, vr, vi, diag,
						xOut, yOut, aOut, bOut, invROut,
						config.headCount,
						config.headDim,
						0.25,
						0.1,
						true,
					)

					for index := range xOut {
						if !float32IsFinite(xOut[index]) {
							t.Fatalf("xOut[%d] not finite: %v", index, xOut[index])
						}

						if !float32IsFinite(yOut[index]) {
							t.Fatalf("yOut[%d] not finite: %v", index, yOut[index])
						}
					}
				},
			)
		}
	})
}

func TestResonantUpdateBackwardGeneric(t *testing.T) {
	convey.Convey("Given ResonantUpdateBackwardGeneric", t, func() {
		elementCount := 64
		headCount := 4
		headDim := 16

		x, y, vr, vi, diag := randomResonantInputs(elementCount, headCount*headDim, 0x5300)
		xOut := make([]float32, elementCount)
		yOut := make([]float32, elementCount)
		aOut := make([]float32, elementCount)
		bOut := make([]float32, elementCount)
		invROut := make([]float32, elementCount)

		ResonantUpdateForwardGeneric(
			x, y, vr, vi, diag,
			xOut, yOut, aOut, bOut, invROut,
			headCount,
			headDim,
			0.25,
			0.1,
			false,
		)

		gradXOut := randomResonantVector(elementCount, 0x5301)
		gradYOut := randomResonantVector(elementCount, 0x5302)
		gradX := make([]float32, elementCount)
		gradY := make([]float32, elementCount)
		gradVR := make([]float32, elementCount)
		gradVI := make([]float32, elementCount)

		ResonantUpdateBackwardGeneric(
			gradXOut, gradYOut,
			x, y, diag, aOut, bOut, invROut,
			gradX, gradY, gradVR, gradVI,
			headCount,
			headDim,
			0.25,
			0.1,
			false,
		)

		for index := range gradX {
			if !float32IsFinite(gradX[index]) {
				t.Fatalf("gradX[%d] not finite: %v", index, gradX[index])
			}
		}
	})
}

func randomResonantInputs(elementCount, diagCount int, seed int64) ([]float32, []float32, []float32, []float32, []float32) {
	return randomResonantVector(elementCount, seed),
		randomResonantVector(elementCount, seed+1),
		randomResonantVector(elementCount, seed+2),
		randomResonantVector(elementCount, seed+3),
		randomResonantVector(diagCount, seed+4)
}

func float32IsFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func randomResonantVector(length int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = rng.Float32()*2.0 - 1.0
	}

	return values
}
