package resonant

import "math"

const resonantNormalizeEpsilon = 1e-6

/*
ResonantUpdateForwardGeneric applies the coupled phasor update from
device/metal/resonant/resonant_update.metal.
*/
func ResonantUpdateForwardGeneric(
	x, y, vr, vi, diag []float32,
	xOut, yOut, aOut, bOut, invROut []float32,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(x)
	invDim := float32(1.0) / float32(headDim)
	oneMinusDamping := float32(1.0) - damping

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := diag[headIndex*headDim+dimension]

		couplingReal := vr[globalIndex] * invDim
		couplingImag := vi[globalIndex] * invDim

		if zeroDiag {
			couplingReal -= diagValue * x[globalIndex]
			couplingImag -= diagValue * y[globalIndex]
		}

		accumReal := x[globalIndex]*oneMinusDamping + scale*couplingReal
		accumImag := y[globalIndex]*oneMinusDamping + scale*couplingImag
		inverseRadius := float32(1.0) / float32(
			math.Sqrt(float64(accumReal*accumReal+accumImag*accumImag+resonantNormalizeEpsilon)),
		)

		xOut[globalIndex] = accumReal * inverseRadius
		yOut[globalIndex] = accumImag * inverseRadius
		aOut[globalIndex] = accumReal
		bOut[globalIndex] = accumImag
		invROut[globalIndex] = inverseRadius
	}
}

/*
ResonantUpdateBackwardGeneric applies the autograd reverse pass from
device/metal/resonant/resonant_update.metal.
*/
func ResonantUpdateBackwardGeneric(
	gradXOut, gradYOut, x, y, diag, a, b, invR []float32,
	gradX, gradY, gradVR, gradVI []float32,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(gradXOut)
	invDim := float32(1.0) / float32(headDim)
	oneMinusDamping := float32(1.0) - damping

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := diag[headIndex*headDim+dimension]

		inverseRadius := invR[globalIndex]
		inverseRadiusCubed := inverseRadius * inverseRadius * inverseRadius
		dotProduct := gradXOut[globalIndex]*a[globalIndex] + gradYOut[globalIndex]*b[globalIndex]
		gradAccumReal := gradXOut[globalIndex]*inverseRadius - a[globalIndex]*dotProduct*inverseRadiusCubed
		gradAccumImag := gradYOut[globalIndex]*inverseRadius - b[globalIndex]*dotProduct*inverseRadiusCubed

		stateCoeff := oneMinusDamping
		if zeroDiag {
			stateCoeff -= scale * diagValue
		}

		gradX[globalIndex] = gradAccumReal * stateCoeff
		gradY[globalIndex] = gradAccumImag * stateCoeff
		gradVR[globalIndex] = gradAccumReal * (scale * invDim)
		gradVI[globalIndex] = gradAccumImag * (scale * invDim)

		_ = x[globalIndex]
		_ = y[globalIndex]
	}
}
