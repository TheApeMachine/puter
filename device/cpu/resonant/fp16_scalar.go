package resonant

import "github.com/theapemachine/manifesto/dtype"

/*
ResonantUpdateForwardFloat16 applies the coupled phasor update at native
fp16 precision.
*/
func ResonantUpdateForwardFloat16(
	x, y, vr, vi, diag []uint16,
	xOut, yOut, aOut, bOut, invROut []uint16,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(x)
	invDim := resonantInvDimF16(headDim)
	scaleValue := resonantScaleF16(scale)
	oneMinusDamping := resonantOneMinusF16(damping)

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := dtype.F16(diag[headIndex*headDim+dimension])

		couplingReal := resonantMulF16(dtype.F16(vr[globalIndex]), invDim)
		couplingImag := resonantMulF16(dtype.F16(vi[globalIndex]), invDim)

		if zeroDiag {
			couplingReal = resonantSubF16(
				couplingReal,
				resonantMulF16(diagValue, dtype.F16(x[globalIndex])),
			)
			couplingImag = resonantSubF16(
				couplingImag,
				resonantMulF16(diagValue, dtype.F16(y[globalIndex])),
			)
		}

		accumReal := resonantAddF16(
			resonantMulF16(dtype.F16(x[globalIndex]), oneMinusDamping),
			resonantMulF16(scaleValue, couplingReal),
		)
		accumImag := resonantAddF16(
			resonantMulF16(dtype.F16(y[globalIndex]), oneMinusDamping),
			resonantMulF16(scaleValue, couplingImag),
		)
		inverseRadius := resonantInvRadiusF16(accumReal, accumImag)

		xOut[globalIndex] = uint16(resonantMulF16(accumReal, inverseRadius))
		yOut[globalIndex] = uint16(resonantMulF16(accumImag, inverseRadius))
		aOut[globalIndex] = uint16(accumReal)
		bOut[globalIndex] = uint16(accumImag)
		invROut[globalIndex] = uint16(inverseRadius)
	}
}

/*
ResonantUpdateBackwardFloat16 applies the autograd reverse pass at native
fp16 precision.
*/
func ResonantUpdateBackwardFloat16(
	gradXOut, gradYOut, x, y, diag, a, b, invR []uint16,
	gradX, gradY, gradVR, gradVI []uint16,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(gradXOut)
	invDim := resonantInvDimF16(headDim)
	scaleValue := resonantScaleF16(scale)
	oneMinusDamping := resonantOneMinusF16(damping)

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := dtype.F16(diag[headIndex*headDim+dimension])

		inverseRadius := dtype.F16(invR[globalIndex])
		inverseRadiusCubed := resonantMulF16(
			resonantMulF16(inverseRadius, inverseRadius),
			inverseRadius,
		)
		aValue := dtype.F16(a[globalIndex])
		bValue := dtype.F16(b[globalIndex])
		gradXOutValue := dtype.F16(gradXOut[globalIndex])
		gradYOutValue := dtype.F16(gradYOut[globalIndex])
		dotProduct := resonantAddF16(
			resonantMulF16(gradXOutValue, aValue),
			resonantMulF16(gradYOutValue, bValue),
		)
		gradAccumReal := resonantSubF16(
			resonantMulF16(gradXOutValue, inverseRadius),
			resonantMulF16(resonantMulF16(aValue, dotProduct), inverseRadiusCubed),
		)
		gradAccumImag := resonantSubF16(
			resonantMulF16(gradYOutValue, inverseRadius),
			resonantMulF16(resonantMulF16(bValue, dotProduct), inverseRadiusCubed),
		)

		stateCoeff := oneMinusDamping
		if zeroDiag {
			stateCoeff = resonantSubF16(stateCoeff, resonantMulF16(scaleValue, diagValue))
		}

		gradX[globalIndex] = uint16(resonantMulF16(gradAccumReal, stateCoeff))
		gradY[globalIndex] = uint16(resonantMulF16(gradAccumImag, stateCoeff))
		gradVR[globalIndex] = uint16(resonantMulF16(gradAccumReal, resonantMulF16(scaleValue, invDim)))
		gradVI[globalIndex] = uint16(resonantMulF16(gradAccumImag, resonantMulF16(scaleValue, invDim)))

		_ = x[globalIndex]
		_ = y[globalIndex]
	}
}
