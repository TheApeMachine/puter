package resonant

import "github.com/theapemachine/manifesto/dtype"

/*
ResonantUpdateForwardBFloat16 applies the coupled phasor update at native
bf16 precision.
*/
func ResonantUpdateForwardBFloat16(
	x, y, vr, vi, diag []uint16,
	xOut, yOut, aOut, bOut, invROut []uint16,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(x)
	invDim := resonantInvDimBF16(headDim)
	scaleValue := resonantScaleBF16(scale)
	oneMinusDamping := resonantOneMinusBF16(damping)

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := dtype.BF16(diag[headIndex*headDim+dimension])

		couplingReal := resonantMulBF16(dtype.BF16(vr[globalIndex]), invDim)
		couplingImag := resonantMulBF16(dtype.BF16(vi[globalIndex]), invDim)

		if zeroDiag {
			couplingReal = resonantSubBF16(
				couplingReal,
				resonantMulBF16(diagValue, dtype.BF16(x[globalIndex])),
			)
			couplingImag = resonantSubBF16(
				couplingImag,
				resonantMulBF16(diagValue, dtype.BF16(y[globalIndex])),
			)
		}

		accumReal := resonantAddBF16(
			resonantMulBF16(dtype.BF16(x[globalIndex]), oneMinusDamping),
			resonantMulBF16(scaleValue, couplingReal),
		)
		accumImag := resonantAddBF16(
			resonantMulBF16(dtype.BF16(y[globalIndex]), oneMinusDamping),
			resonantMulBF16(scaleValue, couplingImag),
		)
		inverseRadius := resonantInvRadiusBF16(accumReal, accumImag)

		xOut[globalIndex] = uint16(resonantMulBF16(accumReal, inverseRadius))
		yOut[globalIndex] = uint16(resonantMulBF16(accumImag, inverseRadius))
		aOut[globalIndex] = uint16(accumReal)
		bOut[globalIndex] = uint16(accumImag)
		invROut[globalIndex] = uint16(inverseRadius)
	}
}

/*
ResonantUpdateBackwardBFloat16 applies the autograd reverse pass at native
bf16 precision.
*/
func ResonantUpdateBackwardBFloat16(
	gradXOut, gradYOut, x, y, diag, a, b, invR []uint16,
	gradX, gradY, gradVR, gradVI []uint16,
	headCount, headDim int,
	scale, damping float32,
	zeroDiag bool,
) {
	elementCount := len(gradXOut)
	invDim := resonantInvDimBF16(headDim)
	scaleValue := resonantScaleBF16(scale)
	oneMinusDamping := resonantOneMinusBF16(damping)

	for globalIndex := 0; globalIndex < elementCount; globalIndex++ {
		dimension := globalIndex % headDim
		headIndex := (globalIndex / headDim) % headCount
		diagValue := dtype.BF16(diag[headIndex*headDim+dimension])

		inverseRadius := dtype.BF16(invR[globalIndex])
		inverseRadiusCubed := resonantMulBF16(
			resonantMulBF16(inverseRadius, inverseRadius),
			inverseRadius,
		)
		aValue := dtype.BF16(a[globalIndex])
		bValue := dtype.BF16(b[globalIndex])
		gradXOutValue := dtype.BF16(gradXOut[globalIndex])
		gradYOutValue := dtype.BF16(gradYOut[globalIndex])
		dotProduct := resonantAddBF16(
			resonantMulBF16(gradXOutValue, aValue),
			resonantMulBF16(gradYOutValue, bValue),
		)
		gradAccumReal := resonantSubBF16(
			resonantMulBF16(gradXOutValue, inverseRadius),
			resonantMulBF16(resonantMulBF16(aValue, dotProduct), inverseRadiusCubed),
		)
		gradAccumImag := resonantSubBF16(
			resonantMulBF16(gradYOutValue, inverseRadius),
			resonantMulBF16(resonantMulBF16(bValue, dotProduct), inverseRadiusCubed),
		)

		stateCoeff := oneMinusDamping
		if zeroDiag {
			stateCoeff = resonantSubBF16(stateCoeff, resonantMulBF16(scaleValue, diagValue))
		}

		gradX[globalIndex] = uint16(resonantMulBF16(gradAccumReal, stateCoeff))
		gradY[globalIndex] = uint16(resonantMulBF16(gradAccumImag, stateCoeff))
		gradVR[globalIndex] = uint16(resonantMulBF16(gradAccumReal, resonantMulBF16(scaleValue, invDim)))
		gradVI[globalIndex] = uint16(resonantMulBF16(gradAccumImag, resonantMulBF16(scaleValue, invDim)))

		_ = x[globalIndex]
		_ = y[globalIndex]
	}
}
