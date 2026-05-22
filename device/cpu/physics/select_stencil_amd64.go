//go:build amd64

package physics

import "golang.org/x/sys/cpu"

func runLaplacian1DInterior(
	out, left, center, right []float32,
	invH2 float32,
	interiorCount int,
) int {
	if interiorCount <= 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		blockCount := interiorCount &^ 7

		if blockCount > 0 {
			Laplacian1DStencilF32AVX512Asm(
				&out[0], &left[0], &center[0], &right[0],
				invH2, blockCount,
			)
		}

		return blockCount
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		Laplacian1DStencilF32AVX2Asm(
			&out[0], &left[0], &center[0], &right[0],
			invH2, interiorCount,
		)

		return interiorCount
	}

	if cpu.X86.HasSSE2 {
		Laplacian1DStencilF32SSE2Asm(
			&out[0], &left[0], &center[0], &right[0],
			invH2, interiorCount,
		)

		return interiorCount
	}

	return 0
}

func runGrad1DInterior(
	out, left, right []float32,
	invTwoDx float32,
	interiorCount int,
) int {
	if interiorCount <= 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		blockCount := interiorCount &^ 7

		if blockCount > 0 {
			Grad1DStencilF32AVX512Asm(
				&out[0], &left[0], &right[0],
				invTwoDx, blockCount,
			)
		}

		return blockCount
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		Grad1DStencilF32AVX2Asm(
			&out[0], &left[0], &right[0],
			invTwoDx, interiorCount,
		)

		return interiorCount
	}

	if cpu.X86.HasSSE2 {
		Grad1DStencilF32SSE2Asm(
			&out[0], &left[0], &right[0],
			invTwoDx, interiorCount,
		)

		return interiorCount
	}

	return 0
}

func runLaplacian4Interior(
	out, um2, um1, u0, up1, up2 []float32,
	invDen float32,
	interiorCount int,
) int {
	if interiorCount <= 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		blockCount := interiorCount &^ 7

		if blockCount > 0 {
			Laplacian4StencilF32AVX512Asm(
				&out[0], &um2[0], &um1[0], &u0[0], &up1[0], &up2[0],
				invDen, blockCount,
			)
		}

		return blockCount
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		Laplacian4StencilF32AVX2Asm(
			&out[0], &um2[0], &um1[0], &u0[0], &up1[0], &up2[0],
			invDen, interiorCount,
		)

		return interiorCount
	}

	if cpu.X86.HasSSE2 {
		Laplacian4StencilF32SSE2Asm(
			&out[0], &um2[0], &um1[0], &u0[0], &up1[0], &up2[0],
			invDen, interiorCount,
		)

		return interiorCount
	}

	return 0
}
