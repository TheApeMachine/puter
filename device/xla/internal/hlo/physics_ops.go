package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderGrad1D(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	invTwoDx float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)
	normalizedLeft := (count - 1) % count
	normalizedRight := 1 % count

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  tail = %s slice(input), slice={[%d:%d]}
  head = %s slice(input), slice={[0:%d]}
  left = %s concatenate(tail, head), dimensions={0}
  tail_r = %s slice(input), slice={[%d:%d]}
  head_r = %s slice(input), slice={[0:%d]}
  right = %s concatenate(tail_r, head_r), dimensions={0}
  diff = %s subtract(right, left)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0}
  ROOT result = %s multiply(diff, scale_b)
}
`, moduleName, entryLayout,
		vectorLiteral,
		vectorLiteral, normalizedLeft, count,
		vectorLiteral, normalizedLeft, vectorLiteral,
		vectorLiteral, normalizedRight, count,
		vectorLiteral, normalizedRight, vectorLiteral,
		vectorLiteral, elementType, invTwoDx, vectorLiteral, vectorLiteral), nil
}

func RenderLaplacian1D(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	invH2 float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)
	normalizedLeft := (count - 1) % count
	normalizedRight := 1 % count

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  tail = %s slice(input), slice={[%d:%d]}
  head = %s slice(input), slice={[0:%d]}
  left = %s concatenate(tail, head), dimensions={0}
  tail_r = %s slice(input), slice={[%d:%d]}
  head_r = %s slice(input), slice={[0:%d]}
  right = %s concatenate(tail_r, head_r), dimensions={0}
  sum_lr = %s add(left, right)
  twice = %s add(input, input)
  diff = %s subtract(sum_lr, twice)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0}
  ROOT result = %s multiply(diff, scale_b)
}
`, moduleName, entryLayout,
		vectorLiteral,
		vectorLiteral, normalizedLeft, count,
		vectorLiteral, normalizedLeft, vectorLiteral,
		vectorLiteral, normalizedRight, count,
		vectorLiteral, normalizedRight, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, invH2, vectorLiteral, vectorLiteral), nil
}

func RenderLaplacian2D(
	moduleName string,
	elementFormat dtype.DType,
	rows, cols int,
	invH2 float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	matrixLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, rows, cols)
	entryLayout := fmt.Sprintf("%s->%s", matrixLiteral, matrixLiteral)
	four := float32(4)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  tail_up = %s slice(input), slice={[1:%d], [0:%d]}
  head_up = %s slice(input), slice={[0:1], [0:%d]}
  up = %s concatenate(tail_up, head_up), dimensions={0}
  tail_dn = %s slice(input), slice={[0:%d], [0:%d]}
  head_dn = %s slice(input), slice={[%d:%d], [0:%d]}
  down = %s concatenate(head_dn, tail_dn), dimensions={0}
  tail_l = %s slice(input), slice={[0:%d], [1:%d]}
  head_l = %s slice(input), slice={[0:%d], [0:1]}
  left = %s concatenate(tail_l, head_l), dimensions={1}
  tail_r = %s slice(input), slice={[0:%d], [0:%d]}
  head_r = %s slice(input), slice={[0:%d], [%d:%d]}
  right = %s concatenate(head_r, tail_r), dimensions={1}
  sum_ud = %s add(up, down)
  sum_lr = %s add(left, right)
  sum_all = %s add(sum_ud, sum_lr)
  four_c = %s[] constant(%g)
  four_b = %s broadcast(four_c), dimensions={0,1}
  scaled_center = %s multiply(input, four_b)
  diff = %s subtract(sum_all, scaled_center)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0,1}
  ROOT result = %s multiply(diff, scale_b)
}
`, moduleName, entryLayout,
		matrixLiteral,
		matrixLiteral, rows, cols, matrixLiteral, cols, matrixLiteral,
		matrixLiteral, rows-1, cols, matrixLiteral, rows-1, rows, cols, matrixLiteral,
		matrixLiteral, rows, cols, matrixLiteral, rows, matrixLiteral,
		matrixLiteral, rows, cols-1, matrixLiteral, rows, cols-1, cols, matrixLiteral,
		matrixLiteral, matrixLiteral, matrixLiteral,
		elementType, four, matrixLiteral, matrixLiteral, matrixLiteral,
		elementType, invH2, matrixLiteral, matrixLiteral), nil
}

func RenderLaplacian3D(
	moduleName string,
	elementFormat dtype.DType,
	depth, rows, cols int,
	invH2 float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	volumeLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, depth, rows, cols)
	entryLayout := fmt.Sprintf("%s->%s", volumeLiteral, volumeLiteral)
	six := float32(6)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  tail_dm = %s slice(input), slice={[1:%d], [0:%d], [0:%d]}
  head_dm = %s slice(input), slice={[0:1], [0:%d], [0:%d]}
  dm = %s concatenate(tail_dm, head_dm), dimensions={0}
  tail_dp = %s slice(input), slice={[0:%d], [0:%d], [0:%d]}
  head_dp = %s slice(input), slice={[%d:%d], [0:%d], [0:%d]}
  dp = %s concatenate(head_dp, tail_dp), dimensions={0}
  tail_rm = %s slice(input), slice={[0:%d], [1:%d], [0:%d]}
  head_rm = %s slice(input), slice={[0:%d], [0:1], [0:%d]}
  rm = %s concatenate(tail_rm, head_rm), dimensions={1}
  tail_rp = %s slice(input), slice={[0:%d], [0:%d], [0:%d]}
  head_rp = %s slice(input), slice={[0:%d], [%d:%d], [0:%d]}
  rp = %s concatenate(head_rp, tail_rp), dimensions={1}
  tail_cm = %s slice(input), slice={[0:%d], [0:%d], [1:%d]}
  head_cm = %s slice(input), slice={[0:%d], [0:%d], [0:1]}
  cm = %s concatenate(tail_cm, head_cm), dimensions={2}
  tail_cp = %s slice(input), slice={[0:%d], [0:%d], [0:%d]}
  head_cp = %s slice(input), slice={[0:%d], [0:%d], [%d:%d]}
  cp = %s concatenate(head_cp, tail_cp), dimensions={2}
  sum01 = %s add(dm, dp)
  sum23 = %s add(rm, rp)
  sum45 = %s add(cm, cp)
  sum0123 = %s add(sum01, sum23)
  sum_all = %s add(sum45, sum0123)
  six_c = %s[] constant(%g)
  six_b = %s broadcast(six_c), dimensions={0,1,2}
  scaled_center = %s multiply(input, six_b)
  diff = %s subtract(sum_all, scaled_center)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0,1,2}
  ROOT result = %s multiply(diff, scale_b)
}
`, moduleName, entryLayout,
		volumeLiteral,
		volumeLiteral, depth, rows, cols, volumeLiteral, rows, cols, volumeLiteral,
		volumeLiteral, depth-1, rows, cols, volumeLiteral, depth-1, depth, rows, cols, volumeLiteral,
		volumeLiteral, depth, rows, cols, volumeLiteral, depth, rows, volumeLiteral,
		volumeLiteral, depth, rows-1, cols, volumeLiteral, depth, rows-1, rows, cols, volumeLiteral,
		volumeLiteral, depth, rows, cols, volumeLiteral, depth, rows, volumeLiteral,
		volumeLiteral, depth, rows, cols-1, volumeLiteral, depth, rows, cols-1, cols, volumeLiteral,
		volumeLiteral, volumeLiteral, volumeLiteral, volumeLiteral, volumeLiteral,
		elementType, six, volumeLiteral, volumeLiteral, volumeLiteral,
		elementType, invH2, volumeLiteral, volumeLiteral), nil
}

func RenderLaplacian4(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	invDen float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)

	shifts := []int{-2, -1, 0, 1, 2}
	names := []string{"um2", "um1", "u0", "up1", "up2"}
	var shiftLines strings.Builder

	for index, shift := range shifts {
		normalized := shift % count

		if normalized < 0 {
			normalized += count
		}

		if shift == 0 {
			fmt.Fprintf(&shiftLines, "  %s = %s copy(input)\n", names[index], vectorLiteral)
			continue
		}

		fmt.Fprintf(&shiftLines, "  tail_%d = %s slice(input), slice={[%d:%d]}\n", index, vectorLiteral, normalized, count)
		fmt.Fprintf(&shiftLines, "  head_%d = %s slice(input), slice={[0:%d]}\n", index, vectorLiteral, normalized)
		fmt.Fprintf(&shiftLines, "  %s = %s concatenate(tail_%d, head_%d), dimensions={0}\n", names[index], vectorLiteral, index, index)
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
%s  c16 = %s[] constant(16)
  c30 = %s[] constant(-30)
  c1 = %s[] constant(-1)
  t16 = %s multiply(um1, %s broadcast(c16), dimensions={0})
  t30 = %s multiply(u0, %s broadcast(c30), dimensions={0})
  t16p = %s multiply(up1, %s broadcast(c16), dimensions={0})
  t1m = %s multiply(um2, %s broadcast(c1), dimensions={0})
  t1p = %s multiply(up2, %s broadcast(c1), dimensions={0})
  acc01 = %s add(t16, t30)
  acc23 = %s add(t16p, t1m)
  acc45 = %s add(t1p, acc01)
  acc = %s add(acc23, acc45)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0}
  ROOT result = %s multiply(acc, scale_b)
}
`, moduleName, entryLayout,
		vectorLiteral, shiftLines.String(),
		elementType, elementType, elementType,
		vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, invDen, vectorLiteral, vectorLiteral), nil
}

func RenderCentralDifferenceInterior(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	scale float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0}
  tail = %s slice(input), slice={[1:%d]}
  head = %s slice(input), slice={[0:%d]}
  left_pad = %s concatenate(zero_b, tail), dimensions={0}
  tail_r = %s slice(input), slice={[%d:%d]}
  right_pad = %s concatenate(tail_r, zero_b), dimensions={0}
  diff = %s subtract(right_pad, left_pad)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={0}
  scaled = %s multiply(diff, scale_b)
  one = %s[] constant(1)
  last = %s[] constant(%d)
  idx = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  interior = pred[%d]{0} compare(idx, one), direction=GE
  last_b = s32[%d]{0} broadcast(last), dimensions={0}
  not_last = pred[%d]{0} compare(idx, last_b), direction=LT
  valid = pred[%d]{0} and(interior, not_last)
  ROOT result = %s select(valid, scaled, zero_b)
}
`, moduleName, entryLayout,
		vectorLiteral, elementType, vectorLiteral,
		vectorLiteral, count-1, vectorLiteral, count-1, vectorLiteral,
		vectorLiteral, count-1, count, vectorLiteral,
		vectorLiteral, elementType, scale, vectorLiteral, vectorLiteral,
		elementType, elementType, count-1, count, count,
		count, count, count, count, vectorLiteral), nil
}

func RenderMadelungContinuity(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	invTwoDx float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	densityLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	velocityLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s,%s->%s", densityLiteral, velocityLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  density = %s parameter(0)
  velocity = %s parameter(1)
  flux = %s multiply(density, velocity)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0}
  tail = %s slice(flux), slice={[1:%d]}
  head = %s slice(flux), slice={[0:%d]}
  left_pad = %s concatenate(zero_b, tail), dimensions={0}
  tail_r = %s slice(flux), slice={[%d:%d]}
  right_pad = %s concatenate(tail_r, zero_b), dimensions={0}
  diff = %s subtract(right_pad, left_pad)
  scale = %s[] constant(%g)
  scale_b = %s broadcast(scale), dimensions={0}
  scaled = %s multiply(diff, scale_b)
  one = %s[] constant(1)
  last = %s[] constant(%d)
  idx = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  interior = pred[%d]{0} compare(idx, one), direction=GE
  last_b = s32[%d]{0} broadcast(last), dimensions={0}
  not_last = pred[%d]{0} compare(idx, last_b), direction=LT
  valid = pred[%d]{0} and(interior, not_last)
  ROOT result = %s select(valid, scaled, zero_b)
}
`, moduleName, entryLayout,
		densityLiteral, velocityLiteral, densityLiteral,
		elementType, outputLiteral,
		outputLiteral, count-1, outputLiteral, count-1, outputLiteral,
		outputLiteral, count-1, count, outputLiteral,
		outputLiteral, elementType, invTwoDx, outputLiteral, outputLiteral,
		elementType, elementType, count-1, count, count,
		count, count, count, count, outputLiteral), nil
}

func RenderQuantumPotential(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	invH2, scale float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)
	epsilon := float32(1e-12)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  density = %s parameter(0)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0}
  eps = %s[] constant(%g)
  tail = %s slice(density), slice={[1:%d]}
  head = %s slice(density), slice={[0:%d]}
  left_pad = %s concatenate(zero_b, tail), dimensions={0}
  tail_r = %s slice(density), slice={[%d:%d]}
  right_pad = %s concatenate(tail_r, zero_b), dimensions={0}
  left_clamped = %s maximum(left_pad, %s broadcast(eps), dimensions={0})
  right_clamped = %s maximum(right_pad, %s broadcast(eps), dimensions={0})
  center_clamped = %s maximum(density, %s broadcast(eps), dimensions={0})
  sqrt_left = %s sqrt(left_clamped)
  sqrt_right = %s sqrt(right_clamped)
  sqrt_center = %s sqrt(center_clamped)
  sum_lr = %s add(sqrt_left, sqrt_right)
  twice_center = %s add(sqrt_center, sqrt_center)
  lap = %s subtract(sum_lr, twice_center)
  inv = %s[] constant(%g)
  inv_b = %s broadcast(inv), dimensions={0}
  lap_scaled = %s multiply(lap, inv_b)
  ratio = %s divide(lap_scaled, sqrt_center)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={0}
  scaled = %s multiply(ratio, scale_b)
  one = %s[] constant(1)
  last = %s[] constant(%d)
  idx = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  interior = pred[%d]{0} compare(idx, one), direction=GE
  last_b = s32[%d]{0} broadcast(last), dimensions={0}
  not_last = pred[%d]{0} compare(idx, last_b), direction=LT
  valid = pred[%d]{0} and(interior, not_last)
  ROOT result = %s select(valid, scaled, zero_b)
}
`, moduleName, entryLayout,
		vectorLiteral, elementType, vectorLiteral, elementType, epsilon,
		vectorLiteral, count-1, vectorLiteral, count-1, vectorLiteral,
		vectorLiteral, count-1, count, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, invH2, vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, scale, vectorLiteral, vectorLiteral,
		elementType, elementType, count-1, count, count,
		count, count, count, count, vectorLiteral), nil
}

func RenderFFT1D(
	moduleName string,
	count int,
	inverse bool,
) (string, error) {
	if count <= 0 {
		return "", fmt.Errorf("fft requires positive count")
	}

	fftType := "FFT"

	if inverse {
		fftType = "IFFT"
	}

	realLiteral := fmt.Sprintf("f32[%d]{0}", count)
	imagLiteral := fmt.Sprintf("f32[%d]{0}", count)
	stackLiteral := fmt.Sprintf("f32[%d]{0}", count*2)
	complexLiteral := fmt.Sprintf("c64[%d]{0}", count)
	entryLayout := fmt.Sprintf("%s,%s->%s", realLiteral, imagLiteral, stackLiteral)

	inverseBlock := ""

	if inverse {
		inverseBlock = fmt.Sprintf(`
  inv_n = f32[] constant(%g)
  inv_b = f32[%d]{0} broadcast(inv_n), dimensions={0}
  real_scaled = f32[%d]{0} multiply(real_out, inv_b)
  imag_scaled = f32[%d]{0} multiply(imag_out, inv_b)
`, 1.0/float32(count), count, count, count)
	}

	realSource := "real_out"
	imagSource := "imag_out"

	if inverse {
		realSource = "real_scaled"
		imagSource = "imag_scaled"
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  real_in = %s parameter(0)
  imag_in = %s parameter(1)
  complex_in = %s complex(real_in, imag_in), shape=%s
  fft_out = %s fft(complex_in), fft_type=%s, fft_length={%d}
  real_out = f32[%d]{0} real(fft_out)
  imag_out = f32[%d]{0} imag(fft_out)
%s  ROOT result = f32[%d]{0} concatenate(%s, %s), dimensions={0}
}
`, moduleName, entryLayout,
		realLiteral, imagLiteral, complexLiteral, complexLiteral,
		complexLiteral, fftType, count, count, count,
		inverseBlock, count*2, realSource, imagSource), nil
}

func RenderVectorSliceCopy(
	moduleName string,
	elementFormat dtype.DType,
	totalCount, offset, length int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, totalCount)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, length)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  ROOT result = %s slice(input), slice={[%d:%d]}
}
`, moduleName, entryLayout, inputLiteral, outputLiteral, offset, offset+length), nil
}
