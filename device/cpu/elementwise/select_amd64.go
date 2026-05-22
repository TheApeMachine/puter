//go:build amd64

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"golang.org/x/sys/cpu"
)

func AddFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func SubFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MulFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func DivFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MaxFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func MinFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float32,
	)
}

func AddFloat64Native(dst, left, right []float64) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float64,
	)
}

func AxpyFloat32Native(y, x []float32, alpha float32) {
	if len(y) == 0 {
		return
	}

	Axpy(
		unsafe.Pointer(&y[0]),
		unsafe.Pointer(&x[0]),
		len(y),
		alpha,
		dtype.Float32,
	)
}

func AbsFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func NegFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func SqrtFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func ReluFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float32)
}

func AddBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func SubBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MulBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func DivBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MaxBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func MinBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.BFloat16,
	)
}

func AbsBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func NegBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func SqrtBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func ReluBFloat16Native(dst, src []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.BFloat16)
}

func AddFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Add(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func SubFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Sub(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MulFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Mul(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func DivFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Div(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MaxFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Max(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func MinFloat16Native(dst, left, right []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Min(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(dst),
		dtype.Float16,
	)
}

func AbsFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Abs(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func NegFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Neg(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func SqrtFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	Sqrt(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

func ReluFloat16Native(dst, src []dtype.F16) {
	if len(dst) == 0 {
		return
	}

	ReLU(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), len(dst), dtype.Float16)
}

var (
	addF32Funcs = []f32BinaryKernelImpl{
		{AddF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{AddF32Generic, "generic", true},
	}
	subF32Funcs = []f32BinaryKernelImpl{
		{SubF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SubF32Generic, "generic", true},
	}
	mulF32Funcs = []f32BinaryKernelImpl{
		{MulF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MulF32Generic, "generic", true},
	}
	divF32Funcs = []f32BinaryKernelImpl{
		{DivF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{DivF32Generic, "generic", true},
	}
	maxF32Funcs = []f32BinaryKernelImpl{
		{MaxF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxF32Generic, "generic", true},
	}
	minF32Funcs = []f32BinaryKernelImpl{
		{MinF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinF32Generic, "generic", true},
	}
	absF32Funcs = []f32UnaryKernelImpl{
		{AbsF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{AbsF32Generic, "generic", true},
	}
	negF32Funcs = []f32UnaryKernelImpl{
		{NegF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{NegF32Generic, "generic", true},
	}
	sqrtF32Funcs = []f32UnaryKernelImpl{
		{SqrtF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{SqrtF32Generic, "generic", true},
	}
	reluF32Funcs = []f32UnaryKernelImpl{
		{ReluF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{ReluF32Generic, "generic", true},
	}
	axpyF32Funcs = []f32AxpyKernelImpl{
		{AxpyF32AVX512, "avx512", cpu.X86.HasAVX512F},
		{AxpyF32Generic, "generic", true},
	}
	addF64Funcs = []f64BinaryKernelImpl{{AddF64Generic, "generic", true}}
	addF16Funcs = []uint16BinaryKernelImpl{
		{AddF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AddF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AddF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AddF16Generic, "generic", true},
	}
	subF16Funcs = []uint16BinaryKernelImpl{
		{SubF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SubF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SubF16SSE2, "sse2", cpu.X86.HasSSE2},
		{SubF16Generic, "generic", true},
	}
	mulF16Funcs = []uint16BinaryKernelImpl{
		{MulF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MulF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MulF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MulF16Generic, "generic", true},
	}
	divF16Funcs = []uint16BinaryKernelImpl{
		{DivF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{DivF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{DivF16SSE2, "sse2", cpu.X86.HasSSE2},
		{DivF16Generic, "generic", true},
	}
	maxF16Funcs = []uint16BinaryKernelImpl{
		{MaxF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MaxF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MaxF16Generic, "generic", true},
	}
	minF16Funcs = []uint16BinaryKernelImpl{
		{MinF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MinF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MinF16Generic, "generic", true},
	}
	absF16Funcs = []uint16UnaryKernelImpl{
		{AbsF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AbsF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AbsF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AbsF16Generic, "generic", true},
	}
	negF16Funcs = []uint16UnaryKernelImpl{
		{NegF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{NegF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{NegF16SSE2, "sse2", cpu.X86.HasSSE2},
		{NegF16Generic, "generic", true},
	}
	sqrtF16Funcs = []uint16UnaryKernelImpl{
		{SqrtF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SqrtF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SqrtF16SSE2, "sse2", cpu.X86.HasSSE2},
		{SqrtF16Generic, "generic", true},
	}
	reluF16Funcs = []uint16UnaryKernelImpl{
		{ReluF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{ReluF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ReluF16SSE2, "sse2", cpu.X86.HasSSE2},
		{ReluF16Generic, "generic", true},
	}
	axpyF16Funcs = []uint16AxpyKernelImpl{
		{AxpyF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AxpyF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AxpyF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AxpyF16Generic, "generic", true},
	}
	addBF16Funcs = []uint16BinaryKernelImpl{
		{AddBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AddBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AddBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AddBF16Generic, "generic", true},
	}
	subBF16Funcs = []uint16BinaryKernelImpl{
		{SubBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SubBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SubBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{SubBF16Generic, "generic", true},
	}
	mulBF16Funcs = []uint16BinaryKernelImpl{
		{MulBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MulBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MulBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MulBF16Generic, "generic", true},
	}
	divBF16Funcs = []uint16BinaryKernelImpl{
		{DivBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{DivBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{DivBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{DivBF16Generic, "generic", true},
	}
	maxBF16Funcs = []uint16BinaryKernelImpl{
		{MaxBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MaxBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MaxBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MaxBF16Generic, "generic", true},
	}
	minBF16Funcs = []uint16BinaryKernelImpl{
		{MinBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{MinBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{MinBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{MinBF16Generic, "generic", true},
	}
	absBF16Funcs = []uint16UnaryKernelImpl{
		{AbsBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AbsBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AbsBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AbsBF16Generic, "generic", true},
	}
	negBF16Funcs = []uint16UnaryKernelImpl{
		{NegBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{NegBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{NegBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{NegBF16Generic, "generic", true},
	}
	sqrtBF16Funcs = []uint16UnaryKernelImpl{
		{SqrtBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{SqrtBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{SqrtBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{SqrtBF16Generic, "generic", true},
	}
	reluBF16Funcs = []uint16UnaryKernelImpl{
		{ReluBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{ReluBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{ReluBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{ReluBF16Generic, "generic", true},
	}
	axpyBF16Funcs = []uint16AxpyKernelImpl{
		{AxpyBF16AVX512, "avx512", cpu.X86.HasAVX512F},
		{AxpyBF16AVX2, "avx2", cpu.X86.HasAVX2 && cpu.X86.HasFMA},
		{AxpyBF16SSE2, "sse2", cpu.X86.HasSSE2},
		{AxpyBF16Generic, "generic", true},
	}
)
