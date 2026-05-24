// Package random implements counter-based Gaussian random number
// generation for the Metal backend.
//
// The Metal path uses the same Philox-4×32-10 algorithm as the CPU
// scalar reference in device/cpu/random/, so the random uint32s
// produced for any (seed, counter) input are bitwise identical across
// CPU and Metal. The Box-Muller conversion that follows runs through
// Metal's native log/sin/cos/sqrt builtins; those are single-precision
// approximations that do not bitwise match Go's F64-then-cast scalar
// reference. Parity tests therefore assert tight ULP tolerance (≤ 4
// ULP per lane) on the final Gaussian outputs, not bitwise equality.
//
// Single-op family: Normal is the only public method. The package
// follows the same hub-and-domain quintet structure as the other
// Metal families (random.go + random.h + random.metal + native/random.m
// for the shared scaffold; normal.go + normal.h + normal.metal +
// native/normal.m for the domain kernel).
package random

/*
NormalKernel selects a Metal random kernel. Only F32 is implemented in
step 3; F16 and BF16 land in step 4.
*/
type NormalKernel int

const (
	KernelNormalFloat32 NormalKernel = iota
)

/*
Random implements device.Random for the Metal backend. (The
device.Random interface declaration lands in step 5, after all four
backends have real implementations.)
*/
type Random struct {
	host Host
}

/*
New wires a Random receiver to its Metal dispatch host.
*/
func New(host Host) Random {
	return Random{host: host}
}

/*
Host is the Metal dispatch surface random operations call into. The
backend Backend type provides this Host by implementing
DispatchRandomNormal via the cgo bridge.
*/
type Host interface {
	NeedsPlatform()
	DispatchRandomNormal(
		dstRef uintptr,
		count int,
		seed uint64,
		counter uint64,
		kernel NormalKernel,
	) error
}
