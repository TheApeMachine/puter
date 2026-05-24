//go:build darwin && cgo

package random

/*
Normal writes `count` standard-normal float32 values into the workspace
pointer `dstRef`, seeded by (seed, counter). The output stream is
bitwise reproducible across CPU and Metal at the Philox level, and
within ≤ 8 ULP per lane after Box-Muller (Metal's native log/sin/cos
do not bit-match Go's F64 math).

The Metal kernel computes 4 Gaussian outputs per thread, advancing the
counter once per thread, so a kernel launch with N/4 threads produces
N Gaussians from counters (counter, counter+1, …, counter+N/4-1).
*/
func (random *Random) Normal(
	dstRef uintptr,
	count int,
	seed uint64,
	counter uint64,
) {
	if count <= 0 {
		return
	}

	if err := random.host.DispatchRandomNormal(dstRef, count, seed, counter, KernelNormalFloat32); err != nil {
		// Surface errors via panic; the device.Backend contract treats
		// kernel dispatch failures as non-recoverable at the call site.
		panic(err)
	}
}
