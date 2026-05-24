//go:build !darwin || !cgo

package random

func (random *Random) Normal(
	dstRef uintptr,
	count int,
	seed uint64,
	counter uint64,
) {
	random.stubHost()
}
