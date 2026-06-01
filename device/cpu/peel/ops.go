package peel

/*
Peel implements device.Peel for the CPU backend.
*/
type Peel struct{}

/*
New constructs a Peel receiver for CPU dispatch.
*/
func New() Peel {
	return Peel{}
}

func (peel Peel) SimdLaneCount(isaName string) int {
	switch isaName {
	case "avx512":
		return 16
	case "avx2":
		return 8
	case "sse2":
		return 4
	default:
		return 1
	}
}

func (peel Peel) ReducedLaneCount(isaName string) int {
	switch isaName {
	case "avx512":
		return 16
	case "avx2", "neon":
		return 8
	case "sse2":
		return 4
	default:
		return 1
	}
}
