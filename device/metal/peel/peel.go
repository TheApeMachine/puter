package peel

/*
Peel implements device.Peel for the Metal backend.
*/
type Peel struct{}

/*
New constructs a Peel receiver.
*/
func New() Peel {
	return Peel{}
}

/*
SimdLaneCount reports the vector element alignment for Metal unary kernels.
Metal threadgroups operate on scalar threads; host-side peel uses width 4
for float32 vector loads in fused reference checks.
*/
func SimdLaneCount(isaName string) int {
	switch isaName {
	case "metal", "float32":
		return 4
	default:
		return 1
	}
}

/*
ReducedLaneCount reports the vector element alignment for reduced-precision
unary kernels on Metal (half/bfloat16 packed as ushort).
*/
func ReducedLaneCount(isaName string) int {
	switch isaName {
	case "metal", "float16", "bfloat16":
		return 8
	default:
		return 1
	}
}

func (peel Peel) SimdLaneCount(isaName string) int {
	return SimdLaneCount(isaName)
}

func (peel Peel) ReducedLaneCount(isaName string) int {
	return ReducedLaneCount(isaName)
}
