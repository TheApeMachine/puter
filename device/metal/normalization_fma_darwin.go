//go:build darwin && cgo

package metal

/*
#include <math.h>
*/
import "C"

/*
normMetalFMAFloat32 matches Metal fma(a,b,c) using libm fmaf.
*/
func normMetalFMAFloat32(a float32, b float32, c float32) float32 {
	return float32(C.fmaf(C.float(a), C.float(b), C.float(c)))
}
