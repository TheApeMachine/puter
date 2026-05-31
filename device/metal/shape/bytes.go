package shape

import "github.com/theapemachine/manifesto/dtype"

/*
ElementByteSize returns the storage width for shape kernels.
*/
func ElementByteSize(format dtype.DType) int {
	switch format {
	case dtype.Float32, dtype.Int32:
		return 4
	case dtype.Float16, dtype.BFloat16:
		return 2
	default:
		return 0
	}
}
