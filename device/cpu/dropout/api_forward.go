package dropout

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

var defaultDropoutLayer = New()

func Dropout(
	dst, src unsafe.Pointer,
	count int,
	config DropoutConfig,
	format dtype.DType,
) {
	defaultDropoutLayer.Dropout(dst, src, count, config, format)
}

func DropoutSeedState(seed uint64) [4]uint32 {
	return defaultDropoutLayer.DropoutSeedState(seed)
}
