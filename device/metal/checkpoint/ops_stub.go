//go:build !darwin || !cgo

package checkpoint

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (checkpoint Checkpoint) CheckpointEncode(input, output unsafe.Pointer, format dtype.DType) {
	checkpoint.host.NeedsPlatform()
}

func (checkpoint Checkpoint) CheckpointDecode(input, output unsafe.Pointer, format dtype.DType) {
	checkpoint.host.NeedsPlatform()
}
