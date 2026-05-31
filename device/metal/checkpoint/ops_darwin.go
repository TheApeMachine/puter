//go:build darwin && cgo

package checkpoint

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (checkpoint Checkpoint) CheckpointEncode(input, output unsafe.Pointer, format dtype.DType) {
	checkpoint.host.DispatchCheckpointEncode(input, output, format)
}

func (checkpoint Checkpoint) CheckpointDecode(input, output unsafe.Pointer, format dtype.DType) {
	checkpoint.host.DispatchCheckpointDecode(input, output, format)
}
