//go:build cuda

package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (embedding *Embedding) TimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	embedding.host.LaunchTimestepEmbedding(config, timesteps, output, count, dim, format)
}
