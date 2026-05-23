//go:build !cuda

package hawkes

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (hawkesProcess *Hawkes) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	hawkesProcess.stubHost()
}

func (hawkesProcess *Hawkes) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	hawkesProcess.stubHost()
}
