//go:build cuda

package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (predictiveCoding *PredictiveCoding) UpdateRepresentation(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictiveCoding.host.DispatchUpdateRepresentation(config, weights, representation, predictionError, output, outDim, inDim, format)
}
