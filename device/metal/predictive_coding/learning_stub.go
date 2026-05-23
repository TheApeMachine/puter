//go:build !darwin || !cgo

package predictive_coding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (predictiveCoding *PredictiveCoding) UpdateWeights(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType,) {
	predictiveCoding.stubHost()
}
