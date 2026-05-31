//go:build darwin && cgo

package optimizer

import (
	"errors"
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "core.h"

extern int metal_dispatch_optimizer4(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef firstRef,
    MetalBufferRef secondRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_optimizer3(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_optimizer2(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_hebbian_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef postRef,
    MetalBufferRef preRef,
    MetalBufferRef outRef,
    uint32_t postCount,
    uint32_t preCount,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
);

extern int metal_dispatch_lars_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef momentumRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t groupCount,
    const void* configBytes,
    size_t configBytesLen,
    uint64_t completionToken,
    MetalStatus* status
);
*/
import "C"

const (
	OperationAdam   = 0
	OperationAdamW  = 1
	OperationAdamax = 2
	OperationAdagrad = 3
	OperationRMSprop = 4
	OperationLion   = 5
	OperationSGD    = 6
	OperationLBFGS  = 7
)

var errUnsupportedDType = errors.New("metal optimizer: unsupported dtype")

type optimizer4Config struct {
	learningRate     float32
	beta1            float32
	beta2            float32
	epsilon          float32
	beta1Correction  float32
	beta2Correction  float32
	weightDecay      float32
}

type optimizer3Config struct {
	learningRate float32
	epsilon      float32
	decay        float32
	beta1        float32
	beta2        float32
	momentum     float32
}

type optimizer2Config struct {
	learningRate float32
}

type hebbianConfig struct {
	learningRate float32
	decay        float32
}

type larsConfig struct {
	learningRate float32
	momentum     float32
	weightDecay  float32
	trustCoeff   float32
	epsilon      float32
}

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.MetalElementDTypeFloat32
	case dtype.Float16:
		return C.MetalElementDTypeFloat16
	case dtype.BFloat16:
		return C.MetalElementDTypeBFloat16
	default:
		return -1
	}
}

func metalStatusError(status C.MetalStatus) error {
	if status.code == 0 {
		return nil
	}

	return tensor.ErrNeedsPlatformSetup
}

func betaCorrection(beta float32, step int) float32 {
	if step <= 0 {
		step = 1
	}

	return 1 - float32(math.Pow(float64(beta), float64(step)))
}

func packOptimizer4(config optimizer4Config) []byte {
	buffer := make([]byte, unsafe.Sizeof(config))
	*(*optimizer4Config)(unsafe.Pointer(&buffer[0])) = config
	return buffer
}

func packOptimizer3(config optimizer3Config) []byte {
	buffer := make([]byte, unsafe.Sizeof(config))
	*(*optimizer3Config)(unsafe.Pointer(&buffer[0])) = config
	return buffer
}

func packOptimizer2(config optimizer2Config) []byte {
	buffer := make([]byte, unsafe.Sizeof(config))
	*(*optimizer2Config)(unsafe.Pointer(&buffer[0])) = config
	return buffer
}

func packHebbian(config hebbianConfig) []byte {
	buffer := make([]byte, unsafe.Sizeof(config))
	*(*hebbianConfig)(unsafe.Pointer(&buffer[0])) = config
	return buffer
}

func packLARS(config larsConfig) []byte {
	buffer := make([]byte, unsafe.Sizeof(config))
	*(*larsConfig)(unsafe.Pointer(&buffer[0])) = config
	return buffer
}

func DispatchOptimizer4Refs(
	contextRef uintptr,
	operation int,
	paramsRef, gradientsRef, firstRef, secondRef, outRef uintptr,
	format dtype.DType,
	count uint32,
	config optimizer4Config,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || count == 0 {
		return errUnsupportedDType
	}

	configBytes := packOptimizer4(config)
	var status C.MetalStatus
	code := C.metal_dispatch_optimizer4(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.int(operation),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(paramsRef)),
		C.MetalBufferRef(unsafe.Pointer(gradientsRef)),
		C.MetalBufferRef(unsafe.Pointer(firstRef)),
		C.MetalBufferRef(unsafe.Pointer(secondRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		unsafe.Pointer(&configBytes[0]),
		C.size_t(len(configBytes)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchOptimizer3Refs(
	contextRef uintptr,
	operation int,
	paramsRef, gradientsRef, stateRef, outRef uintptr,
	format dtype.DType,
	count uint32,
	config optimizer3Config,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || count == 0 {
		return errUnsupportedDType
	}

	configBytes := packOptimizer3(config)
	var status C.MetalStatus
	code := C.metal_dispatch_optimizer3(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.int(operation),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(paramsRef)),
		C.MetalBufferRef(unsafe.Pointer(gradientsRef)),
		C.MetalBufferRef(unsafe.Pointer(stateRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		unsafe.Pointer(&configBytes[0]),
		C.size_t(len(configBytes)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchOptimizer2Refs(
	contextRef uintptr,
	operation int,
	paramsRef, gradientsRef, outRef uintptr,
	format dtype.DType,
	count uint32,
	config optimizer2Config,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || count == 0 {
		return errUnsupportedDType
	}

	configBytes := packOptimizer2(config)
	var status C.MetalStatus
	code := C.metal_dispatch_optimizer2(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		C.int(operation),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(paramsRef)),
		C.MetalBufferRef(unsafe.Pointer(gradientsRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		unsafe.Pointer(&configBytes[0]),
		C.size_t(len(configBytes)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchHebbianRefs(
	contextRef uintptr,
	weightsRef, postRef, preRef, outRef uintptr,
	format dtype.DType,
	postCount, preCount uint32,
	config cpuoptimizer.HebbianConfig,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || postCount == 0 || preCount == 0 {
		return errUnsupportedDType
	}

	configBytes := packHebbian(hebbianConfig{
		learningRate: config.LearningRate,
		decay:        config.Decay,
	})
	var status C.MetalStatus
	code := C.metal_dispatch_hebbian_step(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(weightsRef)),
		C.MetalBufferRef(unsafe.Pointer(postRef)),
		C.MetalBufferRef(unsafe.Pointer(preRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(postCount),
		C.uint32_t(preCount),
		unsafe.Pointer(&configBytes[0]),
		C.size_t(len(configBytes)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func DispatchLARSRefs(
	contextRef uintptr,
	paramsRef, gradientsRef, momentumRef, scratchRef, outRef uintptr,
	format dtype.DType,
	count, groupCount uint32,
	config cpuoptimizer.LARSConfig,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 || count == 0 || groupCount == 0 {
		return errUnsupportedDType
	}

	configBytes := packLARS(larsConfig{
		learningRate: config.LearningRate,
		momentum:     config.Momentum,
		weightDecay:  config.WeightDecay,
		trustCoeff:   config.TrustCoeff,
		epsilon:      config.Epsilon,
	})
	var status C.MetalStatus
	code := C.metal_dispatch_lars_step(
		C.MetalDeviceRef(unsafe.Pointer(contextRef)),
		elementFormat,
		C.MetalBufferRef(unsafe.Pointer(paramsRef)),
		C.MetalBufferRef(unsafe.Pointer(gradientsRef)),
		C.MetalBufferRef(unsafe.Pointer(momentumRef)),
		C.MetalBufferRef(unsafe.Pointer(scratchRef)),
		C.MetalBufferRef(unsafe.Pointer(outRef)),
		C.uint32_t(count),
		C.uint32_t(groupCount),
		unsafe.Pointer(&configBytes[0]),
		C.size_t(len(configBytes)),
		0,
		&status,
	)

	if code != 0 {
		return metalStatusError(status)
	}

	return nil
}

func AdamMetalConfig(config cpuoptimizer.AdamConfig) optimizer4Config {
	return optimizer4Config{
		learningRate:    config.LearningRate,
		beta1:             config.Beta1,
		beta2:             config.Beta2,
		epsilon:           config.Epsilon,
		beta1Correction:   betaCorrection(config.Beta1, config.Step),
		beta2Correction:   betaCorrection(config.Beta2, config.Step),
		weightDecay:       0,
	}
}

func AdamWMetalConfig(config cpuoptimizer.AdamWConfig) optimizer4Config {
	return optimizer4Config{
		learningRate:    config.LearningRate,
		beta1:             config.Beta1,
		beta2:             config.Beta2,
		epsilon:           config.Epsilon,
		beta1Correction:   betaCorrection(config.Beta1, config.Step),
		beta2Correction:   betaCorrection(config.Beta2, config.Step),
		weightDecay:       config.WeightDecay,
	}
}

func AdamaxMetalConfig(config cpuoptimizer.AdamaxConfig) optimizer4Config {
	return optimizer4Config{
		learningRate:    config.LearningRate,
		beta1:             config.Beta1,
		beta2:             config.Beta2,
		epsilon:           config.Epsilon,
		beta1Correction:   betaCorrection(config.Beta1, config.Step),
		beta2Correction:   1,
		weightDecay:       0,
	}
}

func AdagradMetalConfig(config cpuoptimizer.AdagradConfig) optimizer3Config {
	return optimizer3Config{
		learningRate: config.LearningRate,
		epsilon:      config.Epsilon,
	}
}

func RMSpropMetalConfig(config cpuoptimizer.RMSpropConfig) optimizer3Config {
	return optimizer3Config{
		learningRate: config.LearningRate,
		epsilon:      config.Epsilon,
		decay:        config.Decay,
	}
}

func LionMetalConfig(config cpuoptimizer.LionConfig) optimizer3Config {
	return optimizer3Config{
		learningRate: config.LearningRate,
		beta1:        config.Beta1,
		beta2:        config.Beta2,
	}
}

func SGDMetalConfig(config cpuoptimizer.SGDConfig) optimizer3Config {
	return optimizer3Config{
		learningRate: config.LearningRate,
		momentum:     config.Momentum,
	}
}

func LBFGSMetalConfig(config cpuoptimizer.LBFGSConfig) optimizer2Config {
	return optimizer2Config{learningRate: config.LearningRate}
}
