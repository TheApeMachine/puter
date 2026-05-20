//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math/rand/v2"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	computekernels "github.com/theapemachine/puter/kernels"
)

type metalSamplingConfig struct {
	logits       *metalTensor
	out          *metalTensor
	scores       *metalTensor
	indices      *metalTensor
	elementDType metalElementDType
	operation    metalSamplingOp
	count        uint32
	paddedCount  uint32
	target       float32
}

func runMetalSamplingWithConfig(
	operation metalSamplingOp,
	logits tensor.Tensor,
	out tensor.Tensor,
	config device.SamplingConfig,
) error {
	configCopy := config

	return runMetalSampling(operation, logits, out, &configCopy)
}

func runMetalSampling(
	operation metalSamplingOp,
	logits tensor.Tensor,
	out tensor.Tensor,
	config *device.SamplingConfig,
) error {
	metalConfig, err := requireMetalSampling(operation, logits, out, config)
	if err != nil {
		return err
	}

	if metalConfig.count == 0 {
		return nil
	}

	if err := metalConfig.allocateScratch(); err != nil {
		return err
	}

	token, err := metalConfig.beginCompletion()
	if err != nil {
		metalConfig.closeScratch()
		return err
	}
	metalConfig.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_sampling(
		metalConfig.logits.bridge.device,
		C.int(metalConfig.operation),
		C.int(metalConfig.elementDType),
		metalConfig.logits.buffer,
		metalConfig.scoreBuffer(),
		metalConfig.indexBuffer(),
		metalConfig.out.buffer,
		C.uint32_t(metalConfig.count),
		C.uint32_t(metalConfig.paddedCount),
		C.float(metalConfig.target),
		C.uint64_t(token),
		&status,
	)

	return finishMetalSamplingDispatch(token, rc, status)
}

func requireMetalSampling(
	operation metalSamplingOp,
	logits tensor.Tensor,
	out tensor.Tensor,
	config *device.SamplingConfig,
) (metalSamplingConfig, error) {
	logitTensor, outTensor, err := requireMetalSamplingTensors(logits, out)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(logitTensor.dtype)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	count, paddedCount, err := metalSamplingCounts(logitTensor)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	target := metalSamplingDefaultTarget()

	if config != nil && config.Seed != 0 {
		target = metalSamplingTargetFromSeed(config.Seed)
	}

	return metalSamplingConfig{
		logits: logitTensor, out: outTensor, elementDType: elementDType,
		operation: operation, count: count, paddedCount: paddedCount,
		target: target,
	}, nil
}

func requireMetalSamplingTensors(logits tensor.Tensor, out tensor.Tensor) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(logits, out)
	if err != nil {
		return nil, nil, err
	}

	logitTensor := tensors[0]
	outTensor := tensors[1]
	if outTensor.dtype != dtype.Int32 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if logitTensor.bridge != outTensor.bridge {
		return nil, nil, errors.New("metal sampling: tensors belong to different Metal backends")
	}

	if outTensor.shape.Len() < 1 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return logitTensor, outTensor, nil
}

func metalSamplingCounts(logits *metalTensor) (uint32, uint32, error) {
	count := logits.shape.Len()
	if count <= 0 || count > 1<<31 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	paddedCount := uint32(1)
	for paddedCount < uint32(count) {
		paddedCount <<= 1
	}

	return uint32(count), paddedCount, nil
}

func (config *metalSamplingConfig) allocateScratch() error {
	if config.operation == metalSamplingGreedy {
		return nil
	}

	shape, err := tensor.NewShape([]int{int(config.paddedCount)})
	if err != nil {
		return err
	}

	if err := config.allocateScoreScratch(shape); err != nil {
		return err
	}

	return config.allocateIndexScratch(shape)
}

func (config *metalSamplingConfig) allocateScoreScratch(shape tensor.Shape) error {
	var err error
	config.scores, err = config.logits.bridge.empty(shape, dtype.Float32)

	return err
}

func (config *metalSamplingConfig) allocateIndexScratch(shape tensor.Shape) error {
	var err error
	config.indices, err = config.logits.bridge.empty(shape, dtype.Int32)
	if err == nil {
		return nil
	}

	_ = config.scores.Close()
	return err
}

func (config *metalSamplingConfig) beginCompletion() (uint64, error) {
	if config.operation == metalSamplingGreedy {
		return metalCompletions.Begin(config.out, config.logits)
	}

	return metalCompletions.BeginMany(
		[]*metalTensor{config.out, config.scores, config.indices},
		config.logits,
	)
}

func (config *metalSamplingConfig) closeScratch() {
	if config.scores != nil {
		_ = config.scores.Close()
	}

	if config.indices != nil {
		_ = config.indices.Close()
	}
}

func (config *metalSamplingConfig) scoreBuffer() C.MetalBufferRef {
	if config.scores == nil {
		return nil
	}

	return config.scores.buffer
}

func (config *metalSamplingConfig) indexBuffer() C.MetalBufferRef {
	if config.indices == nil {
		return nil
	}

	return config.indices.buffer
}

func finishMetalSamplingDispatch(token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal sampling: %s", metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}

func metalSamplingDefaultTarget() float32 {
	config := computekernels.DefaultSamplingConfig()

	return metalSamplingTargetFromSeed(config.Seed)
}

func metalSamplingTargetFromSeed(seed uint64) float32 {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source).Float32()
}
