package execution

import (
	"fmt"

	"github.com/theapemachine/puter/device"
)

func callMultiAxisRoPE(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 7 {
		return fmt.Errorf("router: MultiAxisRoPE expects 7 args, got %d", len(args))
	}

	config, err := castMultiAxisRoPEConfig(configFields)

	if err != nil {
		return err
	}

	input, output, err := castTwoPointers(args[:2], "MultiAxisRoPE")

	if err != nil {
		return err
	}

	batch, seqLen, numHeads, err := castThreeInts(args[2:5], "MultiAxisRoPE")

	if err != nil {
		return err
	}

	headDim, err := castInt(args[5], "MultiAxisRoPE", "headDim")

	if err != nil {
		return err
	}

	format, err := castDType(args[6], "MultiAxisRoPE", "format")

	if err != nil {
		return err
	}

	deviceBackend.MultiAxisRoPE(
		config,
		input, output,
		batch, seqLen, numHeads, headDim,
		format,
	)

	return nil
}

func castMultiAxisRoPEConfig(fields map[string]any) (device.MultiAxisRoPEConfig, error) {
	baseFreq, err := castFloat64Field(fields, "BaseFreq")

	if err != nil {
		return device.MultiAxisRoPEConfig{}, err
	}

	latentSeqLen, err := castIntField(fields, "LatentSeqLen")

	if err != nil {
		return device.MultiAxisRoPEConfig{}, err
	}

	latentSide, err := castIntField(fields, "LatentSide")

	if err != nil {
		return device.MultiAxisRoPEConfig{}, err
	}

	config := device.MultiAxisRoPEConfig{
		BaseFreq:     baseFreq,
		LatentSeqLen: latentSeqLen,
		LatentSide:   latentSide,
	}

	if err := config.Validate(); err != nil {
		return device.MultiAxisRoPEConfig{}, err
	}

	return config, nil
}
