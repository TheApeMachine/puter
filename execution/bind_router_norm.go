package execution

import (
	"fmt"

	"github.com/theapemachine/puter/device"
)

func callModulatedLayerNorm(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 8 {
		return fmt.Errorf("router: ModulatedLayerNorm expects 8 args, got %d", len(args))
	}

	config, err := castModulatedLayerNormConfig(configFields)

	if err != nil {
		return err
	}

	input, modulation, output, err := castThreePointers(args[:3], "ModulatedLayerNorm")

	if err != nil {
		return err
	}

	if err := requireDispatchPointers("ModulatedLayerNorm", input, modulation, output); err != nil {
		return err
	}

	rows, lastDim, rowsPerBatch, err := castThreeInts(args[3:6], "ModulatedLayerNorm")

	if err != nil {
		return err
	}

	modulationCols, err := castInt(args[6], "ModulatedLayerNorm", "modulationCols")

	if err != nil {
		return err
	}

	format, err := castDType(args[7], "ModulatedLayerNorm", "format")

	if err != nil {
		return err
	}

	deviceBackend.ModulatedLayerNorm(
		config,
		input, modulation, output,
		rows, lastDim, rowsPerBatch, modulationCols,
		format,
	)

	return nil
}

func castModulatedLayerNormConfig(
	fields map[string]any,
) (device.ModulatedLayerNormConfig, error) {
	epsilon, err := castFloat64Field(fields, "Epsilon")

	if err != nil {
		return device.ModulatedLayerNormConfig{}, err
	}

	set, err := castIntField(fields, "Set")

	if err != nil {
		return device.ModulatedLayerNormConfig{}, err
	}

	config := device.ModulatedLayerNormConfig{
		Epsilon: epsilon,
		Set:     set,
	}

	if err := config.Validate(); err != nil {
		return device.ModulatedLayerNormConfig{}, err
	}

	return config, nil
}
