package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
)

func timestepDivisorFromManifestGraph(manifestGraph *ast.Graph) float64 {
	if manifestGraph == nil {
		return 1
	}

	for _, node := range manifestGraph.Nodes {
		if node.Op != "embedding.timestep" {
			continue
		}

		raw, ok := node.Attributes["timestep_divisor"]

		if !ok {
			continue
		}

		switch typed := raw.(type) {
		case int:
			if typed > 0 {
				return float64(typed)
			}
		case int64:
			if typed > 0 {
				return float64(typed)
			}
		case float64:
			if typed > 0 {
				return typed
			}
		case float32:
			if typed > 0 {
				return float64(typed)
			}
		}
	}

	return 1
}

func scaleTimestepProgramInput(value any, divisor float64) (any, error) {
	if divisor <= 0 || divisor == 1 {
		return value, nil
	}

	switch typed := value.(type) {
	case float32:
		return typed / float32(divisor), nil
	case float64:
		return float32(typed / divisor), nil
	case []float32:
		scaled := make([]float32, len(typed))

		for index, element := range typed {
			scaled[index] = element / float32(divisor)
		}

		return scaled, nil
	case []float64:
		scaled := make([]float32, len(typed))

		for index, element := range typed {
			scaled[index] = float32(element / divisor)
		}

		return scaled, nil
	default:
		return nil, fmt.Errorf("runner: cannot scale timestep input type %T", value)
	}
}
