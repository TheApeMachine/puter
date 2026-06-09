package execution

import "github.com/theapemachine/manifesto/ir"

/*
substituteLaunchDimensions rewrites physical tensor dimensions that still
carry planner upper-bound values into the live sizes supplied for this
graph.call invocation.

The static workspace keeps max-sized storage (e.g. N=4096). Launch
bindings carry the active prefix (e.g. N=42). Any dimension equal to the
planner max for a symbol is replaced with the live value so bind
resolution, device dispatch counts, and shape intrinsics operate on the
real sequence rather than the reserved tail.
*/
func substituteLaunchDimensions(
	physical []int,
	maxBindings ir.SymbolMap,
	launchBindings ir.SymbolMap,
) []int {
	substitutable := make([]bool, len(physical))

	for index := range substitutable {
		substitutable[index] = true
	}

	return substituteMarkedLaunchDimensions(physical, substitutable, maxBindings, launchBindings)
}

func substituteMarkedLaunchDimensions(
	physical []int,
	substitutable []bool,
	maxBindings ir.SymbolMap,
	launchBindings ir.SymbolMap,
) []int {
	if len(physical) == 0 || len(launchBindings) == 0 {
		return physical
	}

	result := append([]int(nil), physical...)

	for index, dimension := range result {
		if index >= len(substitutable) || !substitutable[index] {
			continue
		}

		for symbol, maxValue := range maxBindings {
			liveValue, hasLive := launchBindings[symbol]

			if !hasLive || maxValue == liveValue {
				continue
			}

			if int64(dimension) == maxValue {
				result[index] = int(liveValue)

				break
			}
		}
	}

	return result
}
