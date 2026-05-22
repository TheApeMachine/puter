package metal

/*
hawkesIntensityMetalReference matches HawkesIntensityScalar with metal_hawkes_exp32 lanes.
*/
func hawkesIntensityMetalReference(
	eventTimes []float32,
	queryTimes []float32,
	mu float32,
	alpha float32,
	beta float32,
) []float32 {
	output := make([]float32, len(queryTimes))

	for queryIndex, queryTime := range queryTimes {
		intensity := mu

		for _, eventTime := range eventTimes {
			if eventTime > queryTime {
				continue
			}

			exponentArg := -beta * (queryTime - eventTime)
			intensity += alpha * hawkesExpMetalReference32(exponentArg)
		}

		output[queryIndex] = intensity
	}

	return output
}

/*
hawkesKernelMatrixMetalReference matches HawkesKernelMatrixScalar with metal_hawkes_exp32 lanes.
*/
func hawkesKernelMatrixMetalReference(
	eventTimes []float32,
	alpha float32,
	beta float32,
) []float32 {
	eventCount := len(eventTimes)
	output := make([]float32, eventCount*eventCount)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		for colIndex := 0; colIndex < eventCount; colIndex++ {
			if colIndex >= rowIndex {
				continue
			}

			delta := eventTimes[rowIndex] - eventTimes[colIndex]
			output[rowIndex*eventCount+colIndex] = alpha * hawkesExpMetalReference32(-beta*delta)
		}
	}

	return output
}
