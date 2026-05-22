//go:build amd64

package predictive_coding

//go:noescape
func PCPredictionFloat32SSE2Asm(
	weights, representation, output *float32,
	outDim, inDim int,
)

//go:noescape
func PCPredictionErrorFloat32SSE2Asm(
	observed, predicted, output *float32,
	count int,
)

//go:noescape
func PCUpdateRepresentationFloat32SSE2Asm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

//go:noescape
func PCUpdateWeightsFloat32SSE2Asm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

func PCPredictionF32SSE2(
	weights, representation, output *float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCPredictionFloat32SSE2Asm(weights, representation, output, outDim, inDim)
}

func PCPredictionErrorF32SSE2(
	observed, predicted, output *float32,
	count int,
) {
	if count == 0 {
		return
	}

	PCPredictionErrorFloat32SSE2Asm(observed, predicted, output, count)
}

func PCUpdateRepresentationF32SSE2(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateRepresentationFloat32SSE2Asm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}

func PCUpdateWeightsF32SSE2(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateWeightsFloat32SSE2Asm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}
