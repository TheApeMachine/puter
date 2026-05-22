//go:build amd64

package predictive_coding

//go:noescape
func PCPredictionFloat32AVX2Asm(
	weights, representation, output *float32,
	outDim, inDim int,
)

//go:noescape
func PCPredictionErrorFloat32AVX2Asm(
	observed, predicted, output *float32,
	count int,
)

//go:noescape
func PCUpdateRepresentationFloat32AVX2Asm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

//go:noescape
func PCUpdateWeightsFloat32AVX2Asm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

func PCPredictionF32AVX2(
	weights, representation, output *float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCPredictionFloat32AVX2Asm(weights, representation, output, outDim, inDim)
}

func PCPredictionErrorF32AVX2(
	observed, predicted, output *float32,
	count int,
) {
	if count == 0 {
		return
	}

	PCPredictionErrorFloat32AVX2Asm(observed, predicted, output, count)
}

func PCUpdateRepresentationF32AVX2(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateRepresentationFloat32AVX2Asm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}

func PCUpdateWeightsF32AVX2(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateWeightsFloat32AVX2Asm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}
