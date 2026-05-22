//go:build arm64

package predictive_coding

//go:noescape
func PCPredictionFloat32NEONAsm(
	weights, representation, output *float32,
	outDim, inDim int,
)

//go:noescape
func PCPredictionErrorFloat32NEONAsm(
	observed, predicted, output *float32,
	count int,
)

//go:noescape
func PCUpdateRepresentationFloat32NEONAsm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

//go:noescape
func PCUpdateWeightsFloat32NEONAsm(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
)

func PCPredictionF32NEON(
	weights, representation, output *float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCPredictionFloat32NEONAsm(weights, representation, output, outDim, inDim)
}

func PCPredictionErrorF32NEON(
	observed, predicted, output *float32,
	count int,
) {
	if count == 0 {
		return
	}

	PCPredictionErrorFloat32NEONAsm(observed, predicted, output, count)
}

func PCUpdateRepresentationF32NEON(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateRepresentationFloat32NEONAsm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}

func PCUpdateWeightsF32NEON(
	weights, representation, predictionError, output *float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateWeightsFloat32NEONAsm(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}
