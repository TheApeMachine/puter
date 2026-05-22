//go:build arm64

package predictive_coding

func PredictionFloat32Native(
	weights []float32,
	representation, output []float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCPredictionF32NEON(
		&weights[0], &representation[0], &output[0], outDim, inDim,
	)
}

func PredictionErrorFloat32Native(observed, predicted, output []float32) {
	if len(observed) == 0 {
		return
	}

	PCPredictionErrorF32NEON(
		&observed[0], &predicted[0], &output[0], len(observed),
	)
}

func UpdateRepresentationFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateRepresentationF32NEON(
		&weights[0], &representation[0], &predictionError[0], &output[0],
		learningRate, outDim, inDim,
	)
}

func UpdateWeightsFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	PCUpdateWeightsF32NEON(
		&weights[0], &representation[0], &predictionError[0], &output[0],
		learningRate, outDim, inDim,
	)
}
