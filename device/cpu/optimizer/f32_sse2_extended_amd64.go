//go:build amd64

package optimizer

import "math"

//go:noescape
func AdamaxStepFloat32SSE2Asm(
	params, grad, first, infinity, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction float32,
)

//go:noescape
func AdagradStepFloat32SSE2Asm(
	params, grad, accum, output *float32,
	count int,
	learningRate, epsilon float32,
)

//go:noescape
func RmspropStepFloat32SSE2Asm(
	params, grad, second, output *float32,
	count int,
	learningRate, decay, epsilon float32,
)

//go:noescape
func LionStepFloat32SSE2Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, beta1, beta2, weightDecay float32,
)

//go:noescape
func LbfgsStepFloat32SSE2Asm(
	params, grad, output *float32,
	count int,
	learningRate float32,
)

//go:noescape
func LarsStepFloat32SSE2Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, momentumFactor, weightDecay, effectiveLearningRate float32,
)

//go:noescape
func HebbianStepRowFloat32SSE2Asm(
	weights, pre, output *float32,
	count int,
	decayFactor, lrPost float32,
)

func adamaxStepSlicesSSE2(
	config AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamaxStepFloat32SSE2Asm(
		&params[0], &gradients[0], &firstMoment[0], &infinityMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon, beta1Correction,
	)
}

func adagradStepSlicesSSE2(
	config AdagradConfig,
	params, gradients, accumulator, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	AdagradStepFloat32SSE2Asm(
		&params[0], &gradients[0], &accumulator[0], &output[0],
		count, config.LearningRate, config.Epsilon,
	)
}

func rmspropStepSlicesSSE2(
	config RMSpropConfig,
	params, gradients, secondMoment, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	RmspropStepFloat32SSE2Asm(
		&params[0], &gradients[0], &secondMoment[0], &output[0],
		count, config.LearningRate, config.Decay, config.Epsilon,
	)
}

func lionStepSlicesSSE2(
	config LionConfig,
	params, gradients, momentum, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	LionStepFloat32SSE2Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count, config.LearningRate, config.Beta1, config.Beta2, config.WeightDecay,
	)
}

func lbfgsStepSlicesSSE2(config LBFGSConfig, params, gradients, output []float32) {
	count := len(params)

	if count == 0 {
		return
	}

	LbfgsStepFloat32SSE2Asm(
		&params[0], &gradients[0], &output[0],
		count, config.LearningRate,
	)
}

func larsStepSlicesSSE2(
	config LARSConfig,
	params, gradients, momentum, output []float32,
) {
	effectiveLearningRate := larsEffectiveLearningRate(config, params, gradients)
	count := len(params)

	if count == 0 {
		return
	}

	LarsStepFloat32SSE2Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count, config.LearningRate, config.Momentum, config.WeightDecay, effectiveLearningRate,
	)
}

func hebbianStepSlicesSSE2(
	config HebbianConfig,
	weights, post, pre, output []float32,
	preDim int,
) {
	decayFactor := float32(1 - config.Decay)

	for postIndex, postValue := range post {
		rowStart := postIndex * preDim
		weightsRow := weights[rowStart : rowStart+preDim]
		outRow := output[rowStart : rowStart+preDim]
		learningRatePost := config.LearningRate * postValue
		rowCount := len(weightsRow)

		if rowCount == 0 {
			continue
		}

		HebbianStepRowFloat32SSE2Asm(
			&weightsRow[0], &pre[0], &outRow[0],
			rowCount, decayFactor, learningRatePost,
		)
	}
}
