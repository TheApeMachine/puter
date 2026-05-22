//go:build amd64

package optimizer

import "math"

//go:noescape
func AdamStepFloat32AVX2Asm(
	params, grad, first, second, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction, beta2Correction float32,
)

//go:noescape
func SgdStepFloat32AVX2Asm(
	params, grad, momentum, output *float32,
	count int,
	learningRate, momentumFactor, weightDecay float32,
)

//go:noescape
func AdamwStepFloat32AVX2Asm(
	params, grad, first, second, output *float32,
	count int,
	learningRate, beta1, beta2, epsilon, beta1Correction, beta2Correction, weightDecay float32,
)

func adamStepSlicesAVX2(
	config AdamConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamStepFloat32AVX2Asm(
		&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
		beta1Correction, beta2Correction,
	)
}

func adamWStepSlicesAVX2(
	config AdamWConfig,
	params, gradients, firstMoment, secondMoment, output []float32,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))
	count := len(params)

	if count == 0 {
		return
	}

	AdamwStepFloat32AVX2Asm(
		&params[0], &gradients[0], &firstMoment[0], &secondMoment[0], &output[0],
		count,
		config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
		beta1Correction, beta2Correction, config.WeightDecay,
	)
}

func sgdStepSlicesAVX2(
	config SGDConfig,
	params, gradients, momentum, output []float32,
) {
	count := len(params)

	if count == 0 {
		return
	}

	SgdStepFloat32AVX2Asm(
		&params[0], &gradients[0], &momentum[0], &output[0],
		count,
		config.LearningRate, config.Momentum, config.WeightDecay,
	)
}
