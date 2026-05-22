package optimizer

import "math"

type valueLoad func(index int) float32

type valueStore func(index int, value float32)

func adamMixedStep(
	config AdamConfig,
	count int,
	loadParam, loadGrad valueLoad,
	firstMoment, secondMoment []float32,
	storeOut valueStore,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

	for index := range count {
		gradValue := loadGrad(index)
		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		secondMoment[index] = config.Beta2*secondMoment[index] + (1-config.Beta2)*gradValue*gradValue

		biasCorrectedFirst := firstMoment[index] / beta1Correction
		biasCorrectedSecond := secondMoment[index] / beta2Correction

		denominator := float32(math.Sqrt(float64(biasCorrectedSecond))) + config.Epsilon
		paramValue := loadParam(index)

		storeOut(index, paramValue-config.LearningRate*biasCorrectedFirst/denominator)
	}
}

func adamWMixedStep(
	config AdamWConfig,
	count int,
	loadParam, loadGrad valueLoad,
	firstMoment, secondMoment []float32,
	storeOut valueStore,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
	beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)

		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue
		secondMoment[index] = config.Beta2*secondMoment[index] + (1-config.Beta2)*gradValue*gradValue

		biasCorrectedFirst := firstMoment[index] / beta1Correction
		biasCorrectedSecond := secondMoment[index] / beta2Correction

		denominator := float32(math.Sqrt(float64(biasCorrectedSecond))) + config.Epsilon
		gradStep := config.LearningRate * biasCorrectedFirst / denominator
		decayStep := config.LearningRate * config.WeightDecay * paramValue

		storeOut(index, paramValue-gradStep-decayStep)
	}
}

func lionMixedStep(
	config LionConfig,
	count int,
	loadParam, loadGrad valueLoad,
	momentum []float32,
	storeOut valueStore,
) {
	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)
		update := config.Beta1*momentum[index] + (1-config.Beta1)*gradValue

		var sign float32

		switch {
		case update > 0:
			sign = 1
		case update < 0:
			sign = -1
		}

		decayStep := config.WeightDecay * paramValue
		storeOut(index, paramValue-config.LearningRate*(sign+decayStep))

		momentum[index] = config.Beta2*momentum[index] + (1-config.Beta2)*gradValue
	}
}

func sgdMixedStep(
	config SGDConfig,
	count int,
	loadParam, loadGrad valueLoad,
	momentum []float32,
	storeOut valueStore,
) {
	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)
		effective := gradValue + config.WeightDecay*paramValue

		momentum[index] = config.Momentum*momentum[index] + effective

		update := momentum[index]

		if config.Nesterov {
			update = effective + config.Momentum*momentum[index]
		}

		storeOut(index, paramValue-config.LearningRate*update)
	}
}

func adamaxMixedStep(
	config AdamaxConfig,
	count int,
	loadParam, loadGrad valueLoad,
	firstMoment, infinityMoment []float32,
	storeOut valueStore,
) {
	beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))

	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)

		firstMoment[index] = config.Beta1*firstMoment[index] + (1-config.Beta1)*gradValue

		updated := config.Beta2 * infinityMoment[index]
		absGrad := float32(math.Abs(float64(gradValue)))

		if absGrad > updated {
			updated = absGrad
		}

		infinityMoment[index] = updated

		biasCorrectedFirst := firstMoment[index] / beta1Correction
		storeOut(index, paramValue-config.LearningRate*biasCorrectedFirst/(infinityMoment[index]+config.Epsilon))
	}
}

func adagradMixedStep(
	config AdagradConfig,
	count int,
	loadParam, loadGrad valueLoad,
	accumulator []float32,
	storeOut valueStore,
) {
	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)

		accumulator[index] += gradValue * gradValue
		denominator := float32(math.Sqrt(float64(accumulator[index]))) + config.Epsilon

		storeOut(index, paramValue-config.LearningRate*gradValue/denominator)
	}
}

func rmspropMixedStep(
	config RMSpropConfig,
	count int,
	loadParam, loadGrad valueLoad,
	secondMoment []float32,
	storeOut valueStore,
) {
	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)

		gradSquared := gradValue * gradValue
		scaledGrad := (1 - config.Decay) * gradSquared
		secondMoment[index] = scaledGrad + config.Decay*secondMoment[index]
		denominator := optimizerSqrtFloat32(secondMoment[index]) + config.Epsilon

		storeOut(index, paramValue-config.LearningRate*gradValue/denominator)
	}
}

func larsMixedStep(
	config LARSConfig,
	count int,
	loadParam, loadGrad valueLoad,
	momentum []float32,
	storeOut valueStore,
) {
	var paramsNorm, gradsNorm float64

	for index := range count {
		paramValue := loadParam(index)
		gradValue := loadGrad(index)

		paramsNorm += float64(paramValue) * float64(paramValue)
		gradsNorm += float64(gradValue) * float64(gradValue)
	}

	paramsNorm = math.Sqrt(paramsNorm)
	gradsNorm = math.Sqrt(gradsNorm)

	trust := float32(1.0)

	if paramsNorm > 0 && gradsNorm > 0 {
		trust = config.TrustCoeff *
			float32(paramsNorm) /
			(float32(gradsNorm) + config.WeightDecay*float32(paramsNorm) + config.Epsilon)
	}

	effectiveLr := config.LearningRate * trust

	for index := range count {
		gradValue := loadGrad(index)
		paramValue := loadParam(index)

		decayed := gradValue + config.WeightDecay*paramValue
		momentum[index] = config.Momentum*momentum[index] + decayed

		storeOut(index, paramValue-effectiveLr*momentum[index])
	}
}

func lbfgsMixedStep(
	config LBFGSConfig,
	count int,
	loadParam, loadGrad valueLoad,
	storeOut valueStore,
) {
	for index := range count {
		storeOut(index, loadParam(index)-config.LearningRate*loadGrad(index))
	}
}

func hebbianMixedStep(
	config HebbianConfig,
	loadWeight valueLoad,
	loadPost, loadPre valueLoad,
	storeOut valueStore,
	postCount, preDim int,
) {
	for postIndex := range postCount {
		postValue := loadPost(postIndex)

		for preIndex := range preDim {
			weightIndex := postIndex*preDim + preIndex
			preValue := loadPre(preIndex)
			weightValue := loadWeight(weightIndex)

			storeOut(weightIndex, weightValue*(1-config.Decay)+config.LearningRate*postValue*preValue)
		}
	}
}
