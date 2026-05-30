package device

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/optimizer"
)

type PosPop interface {
	Count8(counts *[8]int, buf []uint8)
	Count16(counts *[16]int, buf []uint16)
	Count32(counts *[32]int, buf []uint32)
	Count64(counts *[64]int, buf []uint64)
	CountString(counts *[8]int, str string)
}

type Activation interface {
	CELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	ELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	Exp(dst, src unsafe.Pointer, count int, format dtype.DType)
	Expm1(dst, src unsafe.Pointer, count int, format dtype.DType)
	GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	Gelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32)
	LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	Log(dst, src unsafe.Pointer, count int, format dtype.DType)
	Log1p(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType)
	Mish(dst, src unsafe.Pointer, count int, format dtype.DType)
	PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int)
	QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32)
	SELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	Silu(dst, src unsafe.Pointer, count int, format dtype.DType)
	SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32)
	Softmax(dst, src unsafe.Pointer, count int, format dtype.DType)
	Softplus(dst, src unsafe.Pointer, count int, format dtype.DType)
	SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	Softsign(dst, src unsafe.Pointer, count int, format dtype.DType)
	SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	Swish(dst, src unsafe.Pointer, count int, format dtype.DType)
	Tanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType)
	Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32)
}

type ActiveInference interface {
	BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType)
	ExpectedFreeEnergy(
		predictedObs, preferredObs, predictedState, output unsafe.Pointer,
		obsCount, stateCount int,
		format dtype.DType,
	)
	FreeEnergy(
		likelihood, posterior, prior, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType)
}

type Attention interface {
	FlashAttention(
		config FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
	MultiHeadAttention(
		config MultiHeadAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK int,
		format dtype.DType,
	)
	ScaledDotProductAttention(
		config FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
}

type Causal interface {
	BackdoorAdjustment(
		conditional, marginalZ, output unsafe.Pointer,
		xCount, zCount, yCount int,
		format dtype.DType,
	)
	CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType)
	Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType)
	Counterfactual(
		observedY, observedX, counterfactualX, output unsafe.Pointer,
		count int,
		slope float32,
		format dtype.DType,
	)
	DAGMarkovFactorization(
		conditionals unsafe.Pointer,
		conditionalCount int,
		output unsafe.Pointer,
		format dtype.DType,
	)
	DoIntervene(
		adjacency, intervened, output unsafe.Pointer,
		nodeCount, intervenedCount int,
		format dtype.DType,
	)
	FrontdoorAdjustment(
		mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
		xCount, mediatorCount, yCount int,
		format dtype.DType,
	)
	IVEstimate(
		instrument, treatment, outcome unsafe.Pointer,
		count int,
		output unsafe.Pointer,
		format dtype.DType,
	)
	MarkovFlowActive(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	MarkovFlowInternal(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
}

type Checkpoint interface {
	CheckpointDecode(input, output unsafe.Pointer, format dtype.DType)
	CheckpointEncode(input, output unsafe.Pointer, format dtype.DType)
}

type Convolution interface {
	Conv1D(
		config Conv1DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inLength, outChannels, kernelLength, outLength int,
		format dtype.DType,
	)
	Conv2D(
		config Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
	Conv3D(
		config Conv3DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW int,
		format dtype.DType,
	)
	ConvTranspose2D(
		config Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
}

type Dequant interface {
	Dequant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType)
	Dequant4(dst, src unsafe.Pointer, pairCount int, config DequantInt4Config, dstFormat, srcFormat dtype.DType)
}

type Dot interface {
	// Dot writes the inner product of `left` and `right` into `*dst`.
	// Zero-host-sync per ARCHITECTURE.md §2.2.
	Dot(dst, left, right unsafe.Pointer, count int, format dtype.DType)
}

type Dropout interface {
	Dropout(
		dst, src unsafe.Pointer,
		count int,
		config DropoutConfig,
		format dtype.DType,
	)
}

type Elementwise interface {
	Abs(dst, src unsafe.Pointer, count int, format dtype.DType)
	Add(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType)
	Div(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Max(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Min(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Neg(dst, src unsafe.Pointer, count int, format dtype.DType)
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType)
}

type Embedding interface {
	Bag(
		table, indices, offsets, output unsafe.Pointer,
		vocab, hidden, bagCount, indexCount int,
		format dtype.DType,
	)
	Lookup(
		table, indices, output unsafe.Pointer,
		vocab, hidden, indexCount int,
		format dtype.DType,
	)
	TimestepEmbedding(
		config TimestepEmbeddingConfig,
		timesteps, output unsafe.Pointer,
		count, dim int,
		format dtype.DType,
	)
}

type Hawkes interface {
	HawkesIntensity(
		eventTimes, queryTimes, output unsafe.Pointer,
		eventCount, queryCount int,
		mu, alpha, beta float32,
		format dtype.DType,
	)
	HawkesKernelMatrix(
		eventTimes, output unsafe.Pointer,
		eventCount int,
		alpha, beta float32,
		format dtype.DType,
	)
	HawkesLogLikelihood(
		eventTimes unsafe.Pointer,
		eventCount int,
		totalT, mu, alpha, beta float32,
		output unsafe.Pointer,
		format dtype.DType,
	)
	MarkovBlanketPartition(
		adjacency, internal, output unsafe.Pointer,
		nodeCount, internalCount int,
		format dtype.DType,
	)
	MarkovMutualInformation(
		joint, output unsafe.Pointer,
		xCount, yCount int,
		format dtype.DType,
	)
}

type Geometry interface {
	PhaseCoupling(
		destination, leftGrowth, rightGrowth unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	PhaseVelocity(
		destination, current, previous unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	GeometricProduct(
		destination, left, right unsafe.Pointer,
	)
	PhaseDialNormalize(
		dial unsafe.Pointer,
	)
	PhaseDialSimilarity(
		destination, left, right unsafe.Pointer,
	)
	PhaseDialRotate(
		destination, source, cosine, sine unsafe.Pointer,
	)
	PhaseDialAddPhases(
		dial, cosines, sines unsafe.Pointer,
	)
	PhaseDialComposeMidpoint(
		destination, left, right unsafe.Pointer,
	)
	PhaseRotorSimilarity(
		destination, left, right unsafe.Pointer,
	)
	EigenToroidalFromTags(
		phaseDestination, frequencyDestination, tags unsafe.Pointer,
		tagCount, windowSize int,
	)
	EigenCircularMeanPhase(
		destination, phaseTable, sequence unsafe.Pointer,
		sequenceLength int,
	)
}

type Interpretability interface {
	ActivationSteer(
		destination, base, direction unsafe.Pointer,
		coefficient float32,
		count int,
		format dtype.DType,
	)
}

type LayerNorm interface {
	AdaptiveRMSNorm(
		config RMSNormConfig,
		input, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols int,
		format dtype.DType,
	)
	LayerNorm(
		input, scale, bias, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
	ModulatedLayerNorm(
		config ModulatedLayerNormConfig,
		input, modulation, output unsafe.Pointer,
		rows, lastDim, rowsPerBatch, modulationCols int,
		format dtype.DType,
	)
	RMSNorm(
		config RMSNormConfig,
		input, scale, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
}

type Losses interface {
	// Each loss writes its scalar result into `*dst` (ARCHITECTURE.md §2.2).
	BinaryCrossEntropy(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
	CrossEntropy(
		dst unsafe.Pointer,
		logits unsafe.Pointer,
		targets unsafe.Pointer,
		batchSize, classes int,
		format dtype.DType,
	)
	Huber(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
	KLDivergence(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
	MAE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
	MSE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
}

type Masking interface {
	ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
	ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType)
	CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
}

type Math interface {
	InvSqrtDimScale(out, input unsafe.Pointer, dim int32, format dtype.DType)
	LogSumExp(input, output unsafe.Pointer, cols int, format dtype.DType)
	Outer(left, right, output unsafe.Pointer, leftCount, rightCount int, format dtype.DType)
}

type Matmul interface {
	Matmul(
		out, left, right unsafe.Pointer,
		rows, inner, cols int,
		format dtype.DType,
	)
}

type ModelEditing interface {
	WeightGraftAdd(weights, injection unsafe.Pointer, count int, format dtype.DType)
}

type Normalization interface {
	BatchNormDenorm(
		input, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	BatchNormEval(
		input, scale, bias, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	GroupNorm(
		config GroupNormConfig,
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	InstanceNorm(
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
}

type Optimizer interface {
	Adagrad(
		config optimizer.AdagradConfig,
		params, gradients, accumulator, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	Adam(
		config optimizer.AdamConfig,
		params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	Adamax(
		config optimizer.AdamaxConfig,
		params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	AdamW(
		config optimizer.AdamWConfig,
		params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	Hebbian(
		config optimizer.HebbianConfig,
		weights, post, pre, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	LARS(
		config optimizer.LARSConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	LBFGS(
		config optimizer.LBFGSConfig,
		params, gradients, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	Lion(
		config optimizer.LionConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	RMSprop(
		config optimizer.RMSpropConfig,
		params, gradients, secondMoment, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	SGD(
		config optimizer.SGDConfig,
		params, gradients, momentum, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
}

type Peel interface {
	ReducedLaneCount(isaName string) int
	SimdLaneCount(isaName string) int
}

type Physics interface {
	BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType)
	Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	MadelungContinuity(
		density, velocity, residual unsafe.Pointer,
		count int,
		spacing float32,
		format dtype.DType,
	)
	QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
}

type Pool interface {
	AdaptiveAvgPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	AdaptiveMaxPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	AvgPool2D(
		config PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	MaxPool2D(
		config PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
}

type PredictiveCoding interface {
	Prediction(
		weights, representation, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	PredictionError(
		observed, predicted, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	UpdateRepresentation(
		config PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	UpdateWeights(
		config PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
}

type Quant interface {
	Quant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType)
}

type Reduction interface {
	L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType)
	Prod(dst, values unsafe.Pointer, count int, format dtype.DType)
	ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType)
	ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType)
	Sum(dst, values unsafe.Pointer, count int, format dtype.DType)
}

type RoPE interface {
	MultiAxisRoPE(
		config MultiAxisRoPEConfig,
		input, output unsafe.Pointer,
		batch, seqLen, numHeads, headDim int,
		format dtype.DType,
	)
	RoPE(
		config RoPEConfig,
		input, output unsafe.Pointer,
		seqLen, numHeads, headDim int,
		format dtype.DType,
	)
	RoPEPairs(
		output, input, cosBuffer, sinBuffer unsafe.Pointer,
		halfDim int,
		format dtype.DType,
	)
}

type Sampling interface {
	GreedySample(dst, logits unsafe.Pointer, vocabSize int, format dtype.DType)
	TopKSample(dst, logits unsafe.Pointer, vocabSize int, config SamplingConfig, format dtype.DType)
	TopPSample(dst, logits unsafe.Pointer, vocabSize int, config SamplingConfig, format dtype.DType)
}

type Shape interface {
	Concat(left, right, output unsafe.Pointer, format dtype.DType)
	CopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType)
	Gather(source, indices, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType)
	LastToken(input, output unsafe.Pointer, batch, seq, hidden int, format dtype.DType)
	MaskedFill(input, mask, fill, output unsafe.Pointer, count int, format dtype.DType)
	MergeHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType)
	PageGather(storage, pageTable, pageSize, output unsafe.Pointer, format dtype.DType)
	PageGatherWithLiveRows(
		storage, pageTable, pageSize, output unsafe.Pointer,
		liveRows int,
		format dtype.DType,
	)
	PageWrite(
		storage, values, pageIDs, offsets, output unsafe.Pointer,
		pageSize int,
		format dtype.DType,
	)
	Reshape(input, output unsafe.Pointer, count int, format dtype.DType)
	Scatter(target, indices, updates, output unsafe.Pointer, outerDim, innerDim int, format dtype.DType)
	Slice(input, output unsafe.Pointer, dim, start, end int, format dtype.DType)
	Split2(input, left, right unsafe.Pointer, format dtype.DType)
	SplitHeads(input, output unsafe.Pointer, batch, seq, heads, headDim int, format dtype.DType)
	Transpose(input, permutation, output unsafe.Pointer, rank int, format dtype.DType)
	Transpose2D(input, output unsafe.Pointer, rows, cols int, format dtype.DType)
	UpsampleNearest2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	ViewAsHeads(input, output unsafe.Pointer, batch, seq, numHeads, headDim int, format dtype.DType)
	Where(mask, positive, negative, output unsafe.Pointer, count int, format dtype.DType)
}

type VSA interface {
	Bind(left, right, output unsafe.Pointer, count int, format dtype.DType)
	Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType)
	InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType)
}

type HostBackend interface {
	PosPop
}

type Backend interface {
	Activation
	ActiveInference
	Attention
	Causal
	Checkpoint
	Convolution
	Dequant
	Dot
	Dropout
	Elementwise
	Embedding
	Geometry
	Hawkes
	Interpretability
	LayerNorm
	Losses
	Masking
	Math
	Matmul
	ModelEditing
	Normalization
	Optimizer
	Peel
	Physics
	Pool
	PredictiveCoding
	Quant
	Reduction
	RoPE
	Sampling
	Shape
	VSA
}
