package losses

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type lossLoadFunc func(pointer unsafe.Pointer, index int) float32

func lossLoadFuncFor(format dtype.DType) lossLoadFunc {
	switch format {
	case dtype.Float32:
		return loadF32
	case dtype.Float16:
		return loadF16
	case dtype.BFloat16:
		return loadBF16
	default:
		panic("losses: unsupported dtype")
	}
}

func dispatchMSE(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 {
		return 0
	}

	if format == dtype.Float32 {
		return runMSEF32(predictions, targets, count)
	}

	return mseTyped(predictions, targets, count, lossLoadFuncFor(format))
}

func dispatchMAE(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 {
		return 0
	}

	if format == dtype.Float32 {
		return runMAEF32(predictions, targets, count)
	}

	return maeTyped(predictions, targets, count, lossLoadFuncFor(format))
}

func dispatchHuber(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 {
		return 0
	}

	if format == dtype.Float32 {
		return runHuberF32(predictions, targets, count)
	}

	return huberTyped(predictions, targets, count, lossLoadFuncFor(format))
}

func dispatchBinaryCrossEntropy(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 {
		return 0
	}

	if format == dtype.Float32 {
		return runBinaryCrossEntropyF32(predictions, targets, count)
	}

	return binaryCrossEntropyTyped(predictions, targets, count, lossLoadFuncFor(format))
}

func dispatchKLDivergence(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 {
		return 0
	}

	if format == dtype.Float32 {
		return runKLDivergenceF32(predictions, targets, count)
	}

	return klDivergenceTyped(predictions, targets, count, lossLoadFuncFor(format))
}

func mseTyped(
	predictions, targets unsafe.Pointer,
	count int,
	load lossLoadFunc,
) float32 {
	var sum float64

	for index := 0; index < count; index++ {
		diff := load(predictions, index) - load(targets, index)
		sum += float64(diff) * float64(diff)
	}

	return float32(sum / float64(count))
}

func maeTyped(
	predictions, targets unsafe.Pointer,
	count int,
	load lossLoadFunc,
) float32 {
	var sum float64

	for index := 0; index < count; index++ {
		diff := load(predictions, index) - load(targets, index)
		sum += math.Abs(float64(diff))
	}

	return float32(sum / float64(count))
}

func huberTyped(
	predictions, targets unsafe.Pointer,
	count int,
	load lossLoadFunc,
) float32 {
	const delta = float32(1.0)
	var sum float64

	for index := 0; index < count; index++ {
		diff := load(predictions, index) - load(targets, index)
		absDiff := float32(math.Abs(float64(diff)))

		switch {
		case absDiff <= delta:
			sum += 0.5 * float64(diff) * float64(diff)
		default:
			sum += float64(delta) * (float64(absDiff) - 0.5*float64(delta))
		}
	}

	return float32(sum / float64(count))
}

func binaryCrossEntropyTyped(
	predictions, targets unsafe.Pointer,
	count int,
	load lossLoadFunc,
) float32 {
	var sum float64
	const eps = 1e-7

	for index := 0; index < count; index++ {
		value := load(predictions, index)
		target := load(targets, index)
		clamped := math.Max(eps, math.Min(1-eps, float64(value)))
		sum += -float64(target)*math.Log(clamped) -
			(1-float64(target))*math.Log(1-clamped)
	}

	return float32(sum / float64(count))
}

func klDivergenceTyped(
	predictions, targets unsafe.Pointer,
	count int,
	load lossLoadFunc,
) float32 {
	var sum float64
	const eps = 1e-12

	for index := 0; index < count; index++ {
		predicted := math.Max(eps, float64(load(predictions, index)))
		target := math.Max(eps, float64(load(targets, index)))
		sum += target * math.Log(target/predicted)
	}

	return float32(sum / float64(count))
}

func dispatchCrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	if batchSize == 0 || classes == 0 {
		return 0
	}

	load := lossLoadFuncFor(format)
	return crossEntropyTyped(logits, targets, batchSize, classes, load)
}

func crossEntropyTyped(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	load lossLoadFunc,
) float32 {
	var sum float64

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		rowBase := batchIndex * classes
		maxLogit := load(logits, rowBase)

		for classIndex := 1; classIndex < classes; classIndex++ {
			candidate := load(logits, rowBase+classIndex)

			if candidate > maxLogit {
				maxLogit = candidate
			}
		}

		var denominator float64

		for classIndex := 0; classIndex < classes; classIndex++ {
			candidate := load(logits, rowBase+classIndex)
			denominator += math.Exp(float64(candidate - maxLogit))
		}

		targetClass := int(loadInt32(targets, batchIndex))

		if targetClass < 0 || targetClass >= classes {
			panic("losses: cross entropy target out of range")
		}

		logProb := float64(load(logits, rowBase+targetClass)-maxLogit) - math.Log(denominator)
		sum += -logProb
	}

	return float32(sum / float64(batchSize))
}

func huberMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	const delta = float32(1.0)
	var sum float64

	for index := 0; index < count; index++ {
		diff := loadF32(predictions, index) - loadF32(targets, index)
		absDiff := float32(math.Abs(float64(diff)))

		switch {
		case absDiff <= delta:
			sum += 0.5 * float64(diff) * float64(diff)
		default:
			sum += float64(delta) * (float64(absDiff) - 0.5*float64(delta))
		}
	}

	return float32(sum / float64(count))
}

func binaryCrossEntropyMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	var sum float64
	const eps = 1e-7

	for index := 0; index < count; index++ {
		value := loadF32(predictions, index)
		target := loadF32(targets, index)
		clamped := math.Max(eps, math.Min(1-eps, float64(value)))
		sum += -float64(target)*math.Log(clamped) -
			(1-float64(target))*math.Log(1-clamped)
	}

	return float32(sum / float64(count))
}

func klDivergenceMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	var sum float64
	const eps = 1e-12

	for index := 0; index < count; index++ {
		predicted := math.Max(eps, float64(loadF32(predictions, index)))
		target := math.Max(eps, float64(loadF32(targets, index)))
		sum += target * math.Log(target/predicted)
	}

	return float32(sum / float64(count))
}

func loadF32(pointer unsafe.Pointer, index int) float32 {
	return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
}

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func loadInt32(pointer unsafe.Pointer, index int) int32 {
	return *(*int32)(unsafe.Add(pointer, uintptr(index)*4))
}
