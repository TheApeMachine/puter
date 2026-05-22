package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchSum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runSumF32(values, count)
	case dtype.BFloat16:
		bf16Result := dispatchSumBF16(values, count)
		return (&bf16Result).Float32()
	case dtype.Float16:
		fp16Result := dispatchSumFP16(values, count)
		return fp16Result.Float32()
	default:
		panic("reduction: Sum unsupported dtype")
	}
}

func dispatchProd(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runProdF32(values, count)
	case dtype.Float16, dtype.BFloat16:
		if format == dtype.Float16 {
			return dispatchProdFP16(values, count)
		}

		return dispatchProdBF16(values, count)
	default:
		panic("reduction: Prod unsupported dtype")
	}
}

func dispatchReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runMinF32(values, count)
	case dtype.Float16:
		return dispatchMinFP16(values, count)
	case dtype.BFloat16:
		return dispatchMinBF16(values, count)
	default:
		panic("reduction: ReduceMin unsupported dtype")
	}
}

func dispatchReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runMaxF32(values, count)
	case dtype.Float16:
		return dispatchMaxFP16(values, count)
	case dtype.BFloat16:
		return dispatchMaxBF16(values, count)
	default:
		panic("reduction: ReduceMax unsupported dtype")
	}
}

func dispatchL1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runL1NormF32(values, count)
	case dtype.Float16:
		return dispatchL1NormFP16(values, count)
	case dtype.BFloat16:
		return dispatchL1NormBF16(values, count)
	default:
		panic("reduction: L1Norm unsupported dtype")
	}
}
