package geometry

import "math"

type f64TranscendentalKernelImpl struct {
	sinCos     func(sineDestination, cosineDestination, phases []float64)
	cosine     func(destination, source []float64)
	arcTangent func(destination, yValues, xValues []float64)
	name       string
	available  bool
}

func pickF64TranscendentalKernel(
	candidates []f64TranscendentalKernelImpl,
) f64TranscendentalKernelImpl {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate
		}
	}

	panic("geometry: no float64 transcendental kernel available")
}

var transcendentalFloat64Kernel = func() f64TranscendentalKernelImpl {
	return pickF64TranscendentalKernel(transcendentalFloat64Funcs)
}()

func vecAtan2(destination, yValues, xValues []float64) {
	if len(destination) == 0 {
		return
	}

	transcendentalFloat64Kernel.arcTangent(destination, yValues, xValues)
}

func vecCos(destination, source []float64) {
	if len(destination) == 0 {
		return
	}

	transcendentalFloat64Kernel.cosine(destination, source)
}

func vecSinCos(sineDestination, cosineDestination, phases []float64) {
	if len(phases) == 0 {
		return
	}

	transcendentalFloat64Kernel.sinCos(sineDestination, cosineDestination, phases)
}

func vecAtan2Float64Scalar(destination, yValues, xValues []float64) {
	for index := range destination {
		destination[index] = math.Atan2(yValues[index], xValues[index])
	}
}

func vecCosFloat64Scalar(destination, source []float64) {
	for index := range destination {
		destination[index] = math.Cos(source[index])
	}
}

func vecSinCosFloat64Scalar(sineDestination, cosineDestination, phases []float64) {
	for index := range phases {
		sineDestination[index], cosineDestination[index] = math.Sincos(phases[index])
	}
}
