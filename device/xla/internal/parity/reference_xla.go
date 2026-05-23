//go:build xla

package parity

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
)

/*
UnaryReference computes the CPU production reference for a unary activation.
*/
type UnaryReference func(dst, src unsafe.Pointer, count int)

/*
ReferenceReLU returns the production CPU reference kernel for ReLU.
*/
func ReferenceReLU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.ReLU, format)
}

/*
ReferenceExp returns the production CPU reference kernel for Exp.
*/
func ReferenceExp(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Exp, format)
}

/*
ReferenceGelu returns the production CPU reference kernel for exact erf GELU.
*/
func ReferenceGelu(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Gelu, format)
}

/*
ReferenceGeluTanh returns the production CPU reference kernel for tanh-approx GELU.
*/
func ReferenceGeluTanh(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.GeluTanh, format)
}

/*
ReferenceLog returns the production CPU reference kernel for Log.
*/
func ReferenceLog(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Log, format)
}

/*
ReferenceSigmoid returns the production CPU reference kernel for Sigmoid.
*/
func ReferenceSigmoid(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Sigmoid, format)
}

/*
ReferenceSilu returns the production CPU reference kernel for SiLU.
*/
func ReferenceSilu(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Silu, format)
}

/*
ReferenceCELU returns the production CPU reference kernel for CELU.
*/
func ReferenceCELU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.CELU, format)
}

/*
ReferenceSoftplus returns the production CPU reference kernel for Softplus.
*/
func ReferenceSoftplus(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Softplus, format)
}

/*
ReferenceSoftsign returns the production CPU reference kernel for Softsign.
*/
func ReferenceSoftsign(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Softsign, format)
}

/*
ReferenceHardSigmoid returns the production CPU reference kernel for HardSigmoid.
*/
func ReferenceHardSigmoid(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.HardSigmoid, format)
}

/*
ReferenceHardSwish returns the production CPU reference kernel for HardSwish.
*/
func ReferenceHardSwish(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.HardSwish, format)
}

/*
ReferenceHardTanh returns the production CPU reference kernel for HardTanh.
*/
func ReferenceHardTanh(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.HardTanh, format)
}

/*
ReferenceHardGelu returns the production CPU reference kernel for HardGelu.
*/
func ReferenceHardGelu(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.HardGelu, format)
}

/*
ReferenceQuickGelu returns the production CPU reference kernel for QuickGelu.
*/
func ReferenceQuickGelu(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.QuickGelu, format)
}

/*
ReferenceTanhShrink returns the production CPU reference kernel for TanhShrink.
*/
func ReferenceTanhShrink(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.TanhShrink, format)
}

/*
ReferenceTanh returns the production CPU reference kernel for Tanh.
*/
func ReferenceTanh(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Tanh, format)
}

/*
ReferenceELU returns the production CPU reference kernel for ELU.
*/
func ReferenceELU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.ELU, format)
}

/*
ReferenceSELU returns the production CPU reference kernel for SELU.
*/
func ReferenceSELU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.SELU, format)
}

/*
ReferenceMish returns the production CPU reference kernel for Mish.
*/
func ReferenceMish(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Mish, format)
}

func productionUnaryReference(
	kernel func(dst, src unsafe.Pointer, count int, format dtype.DType),
	format dtype.DType,
) UnaryReference {
	return func(dst, src unsafe.Pointer, count int) {
		kernel(dst, src, count, format)
	}
}

/*
ComputeUnaryReferenceBytes runs the CPU reference and returns encoded storage bytes.
*/
func ComputeUnaryReferenceBytes(
	source []float32,
	format dtype.DType,
	reference UnaryReference,
) []byte {
	sourceBytes, err := encodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceBytes))
	reference(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceBytes[0]),
		len(source),
	)

	return destinationBytes
}
