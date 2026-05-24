package quant

/*
Quantization implements device.{'Quant'} for the CPU backend.
*/
type Quantization struct{}

/*
New constructs a Quantization receiver for CPU dispatch.
*/
func New() Quantization {
	return Quantization{}
}
