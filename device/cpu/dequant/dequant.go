package dequant

/*
Dequantization implements device.{'Dequant'} for the CPU backend.
*/
type Dequantization struct{}

/*
New constructs a Dequantization receiver for CPU dispatch.
*/
func New() Dequantization {
	return Dequantization{}
}
