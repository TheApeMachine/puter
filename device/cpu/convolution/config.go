package convolution

import "github.com/theapemachine/puter/device"

/*
Conv2DConfig binds stride, padding, and dilation for 2-D convolution
and transposed convolution entry points.
*/
type Conv2DConfig = device.Conv2DConfig

func DefaultConv2DConfig() Conv2DConfig {
	return Conv2DConfig{
		StrideH: 1, StrideW: 1,
		PaddingH: 0, PaddingW: 0,
		DilationH: 1, DilationW: 1,
	}
}

/*
Conv1DConfig binds stride, padding, and dilation for 1-D convolution.
*/
type Conv1DConfig = device.Conv1DConfig

func DefaultConv1DConfig() Conv1DConfig {
	return Conv1DConfig{Stride: 1, Padding: 0, Dilation: 1}
}

/*
Conv3DConfig binds stride, padding, and dilation for 3-D convolution.
*/
type Conv3DConfig = device.Conv3DConfig

func DefaultConv3DConfig() Conv3DConfig {
	return Conv3DConfig{
		StrideD: 1, StrideH: 1, StrideW: 1,
		DilationD: 1, DilationH: 1, DilationW: 1,
	}
}
