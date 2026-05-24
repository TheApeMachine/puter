package convolution

func requireConvExtents(inputElements, outputElements int) {
	if inputElements == 0 || outputElements == 0 {
		panic("convolution: zero tensor extent")
	}
}
