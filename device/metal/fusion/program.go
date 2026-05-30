package fusion

/*
Program is one MTLLibrary-compiled elementwise fusion kernel bound to a
Metal backend context at first dispatch.
*/
type Program struct {
	source     string
	kernelName string
	handle     programHandle
}

func (program *Program) close() {
	if program == nil {
		return
	}

	program.releaseHandle()
}
