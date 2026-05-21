package runner

import "github.com/theapemachine/manifesto/tensor"

/*
orderKernelArguments reorders dispatch tensors to match registered kernel signatures.
Manifest graphs wire data inputs before checkpoint weights; some kernels expect
weights first (embedding_lookup).
*/
func orderKernelArguments(kernel string, args []tensor.Tensor) []tensor.Tensor {
	switch kernel {
	case "embedding_lookup":
		if len(args) == 3 {
			return []tensor.Tensor{args[1], args[0], args[2]}
		}
	}

	return args
}
