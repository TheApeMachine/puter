//go:build !darwin || !cgo

package fusion

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

/*
Dispatch is unavailable off Darwin.
*/
func (program *Program) Dispatch(
	contextRef uintptr,
	inputBufferRefs []uintptr,
	outputBufferRef uintptr,
	count int,
) error {
	_ = program
	_ = contextRef
	_ = inputBufferRefs
	_ = outputBufferRef
	_ = count

	return fmt.Errorf("metal fusion: %w", tensor.ErrNeedsPlatformSetup)
}
