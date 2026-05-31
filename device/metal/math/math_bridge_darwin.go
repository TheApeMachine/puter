//go:build darwin && cgo

package math

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge -I${SRCDIR}/../causal
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../causal/native/matrix.m"
*/
import "C"
