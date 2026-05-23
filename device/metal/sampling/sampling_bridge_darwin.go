//go:build darwin && cgo

package sampling

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/sampling.m"
#include "native/greedy.m"
#include "native/nucleus.m"
*/
import "C"
