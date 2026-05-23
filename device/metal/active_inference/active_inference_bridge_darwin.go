//go:build darwin && cgo

package active_inference

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/active_inference.m"
#include "native/belief.m"
#include "native/free_energy.m"
*/
import "C"
