//go:build darwin && cgo

package activation

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/activation.m"
#include "native/gated.m"
#include "native/lut.m"
#include "native/parametric.m"
#include "native/softmax.m"
#include "native/standard.m"
*/
import "C"
