//go:build darwin && cgo

package elementwise

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/elementwise.m"
#include "native/arithmetic.m"
#include "native/axpy.m"
#include "native/math.m"
*/
import "C"
