//go:build darwin && cgo

package dequant

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/dequant.m"
#include "native/int8.m"
*/
import "C"
