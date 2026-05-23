//go:build darwin && cgo

package dropout

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/dropout.m"
#include "native/mask.m"
*/
import "C"
