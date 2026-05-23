//go:build darwin && cgo

package predictive_coding

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/predictive_coding.m"
#include "native/forward.m"
#include "native/learning.m"
*/
import "C"
