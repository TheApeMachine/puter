//go:build darwin && cgo

package normalization

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/normalization.m"
#include "native/batchnorm.m"
#include "native/groupnorm.m"
#include "native/instancenorm.m"
*/
import "C"
