//go:build darwin && cgo

package convolution

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge -I${SRCDIR}/../pool
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../../pool/native/pool.m"
#include "native/convolution.m"
#include "native/conv2d.m"
*/
import "C"
