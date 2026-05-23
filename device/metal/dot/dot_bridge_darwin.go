//go:build darwin && cgo

package dot

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/dot.m"
#include "native/inner_product.m"
*/
import "C"
