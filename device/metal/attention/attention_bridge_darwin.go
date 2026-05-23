//go:build darwin && cgo

package attention

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/attention.m"
#include "native/flash.m"
#include "native/masking.m"
#include "native/multihead.m"
#include "native/scaled_dot_product.m"
*/
import "C"
