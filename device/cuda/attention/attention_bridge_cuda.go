//go:build cuda

package attention

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/attention.c"
#include "native/flash.c"
#include "native/masking.c"
#include "native/multihead.c"
#include "native/scaled_dot_product.c"
*/
import "C"
