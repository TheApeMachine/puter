//go:build cuda

package dot

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/dot.c"
#include "native/inner_product.c"
*/
import "C"
