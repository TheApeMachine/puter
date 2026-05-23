//go:build cuda

package elementwise

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/arithmetic.c"
#include "native/axpy.c"
#include "native/elementwise.c"
#include "native/math.c"
*/
import "C"
