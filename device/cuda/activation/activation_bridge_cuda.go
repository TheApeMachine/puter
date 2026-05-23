//go:build cuda

package activation

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/activation.c"
#include "native/gated.c"
#include "native/parametric.c"
#include "native/softmax.c"
#include "native/standard.c"
*/
import "C"
