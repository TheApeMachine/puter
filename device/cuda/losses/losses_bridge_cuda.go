//go:build cuda

package losses

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/classification.c"
#include "native/losses.c"
#include "native/regression.c"
*/
import "C"
