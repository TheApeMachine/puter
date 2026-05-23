//go:build cuda

package predictive_coding

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/forward.c"
#include "native/learning.c"
#include "native/predictive_coding.c"
*/
import "C"
