//go:build cuda

package hawkes

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/hawkes.c"
#include "native/intensity.c"
#include "native/kernel.c"
#include "native/likelihood.c"
#include "native/markov.c"
*/
import "C"
