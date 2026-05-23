//go:build cuda

package causal

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/adjustment.c"
#include "native/causal.c"
#include "native/dag.c"
#include "native/intervention.c"
#include "native/matrix.c"
*/
import "C"
