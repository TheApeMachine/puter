//go:build cuda

package embedding

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/bag.c"
#include "native/embedding.c"
#include "native/lookup.c"
#include "native/timestep.c"
*/
import "C"
