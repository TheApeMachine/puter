//go:build cuda

package normalization

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/batchnorm.c"
#include "native/groupnorm.c"
#include "native/instancenorm.c"
#include "native/normalization.c"
*/
import "C"
