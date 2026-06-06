//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation -framework MetalPerformanceShaders

#include "internal/bridge/core_private.h"
#include "internal/bridge/core_darwin.m"
#include "internal/bridge/fusion_jit_darwin.m"

#include "internal/runtime/bridge_transformer_darwin.m"
#include "internal/runtime/bridge_projection_darwin.m"
#include "internal/runtime/bridge_utility_darwin.m"
#include "internal/runtime/bridge_shape_darwin.m"
#include "internal/runtime/bridge_shape_common_darwin.m"
#include "internal/runtime/bridge_shape_index_darwin.m"
#include "internal/runtime/bridge_optimizer_darwin.m"
*/
import "C"
