#include "softmax.h"
#include "activation.h"
#include "../internal/bridge/core_private.h"

// CUDA dispatch for softmax — NVRTC launch wired in family bridge.
