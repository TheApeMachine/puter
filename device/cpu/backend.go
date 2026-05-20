package cpu

import (
	"context"
	"sync/atomic"

	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/qpool"
)

type Backend struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
	pool   *qpool.Q
	closed atomic.Bool
}

func NewBackend(ctx context.Context, pool *qpool.Q) (*Backend, error) {
	ctx, cancel := context.WithCancel(ctx)

	return &Backend{
		ctx:    ctx,
		cancel: cancel,
		pool:   pool,
	}, nil
}

var _ device.Backend = (*Backend)(nil)
