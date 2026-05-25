package pool

import (
	"context"
	"errors"
	"sync"

	"github.com/theapemachine/errnie"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu"
	"github.com/theapemachine/puter/device/metal"
	"github.com/theapemachine/qpool"
)

var ErrDeviceNotFound = errors.New("pool: device not found")

/*
Pool discovers and owns every compute device available on the host.
Placement policy belongs in manifesto/runtime; execution belongs in puter/runner.
*/
type Pool struct {
	ctx         context.Context
	cancel      context.CancelFunc
	devices     map[DeviceID]device.Backend
	hostBackend device.HostBackend
	workerPool  *qpool.Q
	closeOnce   sync.Once
}

/*
New discovers resident device backends.
*/
func New(ctx context.Context, workerPool *qpool.Q) (*Pool, error) {
	ctx, cancel := context.WithCancel(ctx)

	cpuBackend, err := cpu.NewBackend(ctx, workerPool)

	if err != nil {
		cancel()
		return nil, err
	}

	deviceMap := map[DeviceID]device.Backend{
		{Location: tensor.Host, Index: 0}: cpuBackend,
	}

	metalBackend, err := metal.NewBackend(ctx, workerPool)

	if err == nil {
		deviceMap[DeviceID{Location: tensor.Metal, Index: 0}] = metalBackend
	}

	devicePool := &Pool{
		ctx:         ctx,
		cancel:      cancel,
		devices:     deviceMap,
		hostBackend: cpu.NewHostBackend(),
		workerPool:  workerPool,
	}

	return devicePool, errnie.Require(map[string]any{
		"ctx":     ctx,
		"devices": devicePool.devices,
	})
}

/*
DeviceIDs returns discovered devices in stable precedence order.
*/
func (devicePool *Pool) DeviceIDs() []DeviceID {
	precedence := []tensor.Location{
		tensor.CUDA,
		tensor.Metal,
		tensor.XLA,
		tensor.Host,
	}

	ids := make([]DeviceID, 0, len(devicePool.devices))

	for _, location := range precedence {
		deviceID := DeviceID{Location: location, Index: 0}

		if _, ok := devicePool.devices[deviceID]; ok {
			ids = append(ids, deviceID)
		}
	}

	return ids
}

/*
Device returns one discovered backend by id.
*/
func (devicePool *Pool) Device(id DeviceID) (device.Backend, error) {
	if devicePool == nil {
		return nil, ErrDeviceNotFound
	}

	backend, ok := devicePool.devices[id]

	if !ok {
		return nil, ErrDeviceNotFound
	}

	return backend, nil
}

/*
MemoryBackend returns the first discovered device that owns resident tensor storage.
*/
func (devicePool *Pool) MemoryBackend() (tensor.Backend, DeviceID, error) {
	for _, deviceID := range devicePool.DeviceIDs() {
		backend, err := devicePool.Device(deviceID)

		if err != nil {
			continue
		}

		memory, ok := backend.(tensor.Backend)

		if !ok {
			continue
		}

		return memory, deviceID, nil
	}

	hostMemory := tensor.NewHostBackend()

	return hostMemory, DeviceID{Location: tensor.Host, Index: 0}, nil
}

/*
PinTo narrows the pool's resident device set to a single Location, releasing
every discovered backend that does not match. The pinned device becomes both
the unique choice for execution.Backend.pickDevice and the only candidate
visible to MemoryBackend.

Used by callers that need deterministic backend selection ahead of the
backend-aware dispatcher work — for example, GAPS.md §8.3 step 6 brings
chat.yml up on CPU before validating the same chain against Metal, so
caramba pins the pool to tensor.Host until the Metal buffer-handle
plumbing (§3.1, lines 1223-1237) is in place.

Returns an error if no discovered device matches the requested location; in
that case the pool is left untouched so the caller can fall back.
*/
func (devicePool *Pool) PinTo(location tensor.Location) error {
	if devicePool == nil {
		return ErrDeviceNotFound
	}

	target := DeviceID{Location: location, Index: 0}

	keep, ok := devicePool.devices[target]

	if !ok {
		return ErrDeviceNotFound
	}

	for deviceID, backend := range devicePool.devices {
		if deviceID == target {
			continue
		}

		if closer, ok := backend.(interface{ Close() error }); ok {
			_ = closer.Close()
		}

		delete(devicePool.devices, deviceID)
	}

	devicePool.devices[target] = keep

	return nil
}

/*
WorkerPool returns the shared goroutine pool used by discovered devices.
*/
func (devicePool *Pool) WorkerPool() *qpool.Q {
	if devicePool == nil {
		return nil
	}

	return devicePool.workerPool
}

/*
HostBackend returns CPU-side preprocessing (PosPop).
*/
func (devicePool *Pool) HostBackend() device.HostBackend {
	if devicePool == nil {
		return nil
	}

	return devicePool.hostBackend
}

/*
Close releases pool resources.
*/
func (devicePool *Pool) Close() error {
	if devicePool == nil {
		return nil
	}

	devicePool.closeOnce.Do(devicePool.cancel)

	return nil
}
