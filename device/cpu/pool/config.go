package pool

import "github.com/theapemachine/puter/device"

/*
PoolConfig describes kernel size, stride, and padding for fixed-size
2-D pooling windows.
*/
type PoolConfig = device.PoolConfig

func DefaultPoolConfig() PoolConfig {
	return PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}
}
