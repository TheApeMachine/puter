//go:build !cuda

package pool

func (pool *Pool) stubHost() {
	pool.host.NeedsPlatform()
}
