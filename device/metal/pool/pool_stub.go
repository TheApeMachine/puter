//go:build !darwin || !cgo

package pool

func (pool *Pool) stubHost() {
	pool.host.NeedsPlatform()
}
