//go:build !darwin || !cgo

package random

func (random *Random) stubHost() {
	random.host.NeedsPlatform()
}
