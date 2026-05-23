//go:build !darwin || !cgo

package physics

func (physics *Physics) stubHost() {
	physics.host.NeedsPlatform()
}
