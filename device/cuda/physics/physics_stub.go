//go:build !cuda

package physics

func (physics *Physics) stubHost() {
	physics.host.NeedsPlatform()
}
