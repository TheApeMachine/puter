//go:build !cuda

package vsa

func (vSA *VSA) stubHost() {
	vSA.host.NeedsPlatform()
}
