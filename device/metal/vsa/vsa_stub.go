//go:build !darwin || !cgo

package vsa

func (vSA *VSA) stubHost() {
	vSA.host.NeedsPlatform()
}
