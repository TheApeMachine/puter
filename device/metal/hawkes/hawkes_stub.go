//go:build !darwin || !cgo

package hawkes

func (hawkes *Hawkes) stubHost() {
	hawkes.host.NeedsPlatform()
}
