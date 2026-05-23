//go:build !cuda

package hawkes

func (hawkes *Hawkes) stubHost() {
	hawkes.host.NeedsPlatform()
}
