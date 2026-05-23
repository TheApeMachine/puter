//go:build !darwin || !cgo

package dot

func (product *Product) stubHost() {
	product.host.NeedsPlatform()
}
