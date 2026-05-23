//go:build !cuda

package dot

func (product *Product) stubHost() {
	product.host.NeedsPlatform()
}
