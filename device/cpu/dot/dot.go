package dot

/*
Product implements device.{'Dot'} for the CPU backend.
*/
type Product struct{}

/*
New constructs a Product receiver for CPU dispatch.
*/
func New() Product {
	return Product{}
}

var Default = New()
