//go:build amd64

package activation

//go:noescape
func ApplyF16LUTAVX512(dst, src *uint16, count int, lut *[65536]uint16)

//go:noescape
func ApplyF16LUTAVX2(dst, src *uint16, count int, lut *[65536]uint16)

//go:noescape
func ApplyF16LUTSSE2(dst, src *uint16, count int, lut *[65536]uint16)
