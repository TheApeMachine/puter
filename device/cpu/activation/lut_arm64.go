//go:build arm64

package activation

//go:noescape
func ApplyF16LUTNEON(dst, src *uint16, count int, lut *[65536]uint16)
