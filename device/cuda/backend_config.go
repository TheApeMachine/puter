//go:build cuda

package cuda

/*
Config holds CUDA backend construction parameters.
*/
type Config struct {
	NativeAlignment int
}

/*
DefaultConfig returns the production CUDA backend configuration.
*/
func DefaultConfig() Config {
	return Config{
		NativeAlignment: 128,
	}
}
