//go:build !amd64 && !arm64

package activation

var f16LUTGatherFuncs = []lutGatherImpl{
	{applyF16LUTScalar, "generic", true},
}
