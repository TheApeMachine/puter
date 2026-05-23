//go:build arm64

package activation

var f16LUTGatherFuncs = []lutGatherImpl{
	{ApplyF16LUTNEON, "neon", true},
	{applyF16LUTScalar, "generic", true},
}
