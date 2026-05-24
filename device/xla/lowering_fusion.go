package xla

func RegisterFusionLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("matmul_bias_gelu", 3))
	registry.Register(NewVariadicLowering("layernorm_residual", 4))
}
