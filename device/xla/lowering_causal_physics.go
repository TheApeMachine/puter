package xla

func RegisterCausalPhysicsLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("cate", 2))
	registry.Register(NewVariadicLowering("counterfactual", 3))
	registry.Register(NewVariadicLowering("backdoor_adjustment", 2))
	registry.Register(NewVariadicLowering("do_intervene", 2))
	registry.Register(NewVariadicLowering("frontdoor_adjustment", 3))
	registry.Register(NewVariadicLowering("iv_estimate", 3))
	registry.Register(NewVariadicLowering("markov_flow", 2))
	registry.Register(UnaryParamLowering{operationName: "dag_markov_factorization"})
	registry.Register(UnaryParamLowering{operationName: "cholesky"})
	registry.Register(UnaryParamLowering{operationName: "grad1d"})
	registry.Register(UnaryParamLowering{operationName: "laplacian1d"})
	registry.Register(UnaryParamLowering{operationName: "laplacian4"})
	registry.Register(UnaryParamLowering{operationName: "central_difference_interior"})
	registry.Register(NewVariadicLowering("madelung_continuity", 2))
	registry.Register(UnaryParamLowering{operationName: "quantum_potential"})
	registry.Register(NewVariadicLowering("fft1d", 2))
	registry.Register(UnaryParamLowering{operationName: "vector_slice_copy"})
	registry.Register(UnaryParamLowering{operationName: "markov_mutual_information"})
}
