package activation

/*
GLUVariantName maps GLU variants to XLA lowering operation names.
*/
func GLUVariantName(variant GLUVariant) (string, bool) {
	switch variant {
	case GLU:
		return "glu", true
	case GeGLU:
		return "geglu", true
	case GeGLUTanh:
		return "geglu_tanh", true
	case SwiGLU:
		return "swiglu", true
	case ReGLU:
		return "reglu", true
	case SiGLU:
		return "siglu", true
	case LinGLU:
		return "linglu", true
	case SeGLU:
		return "seglu", true
	default:
		return "", false
	}
}
