//go:build !cuda

package active_inference

func (activeInference *ActiveInference) stubHost() {
	activeInference.host.NeedsPlatform()
}
