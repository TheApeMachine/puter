//go:build !darwin || !cgo

package active_inference

func (activeInference *ActiveInference) stubHost() {
	activeInference.host.NeedsPlatform()
}
