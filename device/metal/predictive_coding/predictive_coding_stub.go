//go:build !darwin || !cgo

package predictive_coding

func (predictiveCoding *PredictiveCoding) stubHost() {
	predictiveCoding.host.NeedsPlatform()
}
