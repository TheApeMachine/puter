//go:build !cuda

package predictive_coding

func (predictiveCoding *PredictiveCoding) stubHost() {
	predictiveCoding.host.NeedsPlatform()
}
