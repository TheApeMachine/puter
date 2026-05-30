package geometry

import "github.com/theapemachine/puter/device"

/*
PhaseDialDimensions is the complex basis count for PhaseDial encoding.
Each dimension uses one prime frequency omega_k from PhaseDialPrimes.
*/
const PhaseDialDimensions = device.PhaseDialDimensions

/*
PhaseDialPrimes holds the prime frequency table indexed by dial dimension.
*/
var PhaseDialPrimes [PhaseDialDimensions]uint32

func init() {
	fillPhaseDialPrimes()
}

func fillPhaseDialPrimes() {
	var candidate uint32 = 2
	var primeIndex int

	for primeIndex < PhaseDialDimensions {
		if isPrime(candidate) {
			PhaseDialPrimes[primeIndex] = candidate
			primeIndex++
		}

		candidate++
	}
}

func isPrime(candidate uint32) bool {
	if candidate < 2 {
		return false
	}

	if candidate == 2 {
		return true
	}

	if candidate%2 == 0 {
		return false
	}

	for divisor := uint32(3); divisor*divisor <= candidate; divisor += 2 {
		if candidate%divisor == 0 {
			return false
		}
	}

	return true
}
