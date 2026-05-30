package geometry

import (
	"math"
	"math/cmplx"
)

/*
PhaseDial is a high-dimensional complex vector of rotational phase gradients.
Each dimension uses a prime frequency ω to accumulate phase from sequence
position and a structural mix of the Value words; encoding normalizes to
unit L2 magnitude for cosine similarity and composition.
*/
type PhaseDial []complex128

/*
NewPhaseDial allocates a zero-initialized PhaseDial.
*/
func NewPhaseDial() PhaseDial {
	return make(PhaseDial, PhaseDialDimensions)
}

/*
structuralPhaseMix folds the first eight uint64 words of a ValueToken into a
scalar in [0,1) so repetitive token packing still leaves discriminative
phase when the Morton slab saturates.
*/
func structuralPhaseMix(token *ValueToken) float64 {
	if token == nil {
		return 0
	}

	var mix uint64

	const wordBlocks = 8

	for blockIndex := 0; blockIndex < wordBlocks; blockIndex++ {
		mix ^= token[blockIndex] * (0x9e3779b185ebca87 + uint64(blockIndex+1)*0x6c62272e07bb0142)
	}

	return float64(mix>>32) * (1.0 / float64(1<<32))
}

/*
EncodeFromValues generates a PhaseDial from a value sequence.
Value-native: uses word structure and position for phase; no raw byte scan.
*/
func (dial PhaseDial) EncodeFromValues(values []ValueToken) PhaseDial {
	if len(values) == 0 {
		return dial
	}

	if len(dial) < PhaseDialDimensions {
		dial = NewPhaseDial()
	}

	structuralScaled := make([]float64, len(values))

	for valueIndex := range values {
		structuralScaled[valueIndex] = structuralPhaseMix(&values[valueIndex]) * math.Pi * 2
	}

	valuePhases := make([]float64, len(values))
	cosineBuffer := make([]float64, len(values))
	sineBuffer := make([]float64, len(values))

	for dimIndex := 0; dimIndex < PhaseDialDimensions; dimIndex++ {
		omega := float64(PhaseDialPrimes[dimIndex])

		for valueIndex := range values {
			valuePhases[valueIndex] = (omega * float64(valueIndex+1) * 0.1) + structuralScaled[valueIndex]
		}

		vecSinCos(sineBuffer, cosineBuffer, valuePhases)
		dial[dimIndex] = complex(vecSum(cosineBuffer), vecSum(sineBuffer))
	}

	return dial.normalize()
}

/*
AddValuePhase incrementally adds a single value's phase to an unnormalized PhaseDial.
*/
func (dial PhaseDial) AddValuePhase(value ValueToken, position int) {
	if len(dial) < PhaseDialDimensions {
		return
	}

	var phases [PhaseDialDimensions]float64
	structuralPhase := structuralPhaseMix(&value) * math.Pi * 2

	for dimIndex := 0; dimIndex < PhaseDialDimensions; dimIndex++ {
		omega := float64(PhaseDialPrimes[dimIndex])
		phases[dimIndex] = (omega * float64(position+1) * 0.1) + structuralPhase
	}

	var cosines [PhaseDialDimensions]float64
	var sines [PhaseDialDimensions]float64

	vecSinCos(sines[:], cosines[:], phases[:])
	dialAddPhases128Native(dial, cosines[:], sines[:])
}

/*
CopyAndNormalize returns a cloned, unit-normalized copy of the dial.
*/
func (dial PhaseDial) CopyAndNormalize() PhaseDial {
	out := make(PhaseDial, len(dial))
	copy(out, dial)

	return out.normalize()
}

/*
Rotate applies a global phase rotation e^{iθ} to each dimension.
Returns a new PhaseDial; the receiver is unchanged.
*/
func (dial PhaseDial) Rotate(angleRadians float64) PhaseDial {
	if len(dial) == 0 {
		return nil
	}

	out := make(PhaseDial, len(dial))

	if len(dial) == PhaseDialDimensions {
		dialRotate128Native(out, dial, angleRadians)

		return out
	}

	factor := cmplx.Rect(1.0, angleRadians)

	for index, value := range dial {
		out[index] = value * factor
	}

	return out
}

/*
Similarity returns cosine similarity between two PhaseDial vectors
(real part of normalized Hermitian inner product).
*/
func (dial PhaseDial) Similarity(other PhaseDial) float64 {
	if len(dial) != len(other) || len(dial) == 0 {
		return 0
	}

	if len(dial) == PhaseDialDimensions {
		return dialSimilarity128Native(dial, other)
	}

	return dialSimilarity128Scalar(dial, other)
}

/*
ComposeMidpoint returns Normalize(Normalize(a) + Normalize(b)).
*/
func (dial PhaseDial) ComposeMidpoint(other PhaseDial) PhaseDial {
	if len(dial) != len(other) || len(dial) == 0 {
		return nil
	}

	if len(dial) == PhaseDialDimensions {
		return dialComposeMidpoint128Native(dial, other)
	}

	return dialComposeMidpoint128Scalar(dial, other)
}

func (dial PhaseDial) norm() float64 {
	var total float64

	for _, v := range dial {
		re, im := real(v), imag(v)
		total += re*re + im*im
	}

	return math.Sqrt(total)
}

func (dial PhaseDial) normalize() PhaseDial {
	if len(dial) == PhaseDialDimensions {
		dialNormalize128Native(dial)

		return dial
	}

	var sumSq float64

	for _, val := range dial {
		re, im := real(val), imag(val)
		sumSq += re*re + im*im
	}

	if sumSq == 0 {
		return dial
	}

	inv := 1.0 / math.Sqrt(sumSq)

	for index := range dial {
		dial[index] = complex(real(dial[index])*inv, imag(dial[index])*inv)
	}

	return dial
}

/*
PhaseRotor is a PhaseDialDimensions-length array of PGA multivectors. Each
dimension uses a Fibonacci-lattice axis on S² with the same ω_k phase law
as PhaseDial, lifting planar phase into Cl(3,0,1) even subalgebra.
*/
type PhaseRotor []Multivector

/*
NewPhaseRotor allocates a zero-initialized PhaseRotor.
*/
func NewPhaseRotor() PhaseRotor {
	return make(PhaseRotor, PhaseDialDimensions)
}

/*
EncodeFromValues generates a PhaseRotor from a value sequence.
*/
func (rotor PhaseRotor) EncodeFromValues(values []ValueToken) PhaseRotor {
	if len(values) == 0 {
		return rotor
	}

	if len(rotor) < PhaseDialDimensions {
		rotor = NewPhaseRotor()
	}

	goldenAngle := math.Pi * (3 - math.Sqrt(5))
	nBasis := float64(PhaseDialDimensions)

	for k := 0; k < PhaseDialDimensions; k++ {
		theta := goldenAngle * float64(k)
		zCoord := 1 - (2*float64(k)+1)/nBasis
		radius := math.Sqrt(math.Max(0, 1-zCoord*zCoord))

		axisE23 := radius * math.Cos(theta)
		axisE31 := radius * math.Sin(theta)
		axisE12 := zCoord

		omega := float64(PhaseDialPrimes[k])

		var sum Multivector

		for t := range values {
			structuralPhase := structuralPhaseMix(&values[t])
			phase := (omega * float64(t+1) * 0.1) + (structuralPhase * math.Pi * 2)
			halfPhase := phase / 2
			sinHalf := math.Sin(halfPhase)
			cosHalf := math.Cos(halfPhase)

			sum[MvScalar] += cosHalf
			sum[MvE12] += sinHalf * axisE12
			sum[MvE31] += sinHalf * axisE31
			sum[MvE23] += sinHalf * axisE23
		}

		rotor[k] = sum.Normalize()
	}

	return rotor
}

/*
Similarity averages the scalar part of rotor[k]·other[k]† across dimensions.
*/
func (rotor PhaseRotor) Similarity(other PhaseRotor) float64 {
	if len(rotor) != len(other) || len(rotor) == 0 {
		return 0
	}

	if len(rotor) == PhaseDialDimensions {
		return rotorSimilarityAverage(rotor, other)
	}

	var dotSum float64

	for rotorIndex := range rotor {
		product := rotor[rotorIndex].GeometricProduct(other[rotorIndex].Reverse())
		dotSum += product[MvScalar]
	}

	return dotSum / float64(len(rotor))
}

/*
ToDialCompat projects each rotor to a unit complex number per dimension
for consumers that expect a PhaseDial.
*/
func (rotor PhaseRotor) ToDialCompat() PhaseDial {
	dial := make(PhaseDial, len(rotor))

	for k, mv := range rotor {
		eucNorm := math.Sqrt(
			mv[MvE12]*mv[MvE12] +
				mv[MvE31]*mv[MvE31] +
				mv[MvE23]*mv[MvE23],
		)

		angle := 2 * math.Atan2(eucNorm, mv[MvScalar])

		e12 := mv[MvE12]
		e31 := mv[MvE31]
		e23 := mv[MvE23]
		abs12 := math.Abs(e12)
		abs31 := math.Abs(e31)
		abs23 := math.Abs(e23)

		var dominant float64

		if abs12 >= abs31 && abs12 >= abs23 {
			dominant = e12
		} else if abs31 >= abs23 {
			dominant = e31
		} else {
			dominant = e23
		}

		if dominant < 0 {
			angle = -angle
		}

		dial[k] = cmplx.Rect(1.0, angle)
	}

	return dial.normalize()
}
