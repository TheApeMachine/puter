package geometry

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/puter/device"
)

/*
FrequencySpread is the number of octaves to spread the frequency across.
*/
var FrequencySpread = math.Log2(float64(512))

/*
propertyWords is the number of uint64 words in the canonical 512-bit properties region
(words 48–55 on the Value layout).
*/
const propertyWords = 8

/*
phaseScalarSumThresholdLog2 is the inclusive byte-count ceiling for the allocation-free
scalar sin/cos accumulation path in SeqCircularMeanPhaseFromPhases (matches log2(512)).
*/
const phaseScalarSumThresholdLog2 = 9

/*
symbolFromPropertyWord maps one 64-bit lane to a matrix index 0…511. Callers that hold a
full properties snapshot should fold first (e.g. SymbolFromPropertyBand).
*/
func symbolFromPropertyWord(word uint64) int {
	return int(word & 511)
}

/*
SymbolFromPropertyBand folds the eight-word properties region into one 0…511 symbol for
transition statistics (xor-mix, then low 9 bits). Same symbol used for co-occurrence rows.
*/
func SymbolFromPropertyBand(words []uint64) int {
	wordCount := len(words)

	if wordCount == 0 {
		return 0
	}

	if wordCount > propertyWords {
		wordCount = propertyWords
	}

	var mix uint64

	for index := 0; index < wordCount; index++ {
		word := words[index]
		mix ^= word
		mix ^= word >> 32
	}

	return int(mix & 511)
}

/*
EigenModeToroidal

Maps oscillators to an initial phase angle from the forward transition
statistics of **property symbols** (512×512 Markov matrix), via eigendecomposition.
Rows/columns index 0…511 — the same width as the properties band (8×64 bits) used
as discrete tags for forward potential / eigenmode phases (see README “Properties”).

Co-occurrence is computed at every FibWindow scale.

Algorithm:
 1. Build asymmetric 512×512 forward transition matrix T per FibWindow.
 2. Extract top 3 eigenvectors via biorthogonal power iteration.
 3. Skip v1 (all-positive by Perron-Frobenius).
 4. Phase[i] = atan2(v3[i], v2[i]). Combine across scales via weighted circular mean.
*/
type EigenModeToroidal struct {
	ctx       context.Context
	cancel    context.CancelFunc
	phaser    *Phase
	phase     [512]float64
	frequency []float64
	affinity  []uint64
	err       error
}

type eigenOpts func(*EigenModeToroidal)

func NewEigenModeToroidal(opts ...eigenOpts) (*EigenModeToroidal, error) {
	emt := &EigenModeToroidal{
		phase:     [512]float64{},
		frequency: []float64{},
		phaser:    NewPhase(),
	}

	for _, opt := range opts {
		opt(emt)
	}

	if emt.ctx == nil || emt.cancel == nil || emt.phaser == nil || len(emt.affinity) == 0 {
		return nil, fmt.Errorf("eigenmode toroidal: missing required fields")
	}

	return emt, nil
}

/*
BuildCooccurrence builds the co-occurrence matrix and computes the top 3 eigenvectors.

Each input byte is mapped to a symbol 0…255 via its numeric value (same index family as
SeqCircularMeanPhase). For arbitrary uint64 tags per timestep, use BuildCooccurrenceFromWords.
*/
func (emt *EigenModeToroidal) BuildCooccurrence(corpus []byte, windowSize int) {
	tags := make([]uint64, len(corpus))

	for index, b := range corpus {
		tags[index] = uint64(b)
	}

	emt.BuildCooccurrenceFromWords(tags, windowSize)
}

/*
BuildCooccurrenceFromWords is the same Markov lift as BuildCooccurrence, but each timestep is
already a uint64 tag (e.g. one lane or a pre-mixed property key). Symbols are 0…511 via
symbolFromPropertyWord.
*/
func (emt *EigenModeToroidal) BuildCooccurrenceFromWords(tags []uint64, windowSize int) {
	if len(emt.frequency) < device.EigenSymbolDimensions {
		emt.frequency = make([]float64, device.EigenSymbolDimensions)
	}

	eigenToroidalFromTags(emt.phase[:], emt.frequency, tags, windowSize)
}

/*
eigenToroidalFromTags builds the 512×512 forward transition matrix from tag
timesteps, extracts the 2nd/3rd eigenvectors, and writes phase and frequency
maps into the caller-provided slices (each len ≥ EigenSymbolDimensions).
*/
func eigenToroidalFromTags(
	phaseDestination, frequencyDestination []float64,
	tags []uint64,
	windowSize int,
) {
	emt := &EigenModeToroidal{frequency: frequencyDestination}

	var transition [512][512]float64

	emt.buildCooccurrenceInto(&transition, tags, windowSize)

	_, eigenvectorTwo, eigenvectorThree := emt.top3Eigenvectors(&transition)

	var eigenTwoSquared, eigenThreeSquared, magnitudes [512]float64

	vecMul(eigenTwoSquared[:], eigenvectorTwo[:], eigenvectorTwo[:])
	vecMul(eigenThreeSquared[:], eigenvectorThree[:], eigenvectorThree[:])
	vecAdd(magnitudes[:], eigenTwoSquared[:], eigenThreeSquared[:])
	vecSqrt(magnitudes[:], magnitudes[:])

	maxMagnitude := vecMax(magnitudes[:])

	vecAtan2(phaseDestination, eigenvectorThree[:], eigenvectorTwo[:])

	if maxMagnitude > 0 {
		vecScale(frequencyDestination, magnitudes[:], FrequencySpread/maxMagnitude)
		vecAddScalar(frequencyDestination, frequencyDestination, 1.0)
		return
	}

	for index := range device.EigenSymbolDimensions {
		frequencyDestination[index] = 1.0
	}
}

/*
buildCooccurrenceInto fills C with the asymmetric forward transition matrix.
T[i][j] counts how often symbol i PRECEDES symbol j within windowSize positions.
Only forward neighbors (j > pos) are counted, making T asymmetric.

Rows are L1-normalized (sum = 1) so C behaves as a Markov transition matrix.
*/
func (emt *EigenModeToroidal) buildCooccurrenceInto(
	C *[512][512]float64,
	corpus []uint64,
	windowSize int,
) {
	for rowIndex := range 512 {
		for columnIndex := range 512 {
			C[rowIndex][columnIndex] = 0
		}
	}

	for position := range corpus {
		symbol := symbolFromPropertyWord(corpus[position])
		end := min(len(corpus), position+windowSize+1)

		for inner := position + 1; inner < end; inner++ {
			otherSymbol := symbolFromPropertyWord(corpus[inner])
			C[symbol][otherSymbol] += 1.0
		}
	}

	for rowIndex := range 512 {
		sum := vecSum(C[rowIndex][:])

		if sum > 0 {
			vecScale(C[rowIndex][:], C[rowIndex][:], 1.0/sum)
		}
	}
}

/*
top3Eigenvectors returns the top 3 right eigenvectors of the transition matrix
via biorthogonal power iteration with deflation.
*/
func (emt *EigenModeToroidal) top3Eigenvectors(
	C *[512][512]float64,
) (v1, v2, v3 [512]float64) {
	v1, lambda1 := emt.powerIterate(C, emt.uniformStart())
	left1 := emt.powerIterateLeft(C, emt.uniformStart())

	C2 := emt.deflateBiorthogonal(C, &left1, &v1, lambda1)
	v2, lambda2 := emt.powerIterate(&C2, emt.sawtoothStart())
	left2 := emt.powerIterateLeft(&C2, emt.sawtoothStart())

	C3 := emt.deflateBiorthogonal(&C2, &left2, &v2, lambda2)
	v3, _ = emt.powerIterate(&C3, emt.cosineStart())

	return emt.alignAndNormalizeTop3(&v1, &v2, &v3)
}

/*
alignAndNormalizeTop3 flips the leading eigenvector sign to nonnegative sum, then L2-normalizes
all three vectors returned for phase mapping.
*/
func (emt *EigenModeToroidal) alignAndNormalizeTop3(v1, v2, v3 *[512]float64) (v1out, v2out, v3out [512]float64) {
	var v1Sum float64

	for rowIndex := range 512 {
		v1Sum += v1[rowIndex]
	}

	if v1Sum < 0 {
		for rowIndex := range 512 {
			v1[rowIndex] = -v1[rowIndex]
		}
	}

	emt.normalizeVec(v1)
	emt.normalizeVec(v2)
	emt.normalizeVec(v3)

	return *v1, *v2, *v3
}

/*
powerIterate runs power iteration on M until convergence or maxIter steps.
Returns the dominant right eigenvector and its eigenvalue (Rayleigh quotient).
*/
func (emt *EigenModeToroidal) powerIterate(
	M *[512][512]float64, v [512]float64,
) ([512]float64, float64) {
	const maxIter = 2000
	const tol = 1e-10

	emt.normalizeVec(&v)
	var lambda float64

	for range maxIter {
		var mv [512]float64
		matVec512RowMajor(&mv, M, &v)
		newLambda := vecDotProduct(v[:], mv[:])

		normSq := vecSumOfSquares(mv[:])
		if normSq < 1e-24 {
			break
		}
		norm := math.Sqrt(normSq)
		vecScale(mv[:], mv[:], 1.0/norm)

		if math.Abs(newLambda-lambda) < tol {
			return mv, newLambda
		}
		v = mv
		lambda = newLambda
	}
	return v, lambda
}

/*
powerIterateLeft runs power iteration on Mᵀ to find the dominant left eigenvector u.
uᵀ M = λ uᵀ  ⟺  Mᵀ u = λ u. Matrix-vector product: (Mᵀ u)_i = Σ_j M[j][i] u[j].
*/
func (emt *EigenModeToroidal) powerIterateLeft(
	M *[512][512]float64, start [512]float64,
) [512]float64 {
	const maxIter = 2000
	const tol = 1e-10

	u := start
	emt.normalizeVec(&u)
	var lambda float64

	for range maxIter {
		var Mu [512]float64
		matVec512ColMajor(&Mu, M, &u)
		newLambda := vecDotProduct(u[:], Mu[:])

		normSq := vecSumOfSquares(Mu[:])
		if normSq < 1e-24 {
			break
		}
		norm := math.Sqrt(normSq)
		vecScale(Mu[:], Mu[:], 1.0/norm)

		if math.Abs(newLambda-lambda) < tol {
			return Mu
		}
		u = Mu
		lambda = newLambda
	}
	return u
}

/*
deflateBiorthogonal removes the rank-1 component corresponding to right eigenvector v
and left eigenvector u with eigenvalue λ. For asymmetric M, the correct deflation is
M_new = M - λ v uᵀ / (uᵀ v). Element: D[i][j] = M[i][j] - λ v[i] u[j] / dot(u,v).
*/
func (emt *EigenModeToroidal) deflateBiorthogonal(
	M *[512][512]float64,
	u, v *[512]float64,
	lam float64,
) [512][512]float64 {
	uTv := vecDotProduct(u[:], v[:])

	if math.Abs(uTv) < 1e-10 {
		return *M
	}

	scale := lam / uTv

	var D [512][512]float64

	for rowIndex := range 512 {
		for columnIndex := range 512 {
			D[rowIndex][columnIndex] = M[rowIndex][columnIndex] - scale*v[rowIndex]*u[columnIndex]
		}
	}

	return D
}

func (emt *EigenModeToroidal) uniformStart() [512]float64 {
	var v [512]float64

	val := 1.0 / math.Sqrt(float64(512))

	for index := range 512 {
		v[index] = val
	}

	return v
}

func (emt *EigenModeToroidal) sawtoothStart() [512]float64 {
	var (
		v                [512]float64
		intervalMidpoint = 0.5
	)

	for index := range 512 {
		v[index] = float64(index)/float64(512) - intervalMidpoint
	}

	emt.normalizeVec(&v)
	return v
}

func (emt *EigenModeToroidal) cosineStart() [512]float64 {
	var v [512]float64

	for index := range 512 {
		v[index] = 2 * math.Pi * float64(index) / float64(512)
	}

	vecCos(v[:], v[:])
	emt.normalizeVec(&v)

	return v
}

/*
SeqCircularMeanPhase returns the circular mean of the eigen phases of each
byte in seq. Two sequences sharing common bytes get nearby phases, so contexts
built from similar characters cluster together in eigen phase space.

For short sequences, uses scalar loop to avoid slice allocations.
*/
func (emt *EigenModeToroidal) SeqCircularMeanPhase(seq []byte) (float64, error) {
	return SeqCircularMeanPhaseFromPhases(&emt.phase, seq)
}

/*
SeqCircularMeanPhaseFromPhases returns the circular mean of phases for bytes in seq.
Consumers that receive PhasesSnapshot from the eigen group use this. Shared logic.
*/
func SeqCircularMeanPhaseFromPhases(phase *[512]float64, seq []byte) (float64, error) {
	if len(seq) == 0 {
		return 0, EigenErrorEmptySequence
	}

	return eigenCircularMeanPhase(phase[:], seq), nil
}

func eigenCircularMeanPhase(phaseTable []float64, sequence []byte) float64 {
	sequenceLength := len(sequence)

	if sequenceLength <= phaseScalarSumThresholdLog2 {
		var sineSum, cosineSum float64

		for _, byteValue := range sequence {
			phaseAngle := phaseTable[byteValue]
			sineSum += math.Sin(phaseAngle)
			cosineSum += math.Cos(phaseAngle)
		}

		return math.Atan2(sineSum, cosineSum)
	}

	phases := make([]float64, sequenceLength)

	for index, byteValue := range sequence {
		phases[index] = phaseTable[byteValue]
	}

	sineBuffer := make([]float64, sequenceLength)
	cosineBuffer := make([]float64, sequenceLength)
	vecSinCos(sineBuffer, cosineBuffer, phases)

	return math.Atan2(vecSum(sineBuffer), vecSum(cosineBuffer))
}

func (emt *EigenModeToroidal) normalizeVec(v *[512]float64) {
	normSq := vecSumOfSquares(v[:])

	if normSq < 1e-10 {
		return
	}

	norm := math.Sqrt(normSq)
	vecScale(v[:], v[:], 1.0/norm)
}

func EigenWithContext(ctx context.Context) eigenOpts {
	return func(emt *EigenModeToroidal) {
		emt.ctx = ctx
	}
}

type EigenError string

const (
	EigenErrorEmptySequence EigenError = "sequence is empty"
)

func (e EigenError) Error() string {
	return string(e)
}
