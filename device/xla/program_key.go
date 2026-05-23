package xla

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"sort"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
ProgramKey identifies a unique XLA computation for compile-cache lookup.
It captures operation identity, dtypes, shapes, scalar parameters, and target.
*/
type ProgramKey struct {
	Operation   string
	DTypes      []dtype.DType
	Shapes      []tensor.Shape
	FloatParams []float64
	IntParams   []int64
	Target      string
}

/*
Hash returns a stable digest for compile-cache indexing.
*/
func (programKey ProgramKey) Hash() [32]byte {
	hasher := sha256.New()
	writeProgramKey(hasher, programKey)
	var digest [32]byte
	copy(digest[:], hasher.Sum(nil))
	return digest
}

/*
String renders a diagnostic key for logs and test failures.
*/
func (programKey ProgramKey) String() string {
	return fmt.Sprintf(
		"xla:%s:target=%s:dtypes=%d:shapes=%d:floats=%d:ints=%d",
		programKey.Operation,
		programKey.Target,
		len(programKey.DTypes),
		len(programKey.Shapes),
		len(programKey.FloatParams),
		len(programKey.IntParams),
	)
}

func writeProgramKey(hasher hash.Hash, programKey ProgramKey) {
	_, _ = hasher.Write([]byte(programKey.Operation))
	_, _ = hasher.Write([]byte{0})
	_, _ = hasher.Write([]byte(programKey.Target))
	_, _ = hasher.Write([]byte{0})

	dtypeCodes := make([]int32, len(programKey.DTypes))
	for index, elementFormat := range programKey.DTypes {
		dtypeCodes[index] = int32(elementFormat)
	}

	sort.Slice(dtypeCodes, func(leftIndex, rightIndex int) bool {
		return dtypeCodes[leftIndex] < dtypeCodes[rightIndex]
	})

	for _, code := range dtypeCodes {
		_ = binary.Write(hasher, binary.LittleEndian, code)
	}

	for _, shape := range programKey.Shapes {
		dimensions := shape.Dims()
		_ = binary.Write(hasher, binary.LittleEndian, int32(len(dimensions)))

		for _, dimension := range dimensions {
			_ = binary.Write(hasher, binary.LittleEndian, int64(dimension))
		}
	}

	for _, value := range programKey.FloatParams {
		_ = binary.Write(hasher, binary.LittleEndian, value)
	}

	for _, value := range programKey.IntParams {
		_ = binary.Write(hasher, binary.LittleEndian, value)
	}
}
