package xla

import (
	"sync"

	"github.com/theapemachine/manifesto/dtype"
)

/*
XLAElementType is the portable element-type token used before PJRT bridge wiring.
*/
type XLAElementType int

const (
	XLAElementInvalid XLAElementType = iota
	XLAElementF64
	XLAElementF32
	XLAElementF16
	XLAElementBF16
	XLAElementF8E4M3
	XLAElementF8E5M2
	XLAElementS64
	XLAElementS32
	XLAElementS16
	XLAElementS8
	XLAElementU64
	XLAElementU32
	XLAElementU16
	XLAElementU8
	XLAElementPred
)

/*
MapDType converts manifesto dtype values into XLA element tokens.
*/
func MapDType(elementFormat dtype.DType) (XLAElementType, error) {
	switch elementFormat {
	case dtype.Float64:
		return XLAElementF64, nil
	case dtype.Float32:
		return XLAElementF32, nil
	case dtype.Float16:
		return XLAElementF16, nil
	case dtype.BFloat16:
		return XLAElementBF16, nil
	case dtype.Float8E4M3:
		return XLAElementF8E4M3, nil
	case dtype.Float8E5M2:
		return XLAElementF8E5M2, nil
	case dtype.Int64:
		return XLAElementS64, nil
	case dtype.Int32:
		return XLAElementS32, nil
	case dtype.Int16:
		return XLAElementS16, nil
	case dtype.Int8:
		return XLAElementS8, nil
	case dtype.Uint64:
		return XLAElementU64, nil
	case dtype.Uint32:
		return XLAElementU32, nil
	case dtype.Uint16:
		return XLAElementU16, nil
	case dtype.Uint8:
		return XLAElementU8, nil
	case dtype.Bool:
		return XLAElementPred, nil
	default:
		return XLAElementInvalid, errUnsupportedDType
	}
}

var errUnsupportedDType = &loweringError{message: "unsupported XLA dtype"}

/*
loweringError reports dtype or shape failures during lowering setup.
*/
type loweringError struct {
	message string
}

func (loweringError *loweringError) Error() string {
	return loweringError.message
}

/*
SupportedDTypeSet is the canonical dtype list mirrored by Backend.SupportedDTypes.
*/
func SupportedDTypeSet() []dtype.DType {
	return []dtype.DType{
		dtype.Float64,
		dtype.Float32,
		dtype.Float16,
		dtype.BFloat16,
		dtype.Float8E4M3,
		dtype.Float8E5M2,
		dtype.Int64,
		dtype.Int32,
		dtype.Int16,
		dtype.Int8,
		dtype.Uint64,
		dtype.Uint32,
		dtype.Uint16,
		dtype.Uint8,
		dtype.Bool,
	}
}

/*
DTypeRegistry maps every supported dtype to an XLA element token.
*/
type DTypeRegistry struct {
	mutex sync.RWMutex
	cache map[dtype.DType]XLAElementType
}

/*
NewDTypeRegistry constructs a registry seeded from SupportedDTypeSet.
*/
func NewDTypeRegistry() *DTypeRegistry {
	registry := &DTypeRegistry{cache: make(map[dtype.DType]XLAElementType)}

	for _, elementFormat := range SupportedDTypeSet() {
		mapped, err := MapDType(elementFormat)

		if err != nil {
			continue
		}

		registry.cache[elementFormat] = mapped
	}

	return registry
}

/*
Lookup returns the XLA element token for a dtype.
*/
func (dtypeRegistry *DTypeRegistry) Lookup(elementFormat dtype.DType) (XLAElementType, error) {
	dtypeRegistry.mutex.RLock()
	mapped, ok := dtypeRegistry.cache[elementFormat]
	dtypeRegistry.mutex.RUnlock()

	if !ok {
		return XLAElementInvalid, errUnsupportedDType
	}

	return mapped, nil
}
