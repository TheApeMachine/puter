package activation

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestActivationReducedLUTParity(t *testing.T) {
	convey.Convey("Given f16/bf16 LUT activation paths", t, func() {
		type lutCase struct {
			name   string
			maxULP int
			fill   func(float64) float64
			run    func(dst, src unsafe.Pointer, count int, format dtype.DType)
		}

		cases := []lutCase{
			{"Exp", 2, math.FastExp64, Exp},
			{"Log", 2, math.FastLog64, Log},
			{"Log1p", 2, math.FastLog1p64, Log1p},
			{"Expm1", 2, math.FastExpm1_64, Expm1},
			{"Sigmoid", 2, math.FastSigmoid64, Sigmoid},
			{"LogSigmoid", 2, math.FastLogSigmoid64, LogSigmoid},
			{"Tanh", 2, math.FastTanh64, Tanh},
			{"Silu", 2, math.FastSilu64, Silu},
			{"GeluTanh", 2, math.FastGeluTanh64, GeluTanh},
			{"Gelu", 2, math.FastGelu64, Gelu},
			{"ReLU", 1, math.FastReLU64, ReLU},
			{"LeakyReLU", 1, math.FastLeakyReLU64, LeakyReLU},
			{"ELU", 2, math.FastELU64, ELU},
			{"CELU", 2, math.FastCELU64, CELU},
			{"SELU", 2, math.FastSELU64, SELU},
			{"Softplus", 2, math.FastSoftplus64, Softplus},
			{"Mish", 2, math.FastMish64, Mish},
			{"Softsign", 2, math.FastSoftsign64, Softsign},
			{"HardSigmoid", 1, math.FastHardSigmoid64, HardSigmoid},
			{"HardSwish", 1, math.FastHardSwish64, HardSwish},
			{"HardTanh", 1, math.FastHardTanh64, HardTanh},
			{"HardGelu", 1, math.FastHardGelu64, HardGelu},
			{"QuickGelu", 1, math.FastQuickGelu64, QuickGelu},
			{"TanhShrink", 2, math.FastTanhShrink64, TanhShrink},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
					storageDType := storageDType

					convey.Convey(storageDType.Name(), func() {
						for _, count := range parity.Lengths {
							convey.Convey(fmt.Sprintf("N=%d", count), func() {
								source := randomReducedInput(count, int64(0x5100+count), storageDType)
								want := applyScalarReduced(source, storageDType, testCase.fill)
								gotStorage := make([]uint16, count)

								testCase.run(
									unsafe.Pointer(&gotStorage[0]),
									unsafe.Pointer(&source[0]),
									count,
									storageDType,
								)

								got := decodeReduced(gotStorage, storageDType)
								parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULP)
							})
						}
					})
				}
			})
		}
	})
}

func BenchmarkActivationF16LUTExp(b *testing.B) {
	count := 8192
	source := randomReducedInput(count, 1, dtype.Float16)
	destination := make([]uint16, count)

	b.ResetTimer()

	for b.Loop() {
		Default.Exp(
			unsafe.Pointer(&destination[0]),
			unsafe.Pointer(&source[0]),
			count,
			dtype.Float16,
		)
	}
}

func BenchmarkActivationBF16LUTExp(b *testing.B) {
	count := 8192
	source := randomReducedInput(count, 2, dtype.BFloat16)
	destination := make([]uint16, count)

	b.ResetTimer()

	for b.Loop() {
		Default.Exp(
			unsafe.Pointer(&destination[0]),
			unsafe.Pointer(&source[0]),
			count,
			dtype.BFloat16,
		)
	}
}

func randomReducedInput(count int, seed int64, format dtype.DType) []uint16 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]uint16, count)

	for index := range values {
		floatValue := rng.Float32()*4.0 - 2.0

		if format == dtype.BFloat16 {
			values[index] = uint16(dtype.NewBfloat16FromFloat32(floatValue))
			continue
		}

		values[index] = dtype.Fromfloat32(floatValue).Bits()
	}

	return values
}

func applyScalarReduced(
	source []uint16,
	format dtype.DType,
	fill func(float64) float64,
) []float32 {
	output := make([]float32, len(source))

	for index, bits := range source {
		switch format {
		case dtype.Float16:
			inputValue := dtype.Frombits(bits).Float32()
			rounded := dtype.Fromfloat32(float32(fill(float64(inputValue))))
			output[index] = rounded.Float32()
		case dtype.BFloat16:
			bf16Input := dtype.BF16(bits)
			inputValue := (&bf16Input).Float32()
			rounded := dtype.NewBfloat16FromFloat32(float32(fill(float64(inputValue))))
			bf16Rounded := dtype.BF16(rounded)
			output[index] = (&bf16Rounded).Float32()
		default:
			panic("activation parity: unsupported dtype")
		}
	}

	return output
}

func decodeReduced(source []uint16, format dtype.DType) []float32 {
	output := make([]float32, len(source))

	for index, bits := range source {
		switch format {
		case dtype.Float16:
			output[index] = dtype.Frombits(bits).Float32()
		case dtype.BFloat16:
			bf16Value := dtype.BF16(bits)
			output[index] = (&bf16Value).Float32()
		default:
			panic("activation parity: unsupported dtype")
		}
	}

	return output
}
