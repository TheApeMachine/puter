package xla

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func mustShape(testingHandle testing.TB, dimensions ...int) tensor.Shape {
	testingHandle.Helper()

	shape, err := tensor.NewShape(dimensions)

	if err != nil {
		testingHandle.Fatalf("shape: %v", err)
	}

	return shape
}

func TestProgramKeyHash(t *testing.T) {
	convey.Convey("Given program keys", t, func() {
		base := ProgramKey{
			Operation: "add",
			DTypes:    []dtype.DType{dtype.Float32, dtype.Float32},
			Shapes: []tensor.Shape{
				mustShape(t, 1024),
				mustShape(t, 1024),
			},
			Target: "gpu",
		}

		convey.Convey("It should be stable for identical keys", func() {
			other := base
			convey.So(base.Hash(), convey.ShouldResemble, other.Hash())
		})

		convey.Convey("It should differ when dtype changes", func() {
			other := base
			other.DTypes = []dtype.DType{dtype.Float16, dtype.Float16}
			convey.So(base.Hash(), convey.ShouldNotResemble, other.Hash())
		})

		convey.Convey("It should differ when shape changes", func() {
			other := base
			other.Shapes = []tensor.Shape{
				mustShape(t, 8192),
				mustShape(t, 8192),
			}
			convey.So(base.Hash(), convey.ShouldNotResemble, other.Hash())
		})
	})
}

func TestExecutableCacheReuse(t *testing.T) {
	convey.Convey("Given an executable cache", t, func() {
		cache := NewExecutableCache()
		programKey := ProgramKey{
			Operation: "relu",
			DTypes:    []dtype.DType{dtype.Float32},
			Shapes:    []tensor.Shape{mustShape(t, 64)},
			Target:    "gpu",
		}

		executable := &CompiledExecutable{key: programKey, handle: 1}

		convey.Convey("It should store and retrieve the same executable", func() {
			cache.Store(programKey, executable)
			got, ok := cache.Lookup(programKey)
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(got, convey.ShouldEqual, executable)
			convey.So(cache.Len(), convey.ShouldEqual, 1)
		})
	})
}

func TestMapDType(t *testing.T) {
	convey.Convey("Given supported dtypes", t, func() {
		for _, elementFormat := range SupportedDTypeSet() {
			convey.Convey(elementFormat.String(), func() {
				mapped, err := MapDType(elementFormat)
				convey.So(err, convey.ShouldBeNil)
				convey.So(mapped, convey.ShouldNotEqual, XLAElementInvalid)
			})
		}
	})
}

func TestBroadcastShape(t *testing.T) {
	convey.Convey("Given broadcast-compatible shapes", t, func() {
		left := mustShape(t, 4, 1)
		right := mustShape(t, 3)
		got, err := BroadcastShape(left, right)
		convey.So(err, convey.ShouldBeNil)
		convey.So(got.Dims(), convey.ShouldResemble, []int{4, 3})
	})
}

func BenchmarkProgramKeyHash(b *testing.B) {
	programKey := ProgramKey{
		Operation:   "gelu",
		DTypes:      []dtype.DType{dtype.BFloat16},
		Shapes:      []tensor.Shape{mustShape(b, 8192)},
		FloatParams: []float64{1.0},
		Target:      "gpu",
	}

	for b.Loop() {
		_ = programKey.Hash()
	}
}
