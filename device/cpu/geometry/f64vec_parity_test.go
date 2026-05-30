package geometry

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

var f64vecParitySizes = []int{1, 7, 64, 512, 8192}

func TestSumFloat64Parity(testingObject *testing.T) {
	convey.Convey("Given SumFloat64 kernels", testingObject, func() {
		for _, count := range f64vecParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				values := f64vecTestBuffer(count)
				got := sumFloat64Kernel(values)
				want := sumFloat64Scalar(values)
				convey.So(got, convey.ShouldAlmostEqual, want, 1e-9)
			})
		}
	})
}

func TestDotFloat64Parity(testingObject *testing.T) {
	convey.Convey("Given DotFloat64 kernels", testingObject, func() {
		for _, count := range f64vecParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left := f64vecTestBuffer(count)
				right := f64vecTestBuffer(count + 3)
				got := dotFloat64Kernel(left, right)
				want := dotFloat64Scalar(left, right)
				convey.So(got, convey.ShouldAlmostEqual, want, 1e-9)
			})
		}
	})
}

func TestMulFloat64Parity(testingObject *testing.T) {
	convey.Convey("Given MulFloat64 kernels", testingObject, func() {
		for _, count := range f64vecParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left := f64vecTestBuffer(count)
				right := f64vecTestBuffer(count + 5)
				got := make([]float64, count)
				want := make([]float64, count)
				mulFloat64Kernel(got, left, right)
				mulFloat64Scalar(want, left, right)
				convey.So(got, convey.ShouldResemble, want)
			})
		}
	})
}

func TestScaleFloat64Parity(testingObject *testing.T) {
	convey.Convey("Given ScaleFloat64 kernels", testingObject, func() {
		for _, count := range f64vecParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				source := f64vecTestBuffer(count)
				got := make([]float64, count)
				want := make([]float64, count)
				scaleFloat64Kernel(got, source, 1.75)
				scaleFloat64Scalar(want, source, 1.75)
				convey.So(got, convey.ShouldResemble, want)
			})
		}
	})
}

func BenchmarkDotFloat64512(benchmark *testing.B) {
	left := f64vecTestBuffer(512)
	right := f64vecTestBuffer(512)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_ = dotFloat64Kernel(left, right)
	}
}

func f64vecTestBuffer(count int) []float64 {
	values := make([]float64, count)

	for index := range values {
		values[index] = math.Sin(float64(index%17)) * 2.5
	}

	return values
}
