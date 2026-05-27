package attention

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestStableSoftmaxRowNative(testingObject *testing.T) {
	convey.Convey("Given a softmax row with positive infinity", testingObject, func() {
		scores := []float32{1, float32(math.Inf(1)), 2}

		StableSoftmaxRowNative(scores)

		convey.Convey("It should assign all weight to the infinite maximum", func() {
			convey.So(scores, convey.ShouldResemble, []float32{0, 1, 0})
		})
	})

	convey.Convey("Given a softmax row with multiple positive infinities", testingObject, func() {
		scores := []float32{float32(math.Inf(1)), 1, float32(math.Inf(1))}

		StableSoftmaxRowNative(scores)

		convey.Convey("It should split weight across the infinite maxima", func() {
			convey.So(scores, convey.ShouldResemble, []float32{0.5, 0, 0.5})
		})
	})

	convey.Convey("Given a softmax row with only negative infinities", testingObject, func() {
		scores := []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}

		StableSoftmaxRowNative(scores)

		convey.Convey("It should leave no NaN weights", func() {
			convey.So(scores, convey.ShouldResemble, []float32{0, 0})
		})
	})
}

func BenchmarkStableSoftmaxRowNative(benchmark *testing.B) {
	scores := make([]float32, 1024)

	for index := range scores {
		scores[index] = float32(index%17) * 0.125
	}

	for benchmark.Loop() {
		StableSoftmaxRowNative(scores)
	}
}
