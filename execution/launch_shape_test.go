package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
)

func TestSubstituteLaunchDimensions(test *testing.T) {
	convey.Convey("Given planner max N=4096 and live N=42", test, func() {
		maxBindings := ir.SymbolMap{"N": 4096, "T": 4096}
		launchBindings := ir.SymbolMap{"N": 42, "T": 42}

		convey.Convey("It should rewrite matching dimensions", func() {
			dims := substituteLaunchDimensions([]int{4096, 2048}, maxBindings, launchBindings)
			convey.So(dims, convey.ShouldResemble, []int{42, 2048})
		})

		convey.Convey("It should leave unrelated dimensions unchanged", func() {
			dims := substituteLaunchDimensions([]int{32, 64}, maxBindings, launchBindings)
			convey.So(dims, convey.ShouldResemble, []int{32, 64})
		})

		convey.Convey("It should not rewrite unmarked static dimensions", func() {
			dims := substituteMarkedLaunchDimensions(
				[]int{1, 4096, 4096},
				[]bool{false, true, false},
				maxBindings,
				launchBindings,
			)
			convey.So(dims, convey.ShouldResemble, []int{1, 42, 4096})
		})
	})
}
