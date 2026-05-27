package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestNewInitializesWorkspaceMap(testingObject *testing.T) {
	convey.Convey("Given a new execution backend", testingObject, func() {
		backend := New(nil)

		convey.Convey("It should own an empty workspace map", func() {
			convey.So(backend.Workspaces(), convey.ShouldNotBeNil)
		})
	})
}
