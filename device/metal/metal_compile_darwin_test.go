//go:build darwin

package metal

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestMetallibgenCompile(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal compilation requires darwin")
	}

	if _, err := exec.LookPath("xcrun"); err != nil {
		t.Skip("xcrun not available")
	}

	convey.Convey("Given the Metal kernel tree", t, func() {
		packageDir, err := os.Getwd()
		convey.So(err, convey.ShouldBeNil)

		metallibgen := filepath.Join(packageDir, "internal", "metallibgen")
		command := exec.Command("go", "run", metallibgen)
		command.Dir = packageDir
		_, err = command.CombinedOutput()

		convey.So(err, convey.ShouldBeNil)

		info, err := os.Stat(filepath.Join(packageDir, "kernels.metallib"))
		convey.So(err, convey.ShouldBeNil)
		convey.So(info.Size(), convey.ShouldBeGreaterThan, 0)
	})
}
