package runner

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
)

func TestTimestepDivisorFromManifestGraph(testingObject *testing.T) {
	convey.Convey("Given a FLUX topology with timestep_divisor", testingObject, func() {
		graph := &ast.Graph{
			Nodes: []*ast.GraphNode{
				{
					ID: "time_guidance_embed.time_proj",
					Op: "embedding.timestep",
					Attributes: map[string]any{
						"timestep_divisor": 1000,
					},
				},
			},
		}

		convey.Convey("It should read the divisor from embedding.timestep config", func() {
			convey.So(timestepDivisorFromManifestGraph(graph), convey.ShouldEqual, 1000)
		})
	})
}

func TestScaleTimestepProgramInput(testingObject *testing.T) {
	convey.Convey("Given a scheduler timestep in train-scale units", testingObject, func() {
		convey.Convey("It should scale float32 timesteps for the denoiser", func() {
			scaled, err := scaleTimestepProgramInput(float32(882.5), 1000)

			convey.So(err, convey.ShouldBeNil)
			convey.So(scaled, convey.ShouldEqual, float32(0.8825))
		})
	})
}
