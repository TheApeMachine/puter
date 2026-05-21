package runner

import (
	"github.com/theapemachine/manifesto/ast"
)

/*
manifestBindings indexes manifest graph nodes by ID for weight lookup.
*/
type manifestBindings struct {
	byNodeID map[string]*ast.GraphNode
}

func newManifestBindings(manifestGraph *ast.Graph) *manifestBindings {
	bindings := &manifestBindings{
		byNodeID: make(map[string]*ast.GraphNode),
	}

	if manifestGraph == nil {
		return bindings
	}

	for _, node := range manifestGraph.Nodes {
		bindings.byNodeID[node.ID] = node
	}

	return bindings
}

func (bindings *manifestBindings) node(nodeID string) *ast.GraphNode {
	if bindings == nil {
		return nil
	}

	return bindings.byNodeID[nodeID]
}

func (bindings *manifestBindings) weightTensorName(nodeID string) string {
	node := bindings.node(nodeID)

	if node == nil || node.Weights == nil {
		return ""
	}

	return node.Weights.TensorName
}

func (bindings *manifestBindings) biasTensorName(nodeID string) string {
	node := bindings.node(nodeID)

	if node == nil || node.Attributes == nil {
		return ""
	}

	biasName, ok := node.Attributes["bias_weight"].(string)

	if !ok {
		return ""
	}

	return biasName
}
