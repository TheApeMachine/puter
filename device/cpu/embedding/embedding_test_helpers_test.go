package embedding

import (
	"math/rand"
)

func randomEmbeddingTable(vocab, hidden int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	table := make([]float32, vocab*hidden)

	for index := range table {
		table[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return table
}

func randomEmbeddingIndices(indexCount, vocab int, seed int64) []int32 {
	rng := rand.New(rand.NewSource(seed))
	indices := make([]int32, indexCount)

	for index := range indices {
		indices[index] = int32(rng.Intn(vocab))
	}

	return indices
}
