package execution

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"

	"github.com/theapemachine/hf/safetensors"
	"github.com/theapemachine/manifesto/resolve"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/types"
)

/*
CheckpointBundle holds the safetensors parser used at compile time and the
resident weight store used at graph.call dispatch time for one archive.
*/
type CheckpointBundle struct {
	Parser types.Parser
	Store  *ResidentStore
}

/*
LoadSafetensorsArchive parses one in-memory safetensors archive, uploads every
tensor into the supplied memory backend, and returns compile-time and runtime
handles for the same checkpoint.
*/
func LoadSafetensorsArchive(archive []byte, memory tensor.Backend) (*CheckpointBundle, error) {
	if len(archive) == 0 {
		return nil, fmt.Errorf("execution: safetensors archive is required")
	}

	if memory == nil {
		return nil, fmt.Errorf("execution: memory backend is required")
	}

	parser, err := safetensors.NewParser(archive)

	if err != nil {
		return nil, fmt.Errorf("execution: parse safetensors: %w", err)
	}

	store := NewResidentStore(memory)

	if len(archive) < 8 {
		_ = store.Close()
		return nil, fmt.Errorf("execution: safetensors archive too small")
	}

	headerLength := binary.LittleEndian.Uint64(archive[:8])
	dataBase := int64(8) + int64(headerLength)

	for token := range parser.Generate() {
		if token.Kind != types.KindTensor {
			continue
		}

		start := dataBase + token.Span.Offset
		end := start + token.Span.Length

		if start < 0 || end < start || end > int64(len(archive)) {
			_ = store.Close()
			return nil, fmt.Errorf("execution: tensor %q offsets out of bounds", token.Name)
		}

		rawBytes := archive[start:end]
		resident, uploadErr := uploadTokenTensor(memory, token, rawBytes)

		if uploadErr != nil {
			_ = store.Close()
			return nil, fmt.Errorf("execution: upload tensor %q: %w", token.Name, uploadErr)
		}

		store.RegisterTensor(token.Name, token, resident)
	}

	return &CheckpointBundle{
		Parser: parser,
		Store:  store,
	}, nil
}

/*
LoadSafetensorsFile reads one safetensors file from disk and uploads it.
*/
func LoadSafetensorsFile(path string, memory tensor.Backend) (*CheckpointBundle, error) {
	rawBytes, err := os.ReadFile(path)

	if err != nil {
		return nil, fmt.Errorf("execution: read safetensors %q: %w", path, err)
	}

	return LoadSafetensorsArchive(rawBytes, memory)
}

/*
DownloadSafetensorsBundle resolves one HuggingFace repo component, downloads
its primary safetensors archive, and uploads all tensors into memory.
*/
func DownloadSafetensorsBundle(
	ctx context.Context,
	hubClient resolve.Hub,
	location resolve.RepoLocation,
	subfolder string,
	cacheDir string,
	memory tensor.Backend,
) (*CheckpointBundle, string, error) {
	if hubClient == nil {
		return nil, "", fmt.Errorf("execution: hub client is required")
	}

	resolver := resolve.NewResolver(hubClient)
	weightFile, err := resolver.PrimaryWeightFile(ctx, location, subfolder, cacheDir)

	if err != nil {
		return nil, "", fmt.Errorf("execution: resolve weight file: %w", err)
	}

	reader, file, err := hubClient.Open(ctx, location, weightFile, cacheDir)

	if err != nil {
		return nil, "", fmt.Errorf("execution: open weight file %q: %w", weightFile, err)
	}

	defer reader.Close()

	rawBytes, err := os.ReadFile(file.Path)

	if err != nil {
		return nil, "", fmt.Errorf("execution: read weight file %q: %w", file.Path, err)
	}

	bundle, err := LoadSafetensorsArchive(rawBytes, memory)

	if err != nil {
		return nil, "", err
	}

	return bundle, file.Path, nil
}
