package execution

import (
	"context"
	"encoding/binary"
	"fmt"
	"iter"
	"os"
	"strings"

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
	weightFiles, err := resolver.WeightFiles(ctx, location, subfolder, cacheDir)

	if err != nil {
		return nil, "", fmt.Errorf("execution: resolve weight file: %w", err)
	}

	store := NewResidentStore(memory)
	parsers := make([]types.Parser, 0, len(weightFiles))
	loadedPaths := make([]string, 0, len(weightFiles))

	for _, weightFile := range weightFiles {
		reader, file, err := hubClient.Open(ctx, location, weightFile, cacheDir)

		if err != nil {
			return nil, "", fmt.Errorf("execution: open weight file %q: %w", weightFile, err)
		}

		rawBytes, readErr := os.ReadFile(file.Path)
		reader.Close()

		if readErr != nil {
			return nil, "", fmt.Errorf("execution: read weight file %q: %w", file.Path, readErr)
		}

		bundle, loadErr := LoadSafetensorsArchive(rawBytes, memory)

		if loadErr != nil {
			return nil, "", loadErr
		}

		store.Absorb(bundle.Store)
		parsers = append(parsers, bundle.Parser)
		loadedPaths = append(loadedPaths, file.Path)
	}

	return &CheckpointBundle{
		Parser: NewMergedParser(parsers...),
		Store:  store,
	}, strings.Join(loadedPaths, ","), nil
}

/*
MergedParser concatenates token streams from multiple safetensors parsers.
Duplicate names keep the last occurrence.
*/
type MergedParser struct {
	parsers []types.Parser
}

/*
NewMergedParser constructs one parser view over multiple archives.
*/
func NewMergedParser(parsers ...types.Parser) types.Parser {
	filtered := make([]types.Parser, 0, len(parsers))

	for _, parser := range parsers {
		if parser != nil {
			filtered = append(filtered, parser)
		}
	}

	if len(filtered) == 0 {
		return nil
	}

	if len(filtered) == 1 {
		return filtered[0]
	}

	return &MergedParser{parsers: filtered}
}

func (mergedParser *MergedParser) Generate() iter.Seq[types.Token] {
	return func(yield func(types.Token) bool) {
		if mergedParser == nil {
			return
		}

		for _, parser := range mergedParser.parsers {
			for token := range parser.Generate() {
				if !yield(token) {
					return
				}
			}
		}
	}
}
