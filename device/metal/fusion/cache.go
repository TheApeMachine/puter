package fusion

import (
	"fmt"
	"sync"
)

/*
Cache stores MTLLibrary-compiled fusion programs keyed by kernel name and
source text. One cache is shared per Metal backend session.
*/
type Cache struct {
	mu       sync.Mutex
	programs map[string]*Program
}

/*
NewCache constructs an empty fusion program cache.
*/
func NewCache() *Cache {
	return &Cache{
		programs: make(map[string]*Program),
	}
}

/*
Program returns a compiled fusion program for one MSL source and kernel name.
*/
func (cache *Cache) Program(source, kernelName string) (*Program, error) {
	if source == "" {
		return nil, fmt.Errorf("metal fusion: source is required")
	}

	if kernelName == "" {
		return nil, fmt.Errorf("metal fusion: kernel name is required")
	}

	cacheKey := kernelName + "\x00" + source

	cache.mu.Lock()
	defer cache.mu.Unlock()

	existing, ok := cache.programs[cacheKey]

	if ok {
		return existing, nil
	}

	program := &Program{
		source:     source,
		kernelName: kernelName,
	}

	cache.programs[cacheKey] = program

	return program, nil
}

/*
Close releases every cached fusion program.
*/
func (cache *Cache) Close() {
	if cache == nil {
		return
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	for cacheKey, program := range cache.programs {
		program.close()
		delete(cache.programs, cacheKey)
	}
}
