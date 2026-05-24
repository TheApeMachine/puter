package xla

import (
	"sync"
	"sync/atomic"
)

/*
CompiledExecutable is an opaque handle to a compiled XLA program.
The xla build tag supplies a real PJRT-backed implementation.
*/
type CompiledExecutable struct {
	key    ProgramKey
	handle uintptr
}

/*
CacheMetrics reports compile-cache and execution counters for XLA builders.
*/
type CacheMetrics struct {
	Compiles int64
	Hits     int64
	Misses   int64
	Executes int64
}

/*
ExecutableCache stores compiled XLA executables keyed by ProgramKey digest.
*/
type ExecutableCache struct {
	mutex       sync.RWMutex
	executables map[[32]byte]*CompiledExecutable
	compiles    int64
	hits        int64
	misses      int64
	executes    int64
}

/*
NewExecutableCache constructs an empty compile cache.
*/
func NewExecutableCache() *ExecutableCache {
	return &ExecutableCache{
		executables: make(map[[32]byte]*CompiledExecutable),
	}
}

/*
Lookup returns a cached executable when the digest matches.
*/
func (executableCache *ExecutableCache) Lookup(programKey ProgramKey) (*CompiledExecutable, bool) {
	digest := programKey.Hash()
	executableCache.mutex.RLock()
	executable, ok := executableCache.executables[digest]
	executableCache.mutex.RUnlock()
	return executable, ok
}

/*
Store inserts a compiled executable for the given program key.
*/
func (executableCache *ExecutableCache) Store(
	programKey ProgramKey,
	executable *CompiledExecutable,
) {
	digest := programKey.Hash()
	executableCache.mutex.Lock()
	executableCache.executables[digest] = executable
	executableCache.mutex.Unlock()
}

/*
Len reports the number of cached executables.
*/
func (executableCache *ExecutableCache) Len() int {
	executableCache.mutex.RLock()
	count := len(executableCache.executables)
	executableCache.mutex.RUnlock()
	return count
}

/*
Metrics returns a snapshot of compile-cache counters.
*/
func (executableCache *ExecutableCache) Metrics() CacheMetrics {
	return CacheMetrics{
		Compiles: atomic.LoadInt64(&executableCache.compiles),
		Hits:     atomic.LoadInt64(&executableCache.hits),
		Misses:   atomic.LoadInt64(&executableCache.misses),
		Executes: atomic.LoadInt64(&executableCache.executes),
	}
}

func (executableCache *ExecutableCache) recordHit() {
	atomic.AddInt64(&executableCache.hits, 1)
}

func (executableCache *ExecutableCache) recordMiss() {
	atomic.AddInt64(&executableCache.misses, 1)
}

func (executableCache *ExecutableCache) recordCompile() {
	atomic.AddInt64(&executableCache.compiles, 1)
}

func (executableCache *ExecutableCache) recordExecute() {
	atomic.AddInt64(&executableCache.executes, 1)
}
