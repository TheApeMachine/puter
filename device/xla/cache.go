package xla

import (
	"sync"
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
ExecutableCache stores compiled XLA executables keyed by ProgramKey digest.
*/
type ExecutableCache struct {
	mutex       sync.RWMutex
	executables map[[32]byte]*CompiledExecutable
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
