package xla

import (
	"sync"
	"unsafe"
)

const workspaceSlotShift = 6

/*
Workspace maps virtual byte offsets to pre-resolved resident tensor pointers.
*/
type Workspace struct {
	mutex            sync.RWMutex
	slots            []unsafe.Pointer
	resolvedByOffset map[int64]int
}

/*
NewWorkspace constructs an empty XLA workspace slot table.
*/
func NewWorkspace() *Workspace {
	return &Workspace{
		resolvedByOffset: make(map[int64]int),
	}
}

/*
BindSlot registers a resident tensor pointer for a virtual workspace offset.
*/
func (workspace *Workspace) BindSlot(virtualOffset int64, residentPointer unsafe.Pointer) {
	slotIndex := int(virtualOffset >> workspaceSlotShift)
	workspace.mutex.Lock()
	defer workspace.mutex.Unlock()

	if slotIndex >= len(workspace.slots) {
		nextSlots := make([]unsafe.Pointer, slotIndex+1)
		copy(nextSlots, workspace.slots)
		workspace.slots = nextSlots
	}

	workspace.slots[slotIndex] = residentPointer
	workspace.resolvedByOffset[virtualOffset] = slotIndex
}

/*
Resolve returns the resident pointer bound to a virtual offset.
*/
func (workspace *Workspace) Resolve(virtualOffset int64) unsafe.Pointer {
	slotIndex := int(virtualOffset >> workspaceSlotShift)
	workspace.mutex.RLock()
	defer workspace.mutex.RUnlock()

	if slotIndex >= len(workspace.slots) {
		return nil
	}

	return workspace.slots[slotIndex]
}

/*
SlotCount reports the number of allocated workspace slots.
*/
func (workspace *Workspace) SlotCount() int {
	workspace.mutex.RLock()
	defer workspace.mutex.RUnlock()
	return len(workspace.slots)
}

/*
Close clears workspace slot references. PJRT buffer release happens in xla builds.
*/
func (workspace *Workspace) Close() {
	workspace.mutex.Lock()
	defer workspace.mutex.Unlock()
	workspace.slots = nil
	workspace.resolvedByOffset = make(map[int64]int)
}
