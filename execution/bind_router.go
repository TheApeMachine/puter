package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
callRouter is the type-safe static router that maps a method name
string from an OperationBind to the corresponding typed call on
executionDevice. There is exactly one case per device method the
dispatcher invokes — adding a new device method means one case here,
NOT one case per op. Adding a new op that reuses an existing method
is purely a YAML change.

Each case casts the resolved args to the method's declared parameter
types, panic-recovers nothing (any type mismatch surfaces as a clear
runtime error pointing at the bind block that produced the wrong
shape), and invokes. The same pattern works for every device.Backend
family because their signatures are uniform: pointers, ints, dtype,
and an occasional config struct.

The router does not enforce that the bind block's argument count
matches the method's parameter count — that responsibility belongs to
the binder that builds OperationBind from YAML. Length mismatches
surface here as index-out-of-range diagnostics; they're caught by the
migration test before any op ships.
*/
func callRouter(deviceBackend executionDevice, bind OperationBind, configFields map[string]any, args []any) error {
	switch bind.Method {
	case "Lookup":
		// device.Embedding.Lookup(table, indices, output unsafe.Pointer,
		//                          vocab, hidden, indexCount int,
		//                          format dtype.DType)
		if len(args) != 7 {
			return fmt.Errorf("router: Lookup expects 7 args, got %d", len(args))
		}

		table, err := castPointer(args[0], "Lookup", "table")

		if err != nil {
			return err
		}

		indices, err := castPointer(args[1], "Lookup", "indices")

		if err != nil {
			return err
		}

		output, err := castPointer(args[2], "Lookup", "output")

		if err != nil {
			return err
		}

		vocab, err := castInt(args[3], "Lookup", "vocab")

		if err != nil {
			return err
		}

		hidden, err := castInt(args[4], "Lookup", "hidden")

		if err != nil {
			return err
		}

		indexCount, err := castInt(args[5], "Lookup", "indexCount")

		if err != nil {
			return err
		}

		format, err := castDType(args[6], "Lookup", "format")

		if err != nil {
			return err
		}

		deviceBackend.Lookup(table, indices, output, vocab, hidden, indexCount, format)

		return nil

	case "Add":
		// device.Elementwise.Add(dst, left, right unsafe.Pointer,
		//                         count int, format dtype.DType)
		if len(args) != 5 {
			return fmt.Errorf("router: Add expects 5 args, got %d", len(args))
		}

		dst, err := castPointer(args[0], "Add", "dst")

		if err != nil {
			return err
		}

		left, err := castPointer(args[1], "Add", "left")

		if err != nil {
			return err
		}

		right, err := castPointer(args[2], "Add", "right")

		if err != nil {
			return err
		}

		count, err := castInt(args[3], "Add", "count")

		if err != nil {
			return err
		}

		format, err := castDType(args[4], "Add", "format")

		if err != nil {
			return err
		}

		deviceBackend.Add(dst, left, right, count, format)

		return nil

	// Add a case per device.Backend method as ops migrate. The shape
	// stays the same: cast each arg with a name (for diagnostics), then
	// call. No reflection, no generics.

	default:
		return fmt.Errorf("router: unknown method %q (register it in callRouter)", bind.Method)
	}
}

/*
castPointer asserts an unsafe.Pointer-typed argument from the resolver.
Returns a wrapped error naming the method and parameter so a bind-block
typo lands as a clear diagnostic instead of a panic.
*/
func castPointer(value any, method, parameter string) (unsafe.Pointer, error) {
	pointer, ok := value.(unsafe.Pointer)

	if !ok {
		return nil, fmt.Errorf("router %s: arg %q is %T, expected unsafe.Pointer", method, parameter, value)
	}

	return pointer, nil
}

/*
castInt asserts an int-typed argument. Accepts plain int as the
resolver only ever returns int from dim / count sources.
*/
func castInt(value any, method, parameter string) (int, error) {
	asInt, ok := value.(int)

	if !ok {
		return 0, fmt.Errorf("router %s: arg %q is %T, expected int", method, parameter, value)
	}

	return asInt, nil
}

/*
castDType asserts a dtype.DType-typed argument.
*/
func castDType(value any, method, parameter string) (dtype.DType, error) {
	asDType, ok := value.(dtype.DType)

	if !ok {
		return dtype.Invalid, fmt.Errorf("router %s: arg %q is %T, expected dtype.DType", method, parameter, value)
	}

	return asDType, nil
}

// castFloat32 / castBool / castConfigStruct land alongside their first
// router-case consumer when the next op migrates. They aren't unused
// scaffolding right now; they exist as conventions so the next agent
// adding (say) MultiHeadAttention has a clear pattern to follow.

// Compile-time assertion: device.RoPEConfig and
// device.MultiHeadAttentionConfig are referenced here so the import
// stays anchored to the package the router actually consumes from
// when those methods land. Removing this when the second method case
// is added is fine.
var (
	_ device.RoPEConfig
	_ device.MultiHeadAttentionConfig
)
