.PHONY: test bench check verify dump

# qpool uses go:linkname for ScheduleFast goroutine parking.
LDFLAGS := -ldflags='-checklinkname=0'

DUMP ?= puter.txt

# check runs mechanical enforcement of AGENTS.md banned patterns and
# ARCHITECTURE.md §7 invariants. Fail-fast: exits nonzero on any
# violation. Run before claiming a task complete.
check:
	@bash "$(CURDIR)/scripts/check_banned.sh"

test:
	go test $(LDFLAGS) ./...

bench:
	go test $(LDFLAGS) -bench=. ./...

# verify is the gate: banned-pattern check first, then tests. Use this
# before opening a PR or declaring done.
verify: check test

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"
