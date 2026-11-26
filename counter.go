package metrics

import "sync/atomic"

// Counters hold an int64 value that can be incremented and decremented.
type Counter interface {
	Clear()
	Count() int64
	Dec(int64)
	Inc(int64)
	Snapshot() Counter
}

// WritableCounter is a Counter whose mutation can report a registry
// eviction or resurrection conflict instead of silently losing the write.
type WritableCounter interface {
	Counter
	Add(int64) error
}

// GetOrRegisterCounter returns an existing Counter or constructs and registers
// a new StandardCounter.
func GetOrRegisterCounter(name string, r Registry) Counter {
	if nil == r {
		r = DefaultRegistry
	}
	return r.GetOrRegister(name, NewCounter).(Counter)
}

// NewCounter constructs a new StandardCounter.
func NewCounter() Counter {
	if UseNilMetrics {
		return NilCounter{}
	}
	return &StandardCounter{}
}

// NewRegisteredCounter constructs and registers a new StandardCounter.
func NewRegisteredCounter(name string, r Registry) Counter {
	c := NewCounter()
	if nil == r {
		r = DefaultRegistry
	}
	r.Register(name, c)
	return c
}

// CounterSnapshot is a read-only copy of another Counter.
type CounterSnapshot int64

// Clear panics.
func (CounterSnapshot) Clear() {
	panic("Clear called on a CounterSnapshot")
}

// Count returns the count at the time the snapshot was taken.
func (c CounterSnapshot) Count() int64 { return int64(c) }

// Dec panics.
func (CounterSnapshot) Dec(int64) {
	panic("Dec called on a CounterSnapshot")
}

// Inc panics.
func (CounterSnapshot) Inc(int64) {
	panic("Inc called on a CounterSnapshot")
}

// Snapshot returns the snapshot.
func (c CounterSnapshot) Snapshot() Counter { return c }

// NilCounter is a no-op Counter.
type NilCounter struct{}

// Clear is a no-op.
func (NilCounter) Clear() {}

// Count is a no-op.
func (NilCounter) Count() int64 { return 0 }

// Dec is a no-op.
func (NilCounter) Dec(i int64) {}

// Inc is a no-op.
func (NilCounter) Inc(i int64) {}

// Snapshot is a no-op.
func (NilCounter) Snapshot() Counter { return NilCounter{} }

// StandardCounter is the standard implementation of a Counter and uses the
// sync/atomic package to manage a single int64 value.
type StandardCounter struct {
	count int64
	hook  atomic.Value // counterWriteHook
}

type counterWriteHook interface {
	addCounter(*StandardCounter, int64) error
	clearCounter(*StandardCounter) error
}

func (c *StandardCounter) writeHook() counterWriteHook {
	if v := c.hook.Load(); v != nil {
		return v.(counterWriteHook)
	}
	return nil
}

func (c *StandardCounter) setWriteHook(h counterWriteHook) {
	if h == nil {
		h = disabledCounterHook{}
	}
	c.hook.Store(h)
}

// Add adds n to the counter. When the counter is managed by a bounded
// registry, Add returns an error if the write cannot be re-attached to the
// registry.
func (c *StandardCounter) Add(n int64) error {
	if h := c.writeHook(); h != nil {
		return h.addCounter(c, n)
	}
	atomic.AddInt64(&c.count, n)
	return nil
}

// Clear sets the counter to zero.
func (c *StandardCounter) Clear() {
	if h := c.writeHook(); h != nil {
		if err := h.clearCounter(c); err != nil {
			panic(err)
		}
		return
	}
	atomic.StoreInt64(&c.count, 0)
}

// Count returns the current count.
func (c *StandardCounter) Count() int64 {
	return atomic.LoadInt64(&c.count)
}

// Dec decrements the counter by the given amount.
func (c *StandardCounter) Dec(i int64) {
	if err := c.Add(-i); err != nil {
		panic(err)
	}
}

// Inc increments the counter by the given amount.
func (c *StandardCounter) Inc(i int64) {
	if err := c.Add(i); err != nil {
		panic(err)
	}
}

// Snapshot returns a read-only copy of the counter.
func (c *StandardCounter) Snapshot() Counter {
	return CounterSnapshot(c.Count())
}

type disabledCounterHook struct{}

func (disabledCounterHook) addCounter(c *StandardCounter, n int64) error {
	atomic.AddInt64(&c.count, n)
	return nil
}

func (disabledCounterHook) clearCounter(c *StandardCounter) error {
	atomic.StoreInt64(&c.count, 0)
	return nil
}
