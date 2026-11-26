package metrics

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type agentBoundedClock struct {
	mu sync.RWMutex
	t  time.Time
}

func newAgentBoundedClock() *agentBoundedClock {
	return &agentBoundedClock{t: time.Unix(1000, 0)}
}

func (c *agentBoundedClock) now() time.Time {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.t
}

func (c *agentBoundedClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func newAgentBoundedRegistry(max int, ttl time.Duration, clock *agentBoundedClock) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:        max,
		IdleTTL:        ttl,
		ShardCount:     1,
		Clock:          clock.now,
		DisableJanitor: true,
	})
}

func TestAgentBoundedRegistryCapsAndEvictsLeastRecentlyUpdated(t *testing.T) {
	clock := newAgentBoundedClock()
	r := newAgentBoundedRegistry(2, 0, clock)

	a := GetOrRegisterCounter("a", r)
	a.Inc(1)
	clock.advance(time.Millisecond)
	b := GetOrRegisterCounter("b", r)
	b.Inc(2)
	clock.advance(time.Millisecond)
	c := GetOrRegisterCounter("c", r)
	c.Inc(3)

	if got := r.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2", got)
	}
	if r.Get("a") != nil {
		t.Fatal("oldest metric a was not evicted")
	}
	if r.Get("b") == nil || r.Get("c") == nil {
		t.Fatal("newer metrics were evicted")
	}
}

func TestAgentBoundedCounterResurrectsWithItsValue(t *testing.T) {
	clock := newAgentBoundedClock()
	r := newAgentBoundedRegistry(1, 0, clock)

	a := NewRegisteredCounter("a", r)
	a.Inc(41)

	b := GetOrRegisterCounter("b", r)
	b.Inc(1)

	if r.Get("a") != nil {
		t.Fatal("a should have been evicted by b")
	}

	a.Inc(1)
	if got := a.Count(); got != 42 {
		t.Fatalf("evicted handle count = %d, want 42", got)
	}
	registeredA := GetOrRegisterCounter("a", r)
	if got := registeredA.Count(); got != 42 {
		t.Fatalf("resurrected registry count = %d, want 42", got)
	}
	if r.Len() != 1 {
		t.Fatalf("Len() = %d, want hard cap 1", r.Len())
	}
}

func TestAgentBoundedCounterResurrectionConflictFailsExplicitly(t *testing.T) {
	clock := newAgentBoundedClock()
	r := newAgentBoundedRegistry(1, 0, clock)

	old := GetOrRegisterCounter("name", r).(*StandardCounter)
	old.Inc(1)
	other := GetOrRegisterCounter("other", r)
	other.Inc(1)

	replacement := NewCounter()
	if err := r.Register("name", replacement); err != nil {
		t.Fatal(err)
	}

	err := old.IncE(1)
	if !errors.Is(err, ErrMetricNameInUse) {
		t.Fatalf("IncE error = %v, want ErrMetricNameInUse", err)
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("Inc on a conflicting stale handle did not panic")
			}
		}()
		old.Inc(1)
	}()
}

func TestAgentStrictBoundedCounterWritesFailAfterEviction(t *testing.T) {
	clock := newAgentBoundedClock()
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:             1,
		ShardCount:          1,
		Clock:               clock.now,
		DisableJanitor:      true,
		StrictEvictedWrites: true,
	})

	counter := GetOrRegisterCounter("a", r).(*StandardCounter)
	if _, ok := r.GetOrRegister("b", NewCounter()).(Counter); !ok {
		t.Fatal("b is not a Counter")
	}

	if !counter.Evicted() {
		t.Fatal("counter was not marked evicted")
	}
	if err := counter.IncE(1); !errors.Is(err, ErrMetricEvicted) {
		t.Fatalf("IncE error = %v, want ErrMetricEvicted", err)
	}
	if got := counter.Count(); got != 0 {
		t.Fatalf("failed write changed count to %d", got)
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("strict Inc did not panic")
			}
		}()
		counter.Inc(1)
	}()
}

func TestAgentBoundedRegistryIdleEvictionUsesUpdates(t *testing.T) {
	clock := newAgentBoundedClock()
	r := newAgentBoundedRegistry(0, time.Minute, clock)

	GetOrRegisterCounter("a", r)
	b := GetOrRegisterCounter("b", r)

	clock.advance(30 * time.Second)
	b.Inc(1)

	clock.advance(40 * time.Second)
	if removed := r.EvictIdle(); removed != 1 {
		t.Fatalf("EvictIdle() = %d, want 1", removed)
	}
	if r.Get("a") != nil {
		t.Fatal("idle metric a was not evicted")
	}
	if r.Get("b") == nil {
		t.Fatal("recently updated metric b was evicted")
	}

	clock.advance(31 * time.Second)
	if removed := r.EvictIdle(); removed != 1 {
		t.Fatalf("second EvictIdle() = %d, want 1", removed)
	}
}

func TestAgentBoundedRegistryExactShardCapacity(t *testing.T) {
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:    37,
		ShardCount: 16,
	})
	if got := r.ShardCount(); got != 16 {
		t.Fatalf("ShardCount() = %d, want 16", got)
	}
	for i := 0; i < 200; i++ {
		name := fmt.Sprintf("metric-%03d", i)
		if err := r.Register(name, NewCounter()); err != nil {
			t.Fatal(err)
		}
		if got := r.Len(); got > 37 {
			t.Fatalf("Len() = %d exceeded cap after %s", got, name)
		}
	}
	if got := r.Len(); got != 37 {
		t.Fatalf("Len() = %d, want exactly 37", got)
	}
}

func TestAgentBoundedRegistryKeepsExactSmallCapacity(t *testing.T) {
	for max := 1; max <= 40; max++ {
		r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{MaxSize: max})
		for i := 0; i < max; i++ {
			name := fmt.Sprintf("name-%d", i)
			if err := r.Register(name, NewCounter()); err != nil {
				t.Fatalf("max=%d name=%s: %v", max, name, err)
			}
		}
		if got := r.Len(); got != max {
			t.Fatalf("max=%d Len() = %d", max, got)
		}

		for i := max; i < max*3; i++ {
			name := fmt.Sprintf("name-%d", i)
			if _, ok := r.GetOrRegister(name, NewCounter()).(Counter); !ok {
				t.Fatalf("max=%d GetOrRegister(%s) is not a Counter", max, name)
			}
			if got := r.Len(); got != max {
				t.Fatalf("after %s Len() = %d, want %d", name, got, max)
			}
		}
	}
}

func TestAgentBoundedCounterResurrectsAcrossShards(t *testing.T) {
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:    1,
		ShardCount: 16,
	})
	counter := GetOrRegisterCounter("first", r)
	for i := 0; i < 50; i++ {
		GetOrRegisterCounter(fmt.Sprintf("other-%d", i), r).Inc(1)
		counter.Inc(1)
		if got := r.Len(); got != 1 {
			t.Fatalf("iteration %d Len() = %d, want 1", i, got)
		}
		if got := GetOrRegisterCounter("first", r).Count(); got != int64(i+1) {
			t.Fatalf("iteration %d registry count = %d, want %d", i, got, i+1)
		}
	}
}

func TestAgentBoundedRegistryConcurrentGetOrRegisterAndWrites(t *testing.T) {
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:    64,
		ShardCount: 8,
	})
	var wg sync.WaitGroup
	for goroutine := 0; goroutine < 16; goroutine++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				name := fmt.Sprintf("metric-%d-%d", g, i%97)
				c := GetOrRegisterCounter(name, r)
				c.Inc(1)
				if i%37 == 0 {
					r.Each(func(string, interface{}) {})
				}
			}
		}(goroutine)
	}
	wg.Wait()
	if got := r.Len(); got > 64 {
		t.Fatalf("Len() = %d exceeds cap", got)
	}
}

func TestAgentBoundedRegistryConcurrentUniqueNamesHardCap(t *testing.T) {
	const max = 32
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{MaxSize: max})
	var wg sync.WaitGroup
	for goroutine := 0; goroutine < 32; goroutine++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				name := fmt.Sprintf("unique-%d-%d", g, i)
				c := r.GetOrRegister(name, NewCounter()).(Counter)
				c.Inc(1)
			}
		}(goroutine)
	}
	wg.Wait()
	if got := r.Len(); got != max {
		t.Fatalf("Len() = %d, want hard cap %d", got, max)
	}
}

func TestAgentBoundedRegistryPrefixed(t *testing.T) {
	base := NewBoundedRegistry(2, time.Minute)
	r := NewPrefixedChildRegistry(base, "prefix.")
	c := GetOrRegisterCounter("foo", r)
	c.Inc(2)

	if got := GetOrRegisterCounter("foo", r).Count(); got != 2 {
		t.Fatalf("prefixed counter count = %d, want 2", got)
	}
	names := 0
	r.Each(func(name string, metric interface{}) {
		names++
		if name != "prefix.foo" {
			t.Fatalf("name = %q", name)
		}
		if _, ok := metric.(Counter); !ok {
			t.Fatalf("metric type %T is not Counter", metric)
		}
	})
	if names != 1 {
		t.Fatalf("Each visited %d metrics, want 1", names)
	}
}

func TestAgentBoundedGetOrRegisterLazyConstructorOnce(t *testing.T) {
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{ShardCount: 1})
	var calls int32
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = r.GetOrRegister("lazy", func() Counter {
				atomic.AddInt32(&calls, 1)
				return NewCounter()
			}).(Counter)
		}()
	}
	wg.Wait()
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("lazy constructor calls = %d, want 1", got)
	}
}

func BenchmarkAgentBoundedCounterParallel(b *testing.B) {
	r := NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:    4096,
		ShardCount: 32,
	})
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			c := GetOrRegisterCounter(fmt.Sprintf("benchmark-%d", i%1024), r)
			c.Inc(1)
			i++
		}
	})
}
