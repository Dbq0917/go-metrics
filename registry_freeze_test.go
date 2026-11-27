package metrics

import (
	"sync"
	"testing"
)

func TestFreezeCapturesAllMetricTypes(t *testing.T) {
	r := NewRegistry()
	r.Register("counter", NewCounter())
	r.Register("gauge", NewGauge())
	r.Register("gauge-float", NewGaugeFloat64())
	r.Register("histogram", NewHistogram(NewUniformSample(1)))
	r.Register("meter", NewMeter())
	r.Register("timer", NewTimer())

	frozen := r.Freeze()

	names := map[string]bool{}
	frozen.Each(func(name string, iface interface{}) {
		names[name] = true
	})
	for _, name := range []string{"counter", "gauge", "gauge-float", "histogram", "meter", "timer"} {
		if !names[name] {
			t.Fatalf("frozen registry missing metric %q", name)
		}
	}
}

func TestFreezeIsPointInTime(t *testing.T) {
	r := NewRegistry()
	counter := NewCounter()
	gauge := NewGauge()
	r.Register("counter", counter)
	r.Register("gauge", gauge)

	counter.Inc(100)
	gauge.Update(7)

	frozen := r.Freeze()
	frozenCounter := frozen.Get("counter").(Counter)
	frozenGauge := frozen.Get("gauge").(Gauge)

	// Mutate the live metrics after the freeze; the frozen copy must not move.
	counter.Inc(1000)
	gauge.Update(99)

	if got := frozenCounter.Count(); got != 100 {
		t.Fatalf("frozen counter: expected 100, got %d", got)
	}
	if got := frozenGauge.Value(); got != 7 {
		t.Fatalf("frozen gauge: expected 7, got %d", got)
	}
	if got := counter.Count(); got != 1100 {
		t.Fatalf("live counter: expected 1100, got %d", got)
	}
}

func TestFreezeSnapshotTypesAreReadOnly(t *testing.T) {
	r := NewRegistry()
	r.Register("counter", NewCounter())
	frozen := r.Freeze()

	if _, ok := frozen.Get("counter").(CounterSnapshot); !ok {
		t.Fatalf("expected CounterSnapshot, got %T", frozen.Get("counter"))
	}

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when writing a frozen counter")
		}
	}()
	frozen.Get("counter").(Counter).Inc(1)
}

func TestFrozenRegistryRejectsMutation(t *testing.T) {
	frozen := NewRegistry().Freeze()

	mutators := []func(){
		func() { frozen.Register("x", NewCounter()) },
		func() { frozen.GetOrRegister("x", NewCounter()) },
		func() { frozen.Unregister("x") },
		func() { frozen.UnregisterAll() },
	}
	for _, mutate := range mutators {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic when mutating a FrozenRegistry")
				}
			}()
			mutate()
		}()
	}
}

func TestFreezeIsIdempotent(t *testing.T) {
	r := NewRegistry()
	counter := NewCounter()
	r.Register("counter", counter)
	counter.Inc(5)

	first := r.Freeze()
	again := first.Freeze()
	if again != first {
		t.Fatal("Freeze on a FrozenRegistry must return the same instance")
	}
	counter.Inc(5)
	if got := again.Get("counter").(Counter).Count(); got != 5 {
		t.Fatalf("re-frozen counter: expected 5, got %d", got)
	}
}

func TestFreezeDoesNotBlockWriters(t *testing.T) {
	r := NewRegistry()
	counter := NewCounter()
	r.Register("counter", counter)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				counter.Inc(1)
			}
		}
	}()

	for i := 0; i < 100; i++ {
		frozen := r.Freeze()
		// Read back via Each without holding any registry lock.
		frozen.Each(func(name string, iface interface{}) {
			if _, ok := iface.(Counter); !ok {
				t.Fatalf("expected Counter, got %T", iface)
			}
			_ = iface.(Counter).Count()
		})
	}

	close(stop)
	wg.Wait()
}

func TestFrozenGetAllIsConsistent(t *testing.T) {
	r := NewRegistry()
	counter := NewCounter()
	r.Register("counter", counter)
	counter.Inc(42)

	data := r.GetAll()
	if got := data["counter"]["count"]; got != int64(42) {
		t.Fatalf("GetAll counter: expected 42, got %v", got)
	}
}

func TestPrefixedRegistryFreezeScopesToOwnEntries(t *testing.T) {
	base := NewRegistry()
	child := NewPrefixedChildRegistry(base, "prefix.")
	child.Register("foo", NewCounter())
	base.Register("bar", NewCounter())

	frozen := child.Freeze()

	names := map[string]bool{}
	frozen.Each(func(name string, _ interface{}) {
		names[name] = true
	})
	if !names["prefix.foo"] {
		t.Fatalf("expected to see own prefixed metric, got %v", names)
	}
	if names["bar"] {
		t.Fatalf("frozen prefixed registry must not expose sibling metric, got %v", names)
	}

	if frozen.Get("foo") == nil {
		t.Fatal("expected to resolve own metric by un-prefixed name")
	}
	if frozen.Get("bar") != nil {
		t.Fatal("must not resolve a sibling metric by name")
	}
}

func TestPrefixedRegistryGetAllOnlyReturnsOwnEntries(t *testing.T) {
	base := NewRegistry()
	child := NewPrefixedChildRegistry(base, "prefix.")
	child.Register("foo", NewCounter())
	base.Register("bar", NewCounter())

	data := child.GetAll()
	if _, ok := data["prefix.foo"]; !ok {
		t.Fatalf("expected own prefixed metric in GetAll, got %v", data)
	}
	if _, ok := data["bar"]; ok {
		t.Fatalf("GetAll must not include sibling metric, got %v", data)
	}
}

func TestNestedPrefixedRegistryFreeze(t *testing.T) {
	root := NewRegistry()
	outer := NewPrefixedChildRegistry(root, "a.")
	inner := NewPrefixedChildRegistry(outer, "b.")
	inner.Register("foo", NewCounter())
	root.Register("outside", NewCounter())

	frozen := inner.Freeze()

	names := map[string]bool{}
	frozen.Each(func(name string, _ interface{}) {
		names[name] = true
	})
	if !names["a.b.foo"] {
		t.Fatalf("expected nested prefixed metric, got %v", names)
	}
	if names["outside"] {
		t.Fatalf("nested freeze must not expose sibling metric, got %v", names)
	}
	if frozen.Get("foo") == nil {
		t.Fatal("expected to resolve nested metric by its un-prefixed name")
	}
}
