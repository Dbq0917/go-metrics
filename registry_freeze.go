package metrics

import "strings"

// FrozenRegistry is an immutable, read-only Registry holding point-in-time
// snapshots of the metrics it was created from. Every metric type is captured
// in a single pass, so all values belong to the same instant regardless of how
// long the caller spends reading them.
//
// The frozen metrics are the library's own *Snapshot types, which panic on any
// mutating call. The registry itself also rejects registration changes, so a
// report built from a FrozenRegistry can never drift or be written to.
type FrozenRegistry struct {
	metrics map[string]interface{}
	prefix  string
}

var _ Registry = &FrozenRegistry{}

// freezeMetric returns a read-only snapshot of the given metric. Numeric and
// distribution metrics all expose Snapshot(), so counters, gauges, histograms,
// meters and timers are unified onto the same point-in-time semantics here.
// Healthchecks have no snapshot type and are evaluated on demand, so they are
// passed through unchanged.
func freezeMetric(i interface{}) interface{} {
	switch metric := i.(type) {
	case Counter:
		return metric.Snapshot()
	case Gauge:
		return metric.Snapshot()
	case GaugeFloat64:
		return metric.Snapshot()
	case Healthcheck:
		return metric
	case Histogram:
		return metric.Snapshot()
	case Meter:
		return metric.Snapshot()
	case Timer:
		return metric.Snapshot()
	default:
		return i
	}
}

// scoped returns a FrozenRegistry sharing the same immutable metric map but
// exposing only the entries whose names start with prefix. Names keep their
// full, prefixed form, matching PrefixedRegistry.Each.
func (f *FrozenRegistry) scoped(prefix string) *FrozenRegistry {
	return &FrozenRegistry{metrics: f.metrics, prefix: prefix}
}

// Freeze returns the registry itself: it is already a point-in-time copy, so
// freezing it again is idempotent and yields the same frozen instant.
func (f *FrozenRegistry) Freeze() Registry {
	return f
}

// Each calls the given function for every frozen metric in scope. The
// iteration runs without holding any lock on the source registry because the
// snapshots live in their own map.
func (f *FrozenRegistry) Each(fn func(string, interface{})) {
	for name, iface := range f.metrics {
		if strings.HasPrefix(name, f.prefix) {
			fn(name, iface)
		}
	}
}

// Get returns the frozen metric stored under the given (un-prefixed) name or
// nil if none exists.
func (f *FrozenRegistry) Get(name string) interface{} {
	return f.metrics[f.prefix+name]
}

// GetOrRegister panics: a FrozenRegistry cannot be modified.
func (f *FrozenRegistry) GetOrRegister(name string, i interface{}) interface{} {
	panic("GetOrRegister called on a FrozenRegistry")
}

// Register panics: a FrozenRegistry cannot be modified.
func (f *FrozenRegistry) Register(name string, i interface{}) error {
	panic("Register called on a FrozenRegistry")
}

// RunHealthchecks runs every in-scope Healthcheck. Healthchecks are evaluated
// on demand and are not part of the numeric point-in-time snapshot.
func (f *FrozenRegistry) RunHealthchecks() {
	for _, i := range f.metrics {
		if h, ok := i.(Healthcheck); ok {
			h.Check()
		}
	}
}

// GetAll renders the frozen metrics into plain maps. Because every value comes
// from the same FrozenRegistry, totals and their breakdowns are guaranteed to
// describe the same instant.
func (f *FrozenRegistry) GetAll() map[string]map[string]interface{} {
	data := make(map[string]map[string]interface{})
	f.Each(func(name string, i interface{}) {
		values := make(map[string]interface{})
		switch metric := i.(type) {
		case Counter:
			values["count"] = metric.Count()
		case Gauge:
			values["value"] = metric.Value()
		case GaugeFloat64:
			values["value"] = metric.Value()
		case Healthcheck:
			values["error"] = nil
			metric.Check()
			if err := metric.Error(); nil != err {
				values["error"] = metric.Error().Error()
			}
		case Histogram:
			h := metric.Snapshot()
			ps := h.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			values["count"] = h.Count()
			values["min"] = h.Min()
			values["max"] = h.Max()
			values["mean"] = h.Mean()
			values["stddev"] = h.StdDev()
			values["median"] = ps[0]
			values["75%"] = ps[1]
			values["95%"] = ps[2]
			values["99%"] = ps[3]
			values["99.9%"] = ps[4]
		case Meter:
			m := metric.Snapshot()
			values["count"] = m.Count()
			values["1m.rate"] = m.Rate1()
			values["5m.rate"] = m.Rate5()
			values["15m.rate"] = m.Rate15()
			values["mean.rate"] = m.RateMean()
		case Timer:
			t := metric.Snapshot()
			ps := t.Percentiles([]float64{0.5, 0.75, 0.95, 0.99, 0.999})
			values["count"] = t.Count()
			values["min"] = t.Min()
			values["max"] = t.Max()
			values["mean"] = t.Mean()
			values["stddev"] = t.StdDev()
			values["median"] = ps[0]
			values["75%"] = ps[1]
			values["95%"] = ps[2]
			values["99%"] = ps[3]
			values["99.9%"] = ps[4]
			values["1m.rate"] = t.Rate1()
			values["5m.rate"] = t.Rate5()
			values["15m.rate"] = t.Rate15()
			values["mean.rate"] = t.RateMean()
		}
		data[name] = values
	})
	return data
}

// Unregister panics: a FrozenRegistry cannot be modified.
func (f *FrozenRegistry) Unregister(name string) {
	panic("Unregister called on a FrozenRegistry")
}

// UnregisterAll panics: a FrozenRegistry cannot be modified.
func (f *FrozenRegistry) UnregisterAll() {
	panic("UnregisterAll called on a FrozenRegistry")
}

// Freeze captures every registered metric in a single point-in-time copy.
//
// The registry's read lock is held only while walking the map and taking each
// metric's snapshot. Metrics are updated through their own atomics/locks and
// never through the registry lock, so concurrent writes are not blocked while
// snapshots are taken; callbacks run later, on the FrozenRegistry, with no lock
// held at all.
func (r *StandardRegistry) Freeze() Registry {
	r.mutex.RLock()
	frozen := make(map[string]interface{}, len(r.metrics))
	for name, i := range r.metrics {
		frozen[name] = freezeMetric(i)
	}
	r.mutex.RUnlock()
	return &FrozenRegistry{metrics: frozen}
}

// Freeze resolves the underlying standard registry and its accumulated prefix,
// freezes the whole underlying registry, and returns the frozen view scoped to
// this prefixed registry's own entries.
func (r *PrefixedRegistry) Freeze() Registry {
	base, prefix := findPrefix(r, "")
	return base.Freeze().(*FrozenRegistry).scoped(prefix)
}
