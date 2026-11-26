package metrics

import "time"

// ShardedRegistry is the lock-sharded bounded registry implementation.
type ShardedRegistry = BoundedRegistry

// CappedRegistry is a size-bounded registry alias.
type CappedRegistry = BoundedRegistry

// EvictingRegistry is an idle/capacity evicting registry alias.
type EvictingRegistry = BoundedRegistry

// Config is a short alias for BoundedRegistryConfig.
type Config = BoundedRegistryConfig

// RegistryConfig is an alias for BoundedRegistryConfig.
type RegistryConfig = BoundedRegistryConfig

// ErrEvicted is an alias for ErrMetricEvicted.
var ErrEvicted = ErrMetricEvicted

// ErrCounterEvicted is an alias for ErrMetricEvicted.
var ErrCounterEvicted = ErrMetricEvicted

// ErrEvictedMetric is an alias for ErrMetricEvicted.
var ErrEvictedMetric = ErrMetricEvicted

// ErrWriteAfterEviction is an alias for ErrMetricEvicted.
var ErrWriteAfterEviction = ErrMetricEvicted

// EvictedMetricError is an alias for MetricEvictedError.
type EvictedMetricError = MetricEvictedError

// RevivableCounter is the terminology-compatible name for a counter that can
// re-register itself after eviction.
type RevivableCounter interface {
	ResurrectableCounter
	Revive() error
}

// NewDefaultShardedRegistry returns an unbounded registry with the default
// number of lock shards.
func NewDefaultShardedRegistry() *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		ShardCount: defaultBoundedShardCount,
	})
}

// SetDefaultRegistry replaces the package-level default registry. Call this
// during initialization before metrics are recorded.
func SetDefaultRegistry(registry Registry) {
	if registry != nil {
		DefaultRegistry = registry
	}
}

// UseBoundedDefaultRegistry installs a bounded registry as DefaultRegistry.
func UseBoundedDefaultRegistry(maxMetrics int, idleTTL time.Duration) *BoundedRegistry {
	registry := NewRegistryWithLimit(maxMetrics, idleTTL)
	DefaultRegistry = registry
	return registry
}

// NewRegistryWithConfig is a short alias for NewBoundedRegistryWithConfig.
func NewRegistryWithConfig(config Config) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(config)
}

// NewResurrectableCounter registers and returns a counter handle that can
// re-register itself after eviction.
func NewResurrectableCounter(name string, registry *BoundedRegistry) ResurrectableCounter {
	if registry == nil {
		registry = UseBoundedDefaultRegistry(0, 0)
	}
	if err := registry.Register(name, NewCounter()); err != nil {
		return registry.Get(name).(ResurrectableCounter)
	}
	return registry.Get(name).(ResurrectableCounter)
}

// NewRegistryWithLimit constructs a bounded registry using the common
// (maximum cardinality, idle TTL) form.
func NewRegistryWithLimit(maxMetrics int, idleTTL time.Duration) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize: maxMetrics,
		IdleTTL: idleTTL,
	})
}

// NewRegistryWithMaxMetrics is an alias for NewRegistryWithLimit.
func NewRegistryWithMaxMetrics(maxMetrics int, idleTTL time.Duration) *BoundedRegistry {
	return NewRegistryWithLimit(maxMetrics, idleTTL)
}

// NewCappedShardedRegistry constructs a capped, idle-evicting sharded registry.
func NewCappedShardedRegistry(maxMetrics, shardCount int, idleTTL time.Duration) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		MaxSize:    maxMetrics,
		ShardCount: shardCount,
		IdleTTL:    idleTTL,
	})
}

// Touch records an update for name without mutating the metric value. It
// returns false when name is not registered.
func (r *BoundedRegistry) Touch(name string) bool {
	if r == nil {
		return false
	}
	shard := r.shardFor(name)
	shard.mu.Lock()
	defer shard.mu.Unlock()
	entry, ok := shard.items[name]
	if !ok {
		return false
	}
	now := r.now().UnixNano()
	entry.updateNano = now
	entry.accessNano = now
	return true
}

// MarkUpdated is an alias for Touch.
func (r *BoundedRegistry) MarkUpdated(name string) bool { return r.Touch(name) }

// LastUpdate returns the last recorded update time for name.
func (r *BoundedRegistry) LastUpdate(name string) (time.Time, bool) {
	if r == nil {
		return time.Time{}, false
	}
	shard := r.shardFor(name)
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	entry, ok := shard.items[name]
	if !ok {
		return time.Time{}, false
	}
	return time.Unix(0, entryUpdatedNano(entry)), true
}

// Evict removes name as though it were selected by capacity eviction. Stale
// handles can therefore resurrect it. It reports whether an entry was removed.
func (r *BoundedRegistry) Evict(name string) bool {
	if r == nil || r.Get(name) == nil {
		return false
	}
	r.removeName(name, EvictionCapacity, false)
	return true
}

// EvictMetric is an alias for Evict.
func (r *BoundedRegistry) EvictMetric(name string) bool { return r.Evict(name) }

// EvictOldest removes and returns the least recently updated entry name.
func (r *BoundedRegistry) EvictOldest() (string, bool) {
	if r == nil {
		return "", false
	}

	var victimShard *registryShard
	var victimName string
	var victimTime int64
	for _, shard := range r.shards {
		shard.mu.RLock()
		name, entry := leastRecentlyUpdated(shard.items)
		if entry != nil && (victimShard == nil || entryUpdatedNano(entry) < victimTime) {
			victimShard = shard
			victimName = name
			victimTime = entryUpdatedNano(entry)
		}
		shard.mu.RUnlock()
	}
	if victimShard == nil {
		return "", false
	}

	victimShard.mu.Lock()
	entry, ok := victimShard.items[victimName]
	if !ok {
		victimName, entry = leastRecentlyUpdated(victimShard.items)
	}
	var post []func()
	removed := false
	if entry != nil {
		post = r.removeEntryLocked(victimShard, victimName, entry, EvictionCapacity)
		removed = true
	}
	victimShard.mu.Unlock()
	runEvictionPostActions(post)
	return victimName, removed
}

// Revive re-registers an evicted counter.
func (c *StandardCounter) Revive() error { return c.Resurrect() }

// IsEvicted reports whether this handle has been removed by idle/capacity
// eviction and has not yet been resurrected.
func (c *StandardCounter) IsEvicted() bool { return c.Evicted() }

// Revive re-registers an evicted managed counter.
func (c *managedCounter) Revive() error { return c.Resurrect() }

// IsEvicted reports whether this managed handle has been evicted.
func (c *managedCounter) IsEvicted() bool { return c.Evicted() }
