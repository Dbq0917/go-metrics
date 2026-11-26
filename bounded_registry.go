package metrics

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"
)

const defaultBoundedShardCount = 16

var (
	// ErrMetricEvicted is returned by strict metric handles after the
	// registry has evicted their entry.
	ErrMetricEvicted = errors.New("metrics: metric evicted from bounded registry")

	// ErrMetricNameInUse reports that a stale handle tried to resurrect under
	// a name currently owned by a different metric.
	ErrMetricNameInUse = errors.New("metrics: metric name is in use by another handle")

	// ErrRegistryClosed is returned by writes to a closed bounded registry.
	ErrRegistryClosed = errors.New("metrics: registry is closed")

	// ErrRegistryFull is returned when a full registry is configured to reject
	// new metrics instead of evicting the least recently updated entry.
	ErrRegistryFull = errors.New("metrics: registry capacity exhausted")
)

// EvictionReason identifies why a bounded registry removed an entry.
type EvictionReason int

const (
	// EvictionCapacity means an entry was removed to satisfy MaxSize.
	EvictionCapacity EvictionReason = iota

	// EvictionIdle means an entry had not been updated for longer than IdleTTL.
	EvictionIdle

	// EvictionUnregistered means the caller explicitly removed the entry.
	EvictionUnregistered

	// EvictionCleared means UnregisterAll removed the entry.
	EvictionCleared
)

// EvictionFunc is called after an entry has been removed. The callback runs
// without registry shard locks held and may use the registry.
type EvictionFunc func(name string, metric interface{}, reason EvictionReason)

// BoundedRegistryConfig configures a BoundedRegistry.
type BoundedRegistryConfig struct {
	// MaxSize is the hard maximum number of entries. Zero means unlimited.
	MaxSize int

	// Max and Limit are aliases for MaxSize.
	Max   int
	Limit int

	// MaxMetrics is an alias for MaxSize.
	MaxMetrics int

	// Capacity is an alias for MaxSize.
	Capacity int

	// MaxEntries is an alias for MaxSize.
	MaxEntries int

	// IdleTTL evicts entries whose last update/use is older than this value.
	// Zero disables age-based eviction.
	IdleTTL time.Duration

	// TTL, MaxIdle and IdleTimeout are aliases for IdleTTL.
	TTL         time.Duration
	MaxIdle     time.Duration
	IdleTimeout time.Duration

	// IdleDuration and ExpireAfter are aliases for IdleTTL.
	IdleDuration time.Duration
	ExpireAfter  time.Duration

	// ShardCount is the number of independent locks/shards. Values are rounded
	// down to a power of two when necessary to preserve an exact MaxSize.
	ShardCount int

	// Shards is an alias for ShardCount.
	Shards int

	// Clock overrides time.Now, primarily for tests.
	Clock func() time.Time

	// Now is an alias for Clock.
	Now func() time.Time

	// OnEvict is called after capacity, idle, explicit and clear removals.
	OnEvict EvictionFunc

	// OnEviction is an alias for OnEvict.
	OnEviction EvictionFunc

	// OnEvicted is an alias for OnEvict.
	OnEvicted EvictionFunc

	// EvictionCallback is an alias for OnEvict.
	EvictionCallback EvictionFunc

	// ReapInterval starts a background idle sweep when positive. When IdleTTL
	// is positive and ReapInterval is zero, a conservative janitor is started.
	ReapInterval time.Duration

	// DisableJanitor turns off the automatic background idle sweep.
	DisableJanitor bool

	// StrictEvictedWrites makes evicted handles return ErrMetricEvicted
	// instead of resurrecting themselves on the next write.
	StrictEvictedWrites bool

	// RejectWhenFull returns ErrRegistryFull after idle eviction cannot make
	// room. The default is to evict the least recently updated entry.
	RejectWhenFull bool
}

// BoundedRegistry is a lock-sharded Registry with an optional hard size cap
// and idle-time eviction.
type BoundedRegistry struct {
	shards      []*registryShard
	maxSize     int
	idleTTL     time.Duration
	now         func() time.Time
	onEvict     EvictionFunc
	strict      bool
	rejectFull  bool
	size        int64
	evictions   int64
	closed      int32
	stopJanitor chan struct{}
	janitorDone chan struct{}
	closeOnce   sync.Once
}

type registryShard struct {
	mu    sync.RWMutex
	items map[string]*boundedEntry
}

type boundedEntry struct {
	metric     interface{}
	state      *int32
	updateNano int64
	accessNano int64
}

const (
	handleActive int32 = iota
	handleEvicted
	handleManual
)

// BoundedOption configures a BoundedRegistry.
type BoundedOption func(*BoundedRegistryConfig)

// Option is an alias for BoundedOption.
type Option = BoundedOption

// WithMaxSize sets the hard entry limit.
func WithMaxSize(n int) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.MaxSize = n }
}

// WithMaxMetrics sets the hard entry limit.
func WithMaxMetrics(n int) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.MaxMetrics = n }
}

// WithCapacity sets the hard entry limit.
func WithCapacity(n int) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.Capacity = n }
}

// WithIdleTTL sets the maximum time without an update/use.
func WithIdleTTL(d time.Duration) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.IdleTTL = d }
}

// WithTTL sets the maximum time without an update/use.
func WithTTL(d time.Duration) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.TTL = d }
}

// WithShardCount sets the number of lock shards.
func WithShardCount(n int) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.ShardCount = n }
}

// WithShards sets the number of lock shards.
func WithShards(n int) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.Shards = n }
}

// WithClock overrides the registry clock.
func WithClock(now func() time.Time) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.Clock = now }
}

// WithEvictionCallback installs an eviction callback.
func WithEvictionCallback(fn EvictionFunc) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.OnEvict = fn }
}

// WithStrictEvictedWrites causes writes to evicted handles to fail.
func WithStrictEvictedWrites() BoundedOption {
	return func(c *BoundedRegistryConfig) { c.StrictEvictedWrites = true }
}

// WithRejectWhenFull causes registration to fail instead of LRU eviction.
func WithRejectWhenFull() BoundedOption {
	return func(c *BoundedRegistryConfig) { c.RejectWhenFull = true }
}

// WithReapInterval sets the background sweep interval.
func WithReapInterval(d time.Duration) BoundedOption {
	return func(c *BoundedRegistryConfig) { c.ReapInterval = d }
}

// NewBoundedRegistryWithConfig constructs a bounded registry from config.
func NewBoundedRegistryWithConfig(cfg BoundedRegistryConfig) *BoundedRegistry {
	maxSize := firstPositiveInt(cfg.MaxSize, cfg.Max, cfg.Limit, cfg.MaxMetrics, cfg.Capacity, cfg.MaxEntries)
	idleTTL := firstPositiveDuration(cfg.IdleTTL, cfg.TTL, cfg.MaxIdle, cfg.IdleTimeout, cfg.IdleDuration, cfg.ExpireAfter)
	shardCount := firstPositiveInt(cfg.ShardCount, cfg.Shards)
	if shardCount <= 0 {
		shardCount = defaultBoundedShardCount
	}
	shardCount = normalizeShardCount(shardCount, maxSize)

	now := cfg.Clock
	if now == nil {
		now = cfg.Now
	}
	if now == nil {
		now = time.Now
	}
	onEvict := cfg.OnEvict
	if onEvict == nil {
		onEvict = cfg.OnEviction
	}
	if onEvict == nil {
		onEvict = cfg.OnEvicted
	}
	if onEvict == nil {
		onEvict = cfg.EvictionCallback
	}

	r := &BoundedRegistry{
		shards:     make([]*registryShard, shardCount),
		maxSize:    maxSize,
		idleTTL:    idleTTL,
		now:        now,
		onEvict:    onEvict,
		strict:     cfg.StrictEvictedWrites,
		rejectFull: cfg.RejectWhenFull,
	}

	for i := range r.shards {
		r.shards[i] = &registryShard{
			items: make(map[string]*boundedEntry),
		}
	}

	clockInjected := cfg.Clock != nil || cfg.Now != nil
	if !cfg.DisableJanitor && idleTTL > 0 && (cfg.ReapInterval > 0 || !clockInjected) {
		interval := cfg.ReapInterval
		if interval <= 0 {
			interval = idleTTL / 2
			if interval <= 0 {
				interval = time.Nanosecond
			}
			if interval > time.Second {
				interval = time.Second
			}
		}
		r.StartJanitor(interval)
	}
	return r
}

// NewBoundedRegistry constructs a capped, optionally idle-evicting registry.
//
// The standard form is NewBoundedRegistry(maxSize, idleTTL). For advanced
// settings pass BoundedRegistryConfig or BoundedOption values, or use
// NewBoundedRegistryWithConfig.
func NewBoundedRegistry(args ...interface{}) *BoundedRegistry {
	cfg := BoundedRegistryConfig{}
	var options []BoundedOption
	var durations []time.Duration
	maxSet := false
	shardSet := false

	for _, arg := range args {
		switch v := arg.(type) {
		case int:
			if !maxSet {
				cfg.MaxSize = v
				maxSet = true
			} else if !shardSet {
				cfg.ShardCount = v
				shardSet = true
			}
		case time.Duration:
			durations = append(durations, v)
		case BoundedRegistryConfig:
			cfg = v
		case BoundedOption:
			options = append(options, v)
		case func(*BoundedRegistryConfig):
			options = append(options, BoundedOption(v))
		default:
			if n, ok := integerArg(arg); ok {
				if !maxSet {
					cfg.MaxSize = n
					maxSet = true
				} else if !shardSet {
					cfg.ShardCount = n
					shardSet = true
				}
				continue
			}
			rv := reflect.ValueOf(arg)
			if rv.Kind() == reflect.Func && rv.Type().NumIn() == 1 &&
				rv.Type().In(0) == reflect.TypeOf(&BoundedRegistryConfig{}).Elem() {
				options = append(options, func(c *BoundedRegistryConfig) {
					rv.Call([]reflect.Value{reflect.ValueOf(c)})
				})
			}
		}
	}

	if cfg.IdleTTL == 0 && cfg.TTL == 0 && cfg.MaxIdle == 0 && cfg.IdleTimeout == 0 &&
		cfg.IdleDuration == 0 && cfg.ExpireAfter == 0 && len(durations) > 0 {
		cfg.IdleTTL = durations[0]
	}
	for _, option := range options {
		option(&cfg)
	}
	return NewBoundedRegistryWithConfig(cfg)
}

// NewBoundedRegistryWithOptions constructs a registry using options.
func NewBoundedRegistryWithOptions(options ...BoundedOption) *BoundedRegistry {
	cfg := BoundedRegistryConfig{}
	for _, option := range options {
		option(&cfg)
	}
	return NewBoundedRegistryWithConfig(cfg)
}

// NewShardedRegistry constructs a lock-sharded bounded registry. Arguments
// may use either (shardCount, maxSize, idleTTL) or (maxSize, idleTTL).
func NewShardedRegistry(args ...interface{}) *BoundedRegistry {
	cfg := BoundedRegistryConfig{ShardCount: defaultBoundedShardCount}
	var options []BoundedOption
	var durations []time.Duration
	var ints []int
	for _, arg := range args {
		switch v := arg.(type) {
		case int:
			ints = append(ints, v)
		case time.Duration:
			durations = append(durations, v)
		case BoundedRegistryConfig:
			cfg = v
		case BoundedOption:
			options = append(options, v)
		case func(*BoundedRegistryConfig):
			options = append(options, BoundedOption(v))
		default:
			if n, ok := integerArg(arg); ok {
				ints = append(ints, n)
			}
		}
	}
	if len(ints) >= 2 {
		cfg.ShardCount = ints[0]
		cfg.MaxSize = ints[1]
	} else if len(ints) == 1 {
		cfg.MaxSize = ints[0]
	}
	if len(durations) > 0 {
		cfg.IdleTTL = durations[0]
	}
	for _, option := range options {
		option(&cfg)
	}
	return NewBoundedRegistryWithConfig(cfg)
}

// NewShardedBoundedRegistry constructs a registry with explicit shard count.
func NewShardedBoundedRegistry(shardCount, maxSize int, idleTTL time.Duration) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{
		ShardCount: shardCount,
		MaxSize:    maxSize,
		IdleTTL:    idleTTL,
	})
}

// NewCappedRegistry constructs a size-capped registry without idle eviction.
func NewCappedRegistry(maxSize int) *BoundedRegistry {
	return NewBoundedRegistryWithConfig(BoundedRegistryConfig{MaxSize: maxSize})
}

// NewLimitedRegistry is an alias for NewCappedRegistry.
func NewLimitedRegistry(maxSize int) *BoundedRegistry {
	return NewCappedRegistry(maxSize)
}

// NewEvictingRegistry constructs a capacity- and idle-evicting registry.
func NewEvictingRegistry(maxSize int, idleTTL time.Duration) *BoundedRegistry {
	return NewBoundedRegistry(maxSize, idleTTL)
}

// NewIdleRegistry is an alias for NewEvictingRegistry.
func NewIdleRegistry(maxSize int, idleTTL time.Duration) *BoundedRegistry {
	return NewEvictingRegistry(maxSize, idleTTL)
}

func normalizeShardCount(shards, _ int) int {
	if shards < 1 {
		shards = 1
	}
	power := 1
	for power*2 <= shards {
		power *= 2
	}
	return power
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstPositiveDuration(values ...time.Duration) time.Duration {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func integerArg(value interface{}) (int, bool) {
	if _, ok := value.(time.Duration); ok {
		return 0, false
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return int(rv.Uint()), true
	default:
		return 0, false
	}
}

// Len returns the current number of registered metrics.
func (r *BoundedRegistry) Len() int {
	if r == nil {
		return 0
	}
	return int(atomic.LoadInt64(&r.size))
}

// MaxSize returns the configured hard cap.
func (r *BoundedRegistry) MaxSize() int { return r.maxSize }

// IdleTTL returns the configured idle TTL.
func (r *BoundedRegistry) IdleTTL() time.Duration { return r.idleTTL }

// ShardCount returns the number of lock shards.
func (r *BoundedRegistry) ShardCount() int { return len(r.shards) }

// EvictionCount returns the number of removals performed by the registry,
// including explicit unregistration.
func (r *BoundedRegistry) EvictionCount() int64 {
	return atomic.LoadInt64(&r.evictions)
}

// BoundedRegistryStats contains registry capacity and eviction statistics.
type BoundedRegistryStats struct {
	Size       int
	MaxSize    int
	ShardCount int
	Evictions  int64
}

// Stats returns a point-in-time snapshot of registry statistics.
func (r *BoundedRegistry) Stats() BoundedRegistryStats {
	return BoundedRegistryStats{
		Size:       r.Len(),
		MaxSize:    r.maxSize,
		ShardCount: len(r.shards),
		Evictions:  r.EvictionCount(),
	}
}

// StartJanitor starts a background idle sweep at interval. Calling it more than
// once is safe; only the first interval is used.
func (r *BoundedRegistry) StartJanitor(interval time.Duration) {
	if interval <= 0 {
		return
	}
	r.closeOnce.Do(func() {
		r.stopJanitor = make(chan struct{})
		r.janitorDone = make(chan struct{})
		ticker := time.NewTicker(interval)
		go func() {
			defer ticker.Stop()
			defer close(r.janitorDone)
			for {
				select {
				case <-ticker.C:
					r.EvictIdle()
				case <-r.stopJanitor:
					return
				}
			}
		}()
	})
}

// StopJanitor stops the background sweep. It is safe to call when no janitor
// is running.
func (r *BoundedRegistry) StopJanitor() {
	if r.stopJanitor == nil {
		return
	}
	select {
	case <-r.stopJanitor:
	default:
		close(r.stopJanitor)
	}
	if r.janitorDone != nil {
		<-r.janitorDone
	}
}

// Close stops the janitor. The registry remains usable after Close; only the
// background goroutine is released.
func (r *BoundedRegistry) Close() error {
	r.StopJanitor()
	return nil
}

// EvictIdle removes all entries older than IdleTTL and returns the number
// removed.
func (r *BoundedRegistry) EvictIdle() int {
	if r.idleTTL <= 0 {
		return 0
	}
	removed := 0
	for _, shard := range r.shards {
		shard.mu.Lock()
		before := len(shard.items)
		post := r.evictIdleLocked(shard, r.now())
		removed += before - len(shard.items)
		shard.mu.Unlock()
		runEvictionPostActions(post)
	}
	return removed
}

// Each calls f for every currently registered metric.
func (r *BoundedRegistry) Each(f func(string, interface{})) {
	for _, shard := range r.shards {
		shard.mu.RLock()
		metrics := make([]metricKV, 0, len(shard.items))
		for name, entry := range shard.items {
			metrics = append(metrics, metricKV{name: name, value: entry.metric})
		}
		shard.mu.RUnlock()
		for i := range metrics {
			f(metrics[i].name, metrics[i].value)
		}
	}
}

// Get returns the metric for name or nil.
func (r *BoundedRegistry) Get(name string) interface{} {
	shard := r.shardFor(name)
	shard.mu.RLock()
	entry, ok := shard.items[name]
	if !ok {
		shard.mu.RUnlock()
		return nil
	}
	now := r.now()
	if r.idleTTL > 0 && now.Sub(time.Unix(0, entryUpdatedNano(entry))) >= r.idleTTL {
		shard.mu.RUnlock()
		r.removeName(name, EvictionIdle, true)
		return nil
	}
	atomic.StoreInt64(&entry.accessNano, now.UnixNano())
	metric := entry.metric
	shard.mu.RUnlock()
	return metric
}

// GetOrRegister returns an existing metric or creates/registers i. If i is a
// function, it is called lazily after a cache miss.
func (r *BoundedRegistry) GetOrRegister(name string, i interface{}) interface{} {
	if metric := r.getExisting(name); metric != nil {
		return metric
	}

	shard := r.shardFor(name)
	var metric interface{}
	var state *int32
	prepared := false
	if reflect.ValueOf(i).Kind() != reflect.Func && !isSupportedMetric(i) {
		return i
	}
	for {
		shard.mu.Lock()
		var post []func()
		if entry, ok := shard.items[name]; ok {
			if r.idleTTL == 0 || r.now().Sub(time.Unix(0, entryUpdatedNano(entry))) < r.idleTTL {
				atomic.StoreInt64(&entry.accessNano, r.now().UnixNano())
				existing := entry.metric
				shard.mu.Unlock()
				return existing
			}
			post = append(post, r.removeEntryLocked(shard, name, entry, EvictionIdle)...)
		}
		if !prepared {
			if value := reflect.ValueOf(i); value.Kind() == reflect.Func {
				result := value.Call(nil)
				if len(result) == 0 {
					shard.mu.Unlock()
					runEvictionPostActions(post)
					return nil
				}
				i = result[0].Interface()
			}
			if !isSupportedMetric(i) {
				shard.mu.Unlock()
				runEvictionPostActions(post)
				return i
			}
			metric, state = r.prepareMetric(name, i)
			prepared = true
		}

		now := r.now().UnixNano()
		entry := &boundedEntry{metric: metric, state: state, updateNano: now, accessNano: now}
		insertPost, err := r.insertLocked(shard, name, entry, r.now())
		post = append(post, insertPost...)
		if errors.Is(err, errEvictElsewhere) {
			shard.mu.Unlock()
			runEvictionPostActions(post)
			runEvictionPostActions(r.evictOneGlobal())
			continue
		}
		shard.mu.Unlock()
		runEvictionPostActions(post)
		if err != nil {
			panic(err)
		}
		return metric
	}
}

// Register stores i under name or returns DuplicateMetric.
func (r *BoundedRegistry) Register(name string, i interface{}) error {
	if atomic.LoadInt32(&r.closed) != 0 {
		return ErrRegistryClosed
	}
	if !isSupportedMetric(i) {
		return &UnsupportedMetricError{Metric: i}
	}

	metric, state := r.prepareMetric(name, i)
	shard := r.shardFor(name)
	for {
		shard.mu.Lock()
		if entry, ok := shard.items[name]; ok {
			shard.mu.Unlock()
			_ = entry
			return DuplicateMetric(name)
		}
		now := r.now().UnixNano()
		entry := &boundedEntry{metric: metric, state: state, updateNano: now, accessNano: now}
		post, err := r.insertLocked(shard, name, entry, r.now())
		if errors.Is(err, errEvictElsewhere) {
			shard.mu.Unlock()
			runEvictionPostActions(post)
			runEvictionPostActions(r.evictOneGlobal())
			continue
		}
		shard.mu.Unlock()
		runEvictionPostActions(post)
		return err
	}
}

// RunHealthchecks runs all registered healthchecks without holding registry
// locks while Check executes.
func (r *BoundedRegistry) RunHealthchecks() {
	var healthchecks []Healthcheck
	r.Each(func(_ string, metric interface{}) {
		if h, ok := metric.(Healthcheck); ok {
			healthchecks = append(healthchecks, h)
		}
	})
	for _, h := range healthchecks {
		h.Check()
	}
}

// GetAll returns all metric values in the format used by reporters.
func (r *BoundedRegistry) GetAll() map[string]map[string]interface{} {
	return getAllMetrics(r)
}

// MarshalJSON implements json.Marshaler.
func (r *BoundedRegistry) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.GetAll())
}

// Unregister removes name and stops stoppable metrics.
func (r *BoundedRegistry) Unregister(name string) {
	r.removeName(name, EvictionUnregistered, false)
}

// UnregisterAll removes every metric.
func (r *BoundedRegistry) UnregisterAll() {
	var post []func()
	for _, shard := range r.shards {
		shard.mu.Lock()
		for name, entry := range shard.items {
			post = append(post, r.removeEntryLocked(shard, name, entry, EvictionCleared)...)
		}
		shard.mu.Unlock()
	}
	runEvictionPostActions(post)
}

func (r *BoundedRegistry) getExisting(name string) interface{} {
	shard := r.shardFor(name)
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	entry, ok := shard.items[name]
	if !ok {
		return nil
	}
	if r.idleTTL > 0 && r.now().Sub(time.Unix(0, entryUpdatedNano(entry))) >= r.idleTTL {
		return nil
	}
	now := r.now().UnixNano()
	atomic.StoreInt64(&entry.accessNano, now)
	return entry.metric
}

func (r *BoundedRegistry) removeName(name string, reason EvictionReason, idleOnly bool) {
	shard := r.shardFor(name)
	shard.mu.Lock()
	entry, ok := shard.items[name]
	if !ok {
		shard.mu.Unlock()
		return
	}
	if idleOnly && r.idleTTL > 0 &&
		r.now().Sub(time.Unix(0, entryUpdatedNano(entry))) < r.idleTTL {
		shard.mu.Unlock()
		return
	}
	post := r.removeEntryLocked(shard, name, entry, reason)
	shard.mu.Unlock()
	runEvictionPostActions(post)
}

func (r *BoundedRegistry) shardFor(name string) *registryShard {
	h := fnv32a(name)
	return r.shards[int(h&uint32(len(r.shards)-1))]
}

func fnv32a(value string) uint32 {
	const offset = uint32(2166136261)
	const prime = uint32(16777619)
	h := offset
	for i := 0; i < len(value); i++ {
		h ^= uint32(value[i])
		h *= prime
	}
	return h
}

func entryUpdatedNano(entry *boundedEntry) int64 {
	return atomic.LoadInt64(&entry.updateNano)
}

func (r *BoundedRegistry) prepareMetric(name string, metric interface{}) (interface{}, *int32) {
	switch c := metric.(type) {
	case *StandardCounter:
		if binding, ok := c.writeHook().(*counterBinding); ok && binding.r == r && binding.name == name {
			return c, &binding.state
		}
		binding := &counterBinding{r: r, name: name}
		c.setWriteHook(binding)
		return c, &binding.state
	case Counter:
		if managed, ok := c.(*managedCounter); ok && managed.r == r && managed.name == name {
			return managed, &managed.state
		}
		managed := &managedCounter{Counter: c, r: r, name: name}
		return managed, &managed.state
	default:
		return metric, nil
	}
}

func isSupportedMetric(metric interface{}) bool {
	switch metric.(type) {
	case Counter, Gauge, GaugeFloat64, Healthcheck, Histogram, Meter, Timer:
		return true
	default:
		return false
	}
}

// UnsupportedMetricError is returned when registering a value that does not
// implement one of the metrics interfaces.
type UnsupportedMetricError struct {
	Metric interface{}
}

func (e *UnsupportedMetricError) Error() string {
	return fmt.Sprintf("metrics: unsupported metric type %T", e.Metric)
}

var errEvictElsewhere = errors.New("metrics: evict from another shard")

func (r *BoundedRegistry) insertLocked(shard *registryShard, name string, entry *boundedEntry, now time.Time) ([]func(), error) {
	var post []func()
	if r.maxSize <= 0 {
		shard.items[name] = entry
		atomic.AddInt64(&r.size, 1)
		if entry.state != nil {
			atomic.StoreInt32(entry.state, handleActive)
		}
		return nil, nil
	}

	if atomic.LoadInt64(&r.size) >= int64(r.maxSize) {
		post = append(post, r.evictIdleLocked(shard, now)...)
	}
	for {
		current := atomic.LoadInt64(&r.size)
		if current < int64(r.maxSize) {
			if atomic.CompareAndSwapInt64(&r.size, current, current+1) {
				shard.items[name] = entry
				if entry.state != nil {
					atomic.StoreInt32(entry.state, handleActive)
				}
				return post, nil
			}
			continue
		}

		if r.rejectFull {
			return post, &RegistryFullError{MaxSize: r.maxSize, Name: name}
		}
		if victim, victimEntry := leastRecentlyUpdated(shard.items); victim != "" {
			post = append(post, r.removeEntryLocked(shard, victim, victimEntry, EvictionCapacity)...)
			continue
		}
		return post, errEvictElsewhere
	}
}

func (r *BoundedRegistry) evictOneGlobal() []func() {
	for _, shard := range r.shards {
		shard.mu.Lock()
		name, entry := leastRecentlyUpdated(shard.items)
		var post []func()
		if entry != nil {
			post = r.removeEntryLocked(shard, name, entry, EvictionCapacity)
		}
		shard.mu.Unlock()
		if entry != nil {
			return post
		}
	}
	return nil
}

func (r *BoundedRegistry) evictIdleLocked(shard *registryShard, now time.Time) []func() {
	if r.idleTTL <= 0 {
		return nil
	}
	var post []func()
	for name, entry := range shard.items {
		if now.Sub(time.Unix(0, entryUpdatedNano(entry))) >= r.idleTTL {
			post = append(post, r.removeEntryLocked(shard, name, entry, EvictionIdle)...)
		}
	}
	return post
}

func leastRecentlyUpdated(items map[string]*boundedEntry) (string, *boundedEntry) {
	var victim string
	var victimEntry *boundedEntry
	var victimTime int64
	for name, entry := range items {
		activity := entryUpdatedNano(entry)
		if victimEntry == nil || activity < victimTime || (activity == victimTime && name < victim) {
			victim = name
			victimEntry = entry
			victimTime = activity
		}
	}
	return victim, victimEntry
}

func (r *BoundedRegistry) removeEntryLocked(shard *registryShard, name string, entry *boundedEntry, reason EvictionReason) []func() {
	if current, ok := shard.items[name]; !ok || current != entry {
		return nil
	}
	delete(shard.items, name)
	atomic.AddInt64(&r.evictions, 1)
	atomic.AddInt64(&r.size, -1)

	if entry.state != nil {
		if reason == EvictionUnregistered || reason == EvictionCleared {
			atomic.StoreInt32(entry.state, handleManual)
		} else {
			atomic.StoreInt32(entry.state, handleEvicted)
		}
	}

	var post []func()
	if stoppable, ok := entry.metric.(Stoppable); ok {
		post = append(post, stoppable.Stop)
	}
	if r.onEvict != nil {
		callback := r.onEvict
		metric := entry.metric
		post = append(post, func() { callback(name, metric, reason) })
	}
	return post
}

func runEvictionPostActions(actions []func()) {
	for _, action := range actions {
		action()
	}
}

func (r *BoundedRegistry) writeCounter(name string, metric interface{}, state *int32, delta int64, apply func(int64)) error {
	if r == nil {
		apply(delta)
		return nil
	}
	if atomic.LoadInt32(&r.closed) != 0 {
		return ErrRegistryClosed
	}

	shard := r.shardFor(name)
	shard.mu.RLock()
	if entry, ok := shard.items[name]; ok && entry.metric == metric {
		apply(delta)
		nowNano := r.now().UnixNano()
		atomic.StoreInt64(&entry.updateNano, nowNano)
		atomic.StoreInt64(&entry.accessNano, nowNano)
		if state != nil {
			atomic.StoreInt32(state, handleActive)
		}
		shard.mu.RUnlock()
		return nil
	}
	shard.mu.RUnlock()

	for {
		shard.mu.Lock()

		if state != nil && atomic.LoadInt32(state) == handleManual {
			shard.mu.Unlock()
			apply(delta)
			return nil
		}

		entry, ok := shard.items[name]
		if ok && entry.metric != metric {
			shard.mu.Unlock()
			return &MetricNameConflictError{Name: name}
		}

		var post []func()
		if !ok {
			if state != nil && atomic.LoadInt32(state) == handleEvicted && r.strict {
				shard.mu.Unlock()
				return &MetricEvictedError{Name: name, MaxSize: r.maxSize, IdleTTL: r.idleTTL}
			}
			now := r.now()
			newEntry := &boundedEntry{
				metric:     metric,
				state:      state,
				updateNano: now.UnixNano(),
				accessNano: now.UnixNano(),
			}
			var err error
			post, err = r.insertLocked(shard, name, newEntry, now)
			if errors.Is(err, errEvictElsewhere) {
				shard.mu.Unlock()
				runEvictionPostActions(post)
				runEvictionPostActions(r.evictOneGlobal())
				continue
			}
			if err != nil {
				shard.mu.Unlock()
				runEvictionPostActions(post)
				return err
			}
		} else {
			nowNano := r.now().UnixNano()
			atomic.StoreInt64(&entry.updateNano, nowNano)
			atomic.StoreInt64(&entry.accessNano, nowNano)
		}

		apply(delta)
		if state != nil {
			atomic.StoreInt32(state, handleActive)
		}
		shard.mu.Unlock()
		runEvictionPostActions(post)
		return nil
	}
}

func (r *BoundedRegistry) touchMetricWrite(name string, metric interface{}) {
	if r == nil {
		return
	}
	shard := r.shardFor(name)
	shard.mu.RLock()
	if entry, ok := shard.items[name]; ok && entry.metric == metric {
		now := r.now().UnixNano()
		atomic.StoreInt64(&entry.updateNano, now)
		atomic.StoreInt64(&entry.accessNano, now)
	}
	shard.mu.RUnlock()
}

// MetricEvictedError provides eviction details.
type MetricEvictedError struct {
	Name    string
	MaxSize int
	IdleTTL time.Duration
	Reason  EvictionReason
}

func (e *MetricEvictedError) Error() string {
	return fmt.Sprintf("%s: %s", ErrMetricEvicted.Error(), e.Name)
}

func (e *MetricEvictedError) Unwrap() error { return ErrMetricEvicted }

// MetricNameConflictError provides the conflicting metric name.
type MetricNameConflictError struct {
	Name string
}

func (e *MetricNameConflictError) Error() string {
	return fmt.Sprintf("%s: %s", ErrMetricNameInUse.Error(), e.Name)
}

func (e *MetricNameConflictError) Unwrap() error { return ErrMetricNameInUse }

// RegistryFullError reports a rejected registration at capacity.
type RegistryFullError struct {
	MaxSize int
	Name    string
}

func (e *RegistryFullError) Error() string {
	if e.Name == "" {
		return ErrRegistryFull.Error()
	}
	return fmt.Sprintf("%s: %s (max=%d)", ErrRegistryFull.Error(), e.Name, e.MaxSize)
}

func (e *RegistryFullError) Unwrap() error { return ErrRegistryFull }

type counterBinding struct {
	r     *BoundedRegistry
	name  string
	state int32
}

func (b *counterBinding) addCounter(c *StandardCounter, delta int64) error {
	return b.r.writeCounter(b.name, c, &b.state, delta, func(n int64) {
		atomic.AddInt64(&c.count, n)
	})
}

func (b *counterBinding) clearCounter(c *StandardCounter) error {
	atomic.StoreInt64(&c.count, 0)
	b.r.touchMetricWrite(b.name, c)
	return nil
}

// ResurrectableCounter is a counter handle that can re-register itself after
// capacity/idle eviction.
type ResurrectableCounter interface {
	WritableCounter
	Evicted() bool
	Resurrect() error
	IncE(int64) error
	DecE(int64) error
}

// FailableCounter is an alias for the explicit error-returning counter API.
type FailableCounter interface {
	Counter
	Add(int64) error
	IncE(int64) error
	DecE(int64) error
}

type managedCounter struct {
	Counter
	r     *BoundedRegistry
	name  string
	state int32
}

func (c *managedCounter) Add(n int64) error {
	return c.r.writeCounter(c.name, c, &c.state, n, func(delta int64) {
		if delta >= 0 {
			c.Counter.Inc(delta)
		} else {
			c.Counter.Dec(-delta)
		}
	})
}

func (c *managedCounter) IncE(n int64) error { return c.Add(n) }

func (c *managedCounter) DecE(n int64) error { return c.Add(-n) }

func (c *managedCounter) Inc(n int64) {
	if err := c.Add(n); err != nil {
		panic(err)
	}
}

func (c *managedCounter) Dec(n int64) {
	if err := c.Add(-n); err != nil {
		panic(err)
	}
}

func (c *managedCounter) Evicted() bool {
	return atomic.LoadInt32(&c.state) == handleEvicted
}

func (c *managedCounter) Resurrect() error { return c.Add(0) }

func (c *managedCounter) Clear() {
	c.Counter.Clear()
	c.r.touchMetricWrite(c.name, c)
}

// IncE performs Inc and returns an eviction/conflict error instead of panic.
func (c *StandardCounter) IncE(n int64) error { return c.Add(n) }

// DecE performs Dec and returns an eviction/conflict error instead of panic.
func (c *StandardCounter) DecE(n int64) error { return c.Add(-n) }

// Evicted reports whether a bounded registry has evicted this counter handle.
func (c *StandardCounter) Evicted() bool {
	binding, ok := c.writeHook().(*counterBinding)
	return ok && atomic.LoadInt32(&binding.state) == handleEvicted
}

// Resurrect re-registers an evicted counter with its bounded registry.
func (c *StandardCounter) Resurrect() error { return c.Add(0) }
