package metrics

import (
	"sync"
	"sync/atomic"
	"time"
)

// TimeWindowSample is a time-bucketed sliding-window Sample.
//
// Unlike ExpDecaySample (whose decay speed is fixed by alpha at construction)
// and UniformSample (which is unrelated to time), a TimeWindowSample slices
// the stream into fixed-width time buckets and keeps only the most recent
// bucketCount buckets on a fixed-size ring. Memory is therefore capped by
// bucketCount and does not grow with process uptime: old slots are reused in
// place and their backing arrays are retained.
//
// Percentiles for any concrete window, e.g. "the last minute", are computed by
// WindowPercentile / WindowPercentiles over the buckets covered by that
// window. The ordinary Sample methods answer over every retained bucket (the
// full ring span).
//
// The write path never performs a global window rotation: locating the active
// slot and resetting a stale slot is constant work done under that slot's own
// mutex, so a query aggregating old buckets does not block writers landing in
// the current bucket. No background goroutine or ticker is used.
type TimeWindowSample struct {
	// count is first to keep it 64-bit aligned for atomic access on 32-bit.
	count       int64
	bucketWidth int64 // nanoseconds
	bucketCount int
	buckets     []timeWindowBucket
	now         func() time.Time
}

// timeWindowBucket is one ring slot. index is the absolute bucket index
// (unix nanoseconds / bucketWidth) currently stored in the slot; a slot whose
// index differs from the wanted index is stale and gets reused.
type timeWindowBucket struct {
	mu     sync.Mutex
	index  int64
	values []int64
}

// NewTimeWindowSample constructs a time-bucketed sliding-window sample.
//
// bucketWidth is the size of one time slice and bucketCount is the hard cap on
// how many buckets (the most recent ones) are retained. The total retained
// span is bucketWidth * bucketCount. Invalid arguments are normalized to safe
// defaults: a non-positive bucketWidth becomes one second and a bucketCount
// below one becomes one.
func NewTimeWindowSample(bucketWidth time.Duration, bucketCount int) *TimeWindowSample {
	if bucketWidth <= 0 {
		bucketWidth = time.Second
	}
	if bucketCount < 1 {
		bucketCount = 1
	}
	return &TimeWindowSample{
		bucketWidth: int64(bucketWidth),
		bucketCount: bucketCount,
		buckets:     make([]timeWindowBucket, bucketCount),
		now:         time.Now,
	}
}

// SlidingWindowSample is the conventional alias for TimeWindowSample.
type SlidingWindowSample = TimeWindowSample

// NewSlidingWindowSample is an alias for NewTimeWindowSample.
func NewSlidingWindowSample(bucketWidth time.Duration, bucketCount int) *SlidingWindowSample {
	return NewTimeWindowSample(bucketWidth, bucketCount)
}

// NewWindowSample is an alias for NewTimeWindowSample.
func NewWindowSample(bucketWidth time.Duration, bucketCount int) *TimeWindowSample {
	return NewTimeWindowSample(bucketWidth, bucketCount)
}

// NewSpanWindowSample splits totalSpan into bucketCount equal buckets. It is a
// convenience constructor when the window span, rather than the bucket width,
// is the natural configuration value.
func NewSpanWindowSample(totalSpan time.Duration, bucketCount int) *TimeWindowSample {
	if bucketCount < 1 {
		bucketCount = 1
	}
	return NewTimeWindowSample(totalSpan/time.Duration(bucketCount), bucketCount)
}

// WithClock replaces the time source used by Update and window anchoring and
// returns s. It is meant for deterministic tests and must be called before
// concurrent Updates; it is not safe to swap the clock while writers run.
func (s *TimeWindowSample) WithClock(now func() time.Time) *TimeWindowSample {
	if now != nil {
		s.now = now
	}
	return s
}

var _ Sample = (*TimeWindowSample)(nil)

// Update samples a new value at the current wall-clock time.
func (s *TimeWindowSample) Update(v int64) {
	s.UpdateAt(s.now(), v)
}

// update keeps the timestamp-injection convention used by ExpDecaySample's
// test-only update method.
func (s *TimeWindowSample) update(t time.Time, v int64) {
	s.UpdateAt(t, v)
}

// UpdateAt samples a new value attributed to t. It is useful for backfilling
// and for deterministic tests that must advance the window without sleeping.
func (s *TimeWindowSample) UpdateAt(t time.Time, v int64) {
	idx := t.UnixNano() / s.bucketWidth
	slot := &s.buckets[ringSlot(idx, s.bucketCount)]
	slot.mu.Lock()
	switch {
	case slot.index == idx:
		slot.values = append(slot.values, v)
	case slot.index < idx:
		// Stale or never-used slot wrapping around the ring: reset in place so
		// the backing array stays warm and memory stays capped.
		slot.values = append(slot.values[:0], v)
		slot.index = idx
	default:
		// The slot already hosts a newer bucket, which only happens for an
		// out-of-order write older than a full ring turn. The value is beyond
		// the retained window and is discarded.
	}
	slot.mu.Unlock()
	atomic.AddInt64(&s.count, 1)
}

// Clear drops every retained value and resets the total count.
func (s *TimeWindowSample) Clear() {
	for i := range s.buckets {
		bucket := &s.buckets[i]
		bucket.mu.Lock()
		bucket.index = 0
		bucket.values = bucket.values[:0]
		bucket.mu.Unlock()
	}
	atomic.StoreInt64(&s.count, 0)
}

// Count returns the number of Update/UpdateAt calls, which may exceed the
// number of values currently retained in the window.
func (s *TimeWindowSample) Count() int64 {
	return atomic.LoadInt64(&s.count)
}

// BucketWidth returns the configured width of one time bucket.
func (s *TimeWindowSample) BucketWidth() time.Duration {
	return time.Duration(s.bucketWidth)
}

// BucketCount returns the fixed cap on retained buckets (the ring size).
func (s *TimeWindowSample) BucketCount() int {
	return s.bucketCount
}

// ActiveBucketCount returns how many ring slots currently hold a non-empty
// bucket inside the retained span ending at now.
func (s *TimeWindowSample) ActiveBucketCount() int {
	end := s.currentIndex()
	from := end - int64(s.bucketCount) + 1
	active := 0
	for i := range s.buckets {
		bucket := &s.buckets[i]
		bucket.mu.Lock()
		if bucket.index >= from && bucket.index <= end && len(bucket.values) > 0 {
			active++
		}
		bucket.mu.Unlock()
	}
	return active
}

// BucketCounts returns the value counts of the retained buckets ordered from
// oldest to newest, including zero entries for empty or expired positions.
func (s *TimeWindowSample) BucketCounts() []int {
	end := s.currentIndex()
	from := end - int64(s.bucketCount) + 1
	counts := make([]int, s.bucketCount)
	for i := range s.buckets {
		bucket := &s.buckets[i]
		bucket.mu.Lock()
		idx := bucket.index
		n := len(bucket.values)
		bucket.mu.Unlock()
		if idx >= from && idx <= end {
			counts[idx-from] = n
		}
	}
	return counts
}

// Max returns the maximum value across all retained buckets.
func (s *TimeWindowSample) Max() int64 { return SampleMax(s.Values()) }

// Mean returns the mean across all retained buckets.
func (s *TimeWindowSample) Mean() float64 { return SampleMean(s.Values()) }

// Min returns the minimum value across all retained buckets.
func (s *TimeWindowSample) Min() int64 { return SampleMin(s.Values()) }

// Percentile returns an arbitrary percentile over every retained bucket.
func (s *TimeWindowSample) Percentile(p float64) float64 {
	return SamplePercentile(s.Values(), p)
}

// Percentiles returns arbitrary percentiles over every retained bucket.
func (s *TimeWindowSample) Percentiles(ps []float64) []float64 {
	return SamplePercentiles(s.Values(), ps)
}

// Size returns the number of values currently retained across all buckets.
func (s *TimeWindowSample) Size() int {
	return s.sizeWithin(s.bucketCount)
}

// Snapshot returns a read-only SampleSnapshot of every retained bucket.
func (s *TimeWindowSample) Snapshot() Sample {
	return &SampleSnapshot{
		count:  atomic.LoadInt64(&s.count),
		values: s.Values(),
	}
}

// StdDev returns the standard deviation across all retained buckets.
func (s *TimeWindowSample) StdDev() float64 { return SampleStdDev(s.Values()) }

// Sum returns the sum across all retained buckets.
func (s *TimeWindowSample) Sum() int64 { return SampleSum(s.Values()) }

// Values returns a copy of every value in the retained span.
func (s *TimeWindowSample) Values() []int64 {
	return s.collect(s.currentIndex(), s.bucketCount)
}

// Variance returns the variance across all retained buckets.
func (s *TimeWindowSample) Variance() float64 { return SampleVariance(s.Values()) }

// WindowSize returns the number of values inside the last window. The window
// is rounded up to whole buckets and capped at the retained ring span.
func (s *TimeWindowSample) WindowSize(window time.Duration) int {
	return s.sizeWithin(s.windowBucketCount(window))
}

// WindowValues returns a copy of the values inside the last window. The window
// is rounded up to whole buckets and capped at the retained ring span.
func (s *TimeWindowSample) WindowValues(window time.Duration) []int64 {
	return s.collect(s.currentIndex(), s.windowBucketCount(window))
}

// WindowPercentile returns a percentile over just the last window of length
// window, e.g. WindowPercentile(time.Minute, 0.99) for the one-minute p99.
func (s *TimeWindowSample) WindowPercentile(window time.Duration, p float64) float64 {
	return SamplePercentile(s.WindowValues(window), p)
}

// WindowPercentiles returns percentiles over just the last window.
func (s *TimeWindowSample) WindowPercentiles(window time.Duration, ps []float64) []float64 {
	return SamplePercentiles(s.WindowValues(window), ps)
}

// PercentileForWindow is an alias for WindowPercentile.
func (s *TimeWindowSample) PercentileForWindow(window time.Duration, p float64) float64 {
	return s.WindowPercentile(window, p)
}

// PercentilesForWindow is an alias for WindowPercentiles.
func (s *TimeWindowSample) PercentilesForWindow(window time.Duration, ps []float64) []float64 {
	return s.WindowPercentiles(window, ps)
}

// BucketValues returns a copy of the values in the last buckets buckets.
func (s *TimeWindowSample) BucketValues(buckets int) []int64 {
	return s.collect(s.currentIndex(), buckets)
}

// BucketPercentile returns a percentile over the last buckets buckets.
func (s *TimeWindowSample) BucketPercentile(buckets int, p float64) float64 {
	return SamplePercentile(s.BucketValues(buckets), p)
}

// BucketPercentiles returns percentiles over the last buckets buckets.
func (s *TimeWindowSample) BucketPercentiles(buckets int, ps []float64) []float64 {
	return SamplePercentiles(s.BucketValues(buckets), ps)
}

func (s *TimeWindowSample) currentIndex() int64 {
	return s.now().UnixNano() / s.bucketWidth
}

// windowBucketCount converts a duration to the number of buckets to aggregate,
// rounding up and clamping to [1, bucketCount].
func (s *TimeWindowSample) windowBucketCount(window time.Duration) int {
	n := (int64(window) + s.bucketWidth - 1) / s.bucketWidth
	if n < 1 {
		n = 1
	}
	if n > int64(s.bucketCount) {
		n = int64(s.bucketCount)
	}
	return int(n)
}

// sizeWithin counts values in the last want buckets ending at end without
// allocating a result slice.
func (s *TimeWindowSample) sizeWithin(want int) int {
	end := s.currentIndex()
	if want <= 0 {
		return 0
	}
	if want > s.bucketCount {
		want = s.bucketCount
	}
	from := end - int64(want) + 1
	total := 0
	for i := range s.buckets {
		bucket := &s.buckets[i]
		bucket.mu.Lock()
		if bucket.index >= from && bucket.index <= end {
			total += len(bucket.values)
		}
		bucket.mu.Unlock()
	}
	return total
}

// collect copies the values of the last want buckets ending at end. Each
// bucket lock is held only for that bucket's slice copy, so aggregation never
// blocks writers on other buckets.
func (s *TimeWindowSample) collect(end int64, want int) []int64 {
	if want <= 0 {
		return []int64{}
	}
	if want > s.bucketCount {
		want = s.bucketCount
	}
	from := end - int64(want) + 1
	values := make([]int64, 0)
	for i := range s.buckets {
		bucket := &s.buckets[i]
		bucket.mu.Lock()
		if bucket.index >= from && bucket.index <= end {
			values = append(values, bucket.values...)
		}
		bucket.mu.Unlock()
	}
	return values
}

// ringSlot maps a possibly negative absolute bucket index to a ring position.
func ringSlot(index int64, size int) int {
	slot := index % int64(size)
	if slot < 0 {
		slot += int64(size)
	}
	return int(slot)
}
