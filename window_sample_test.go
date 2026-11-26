package metrics

import (
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var windowTestBase = time.Unix(1700000000, 0).UTC()

// A fixed ring must never grow past the configured bucket count and only the
// newest buckets survive a wrap.
func TestTimeWindowSampleRingCapAndEviction(t *testing.T) {
	s := NewTimeWindowSample(time.Second, 3)
	now := windowTestBase.Add(9 * time.Second)
	s.now = func() time.Time { return now }
	for i := 0; i < 10; i++ {
		s.UpdateAt(windowTestBase.Add(time.Duration(i)*time.Second), int64(i*10))
	}

	if got := s.BucketCount(); got != 3 {
		t.Fatalf("BucketCount(): 3 != %d", got)
	}
	if got := len(s.buckets); got != 3 {
		t.Fatalf("ring length grew: 3 != %d", got)
	}

	// Buckets 7, 8, 9 are the three most recent ones.
	got := s.Values()
	want := []int64{70, 80, 90}
	if len(got) != len(want) {
		t.Fatalf("Values(): want %v, got %v", want, got)
	}
	sortInt64s(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Values(): want %v, got %v", want, got)
	}

	// BucketCounts is ordered oldest -> newest for the retained positions.
	if counts := s.BucketCounts(); !reflect.DeepEqual(counts, []int{1, 1, 1}) {
		t.Fatalf("BucketCounts(): [1 1 1] != %v", counts)
	}
	if active := s.ActiveBucketCount(); active != 3 {
		t.Fatalf("ActiveBucketCount(): 3 != %d", active)
	}
}

// Running across hundreds of thousands of buckets must not grow the ring:
// exactly bucketCount slots remain and only the newest buckets survive.
func TestTimeWindowSampleBucketCountNeverGrows(t *testing.T) {
	const n = 16
	const turns = 200000
	s := NewTimeWindowSample(time.Millisecond, n)
	last := windowTestBase.Add((turns - 1) * time.Millisecond)
	s.now = func() time.Time { return last }

	for i := 0; i < turns; i++ {
		s.UpdateAt(windowTestBase.Add(time.Duration(i)*time.Millisecond), int64(i))
	}

	if s.BucketCount() != n || len(s.buckets) != n {
		t.Fatalf("bucket cap grew: BucketCount=%d ring=%d", s.BucketCount(), len(s.buckets))
	}
	if s.Size() != n {
		t.Fatalf("only the last %d one-value buckets should remain, got Size=%d", n, s.Size())
	}
	if s.Count() != turns {
		t.Fatalf("Count should track every write: %d != %d", turns, s.Count())
	}
}

// The headline scenario, driven by a fake clock: a slow wave keeps recent
// percentiles high, then a fast wave makes the recent-window percentile fall
// while the full retained range still remembers the slow values.
func TestWindowSampleSlowThenFast(t *testing.T) {
	const width = 100 * time.Millisecond
	s := NewTimeWindowSample(width, 60) // 6s of retention
	now := windowTestBase
	s.now = func() time.Time { return now }

	// Slow request wave: ten buckets of 40 x 1000ms.
	for b := 0; b < 10; b++ {
		ts := windowTestBase.Add(time.Duration(b) * width)
		for j := 0; j < 40; j++ {
			s.UpdateAt(ts, 1000)
		}
	}

	// Right after the slow wave the last-second p99 is still 1000.
	now = windowTestBase.Add(9 * width)
	if p := s.WindowPercentile(time.Second, 0.99); p != 1000 {
		t.Fatalf("slow wave window p99: 1000 != %v", p)
	}

	// Fast request wave lands ~4s later: 400 x 1ms in one bucket.
	fastAt := windowTestBase.Add(50 * width)
	for j := 0; j < 400; j++ {
		s.UpdateAt(fastAt, 1)
	}
	now = fastAt

	if n := s.WindowSize(time.Second); n != 400 {
		t.Fatalf("WindowSize(time.Second): 400 != %d", n)
	}
	if p := s.WindowPercentile(time.Second, 0.99); p != 1 {
		t.Fatalf("recent window p99 should fall to 1, got %v", p)
	}
	ps := s.WindowPercentiles(time.Second, []float64{0.5, 0.95, 0.99})
	for i, p := range ps {
		if p != 1 {
			t.Fatalf("recent window percentile[%d]: 1 != %v", i, p)
		}
	}

	// The whole retained span still contains the slow wave (400 slow vs 400
	// fast points), so its p99 stays at 1000.
	if p := s.Percentile(0.99); p != 1000 {
		t.Fatalf("full-range p99 should still remember slow wave: 1000 != %v", p)
	}

	// Two seconds of silence age the fast bucket out of a one-second window.
	now = fastAt.Add(2 * time.Second)
	if n := s.WindowSize(time.Second); n != 0 {
		t.Fatalf("aged window should be empty, got %d values", n)
	}
	if p := s.WindowPercentile(time.Second, 0.99); p != 0 {
		t.Fatalf("empty window percentile: 0 != %v", p)
	}
}

// Same slow-then-fast story against the real clock, the way a live caller or a
// repository test would exercise it. Generous timing margins keep it stable on
// a loaded machine.
func TestWindowSampleSlowThenFastRealtime(t *testing.T) {
	width := 50 * time.Millisecond
	s := NewTimeWindowSample(width, 240) // 12s retention

	for i := 0; i < 300; i++ {
		s.Update(1000) // slow requests
	}
	slowP95 := s.WindowPercentile(time.Second, 0.95)
	t.Logf("after slow wave: last-1s p95=%.1f full p95=%.1f", slowP95, s.Percentile(0.95))
	if slowP95 < 900 {
		t.Fatalf("slow wave recent p95 should be high, got %v", slowP95)
	}

	// Let the slow wave slide several buckets out of a two-bucket query.
	time.Sleep(300 * time.Millisecond)

	for i := 0; i < 300; i++ {
		s.Update(1) // fast requests
	}

	recentP95 := s.WindowPercentile(2*width, 0.95)
	fullP95 := s.Percentile(0.95)
	t.Logf("after fast wave: recent-2-bucket p95=%.1f full p95=%.1f", recentP95, fullP95)
	if recentP95 > 50 {
		t.Fatalf("recent p95 should fall toward 1 after fast wave, got %v", recentP95)
	}
	if fullP95 < 900 {
		t.Fatalf("full-range p95 should still include slow wave, got %v", fullP95)
	}
}

// Window durations map to whole buckets and are clamped to the ring.
func TestTimeWindowSampleWindowClamping(t *testing.T) {
	s := NewTimeWindowSample(100*time.Millisecond, 5)
	base := windowTestBase
	now := base.Add(400 * time.Millisecond)
	s.now = func() time.Time { return now }
	for i := 0; i < 5; i++ {
		s.UpdateAt(base.Add(time.Duration(i)*100*time.Millisecond), int64(i+1))
	}

	// Sub-bucket duration resolves to the current bucket only.
	if v := s.WindowValues(time.Millisecond); !reflect.DeepEqual(v, []int64{5}) {
		t.Fatalf("sub-bucket window: [5] != %v", v)
	}
	// Duration above the retained span returns all five buckets.
	if v := s.WindowValues(time.Hour); len(v) != 5 {
		t.Fatalf("oversized window should clamp to ring, got %v", v)
	}
	// Bucket-count based query: last two buckets hold 4 and 5.
	if v := s.BucketValues(2); len(v) != 2 {
		t.Fatalf("BucketValues(2): %v", v)
	}
	if p := s.BucketPercentile(5, 0.5); p < 1 || p > 5 {
		t.Fatalf("BucketPercentile out of range: %v", p)
	}
}

func TestTimeWindowSampleClearAndDefaults(t *testing.T) {
	s := NewTimeWindowSample(0, 0)
	if s.BucketWidth() != time.Second {
		t.Fatalf("default width: %v", s.BucketWidth())
	}
	if s.BucketCount() != 1 {
		t.Fatalf("default count: %d", s.BucketCount())
	}
	s.Update(7)
	s.Clear()
	if s.Count() != 0 || s.Size() != 0 {
		t.Fatalf("after Clear: count=%d size=%d", s.Count(), s.Size())
	}
	if len(s.Values()) != 0 {
		t.Fatalf("after Clear values: %v", s.Values())
	}
	if s.Percentile(0.99) != 0 {
		t.Fatal("empty sample percentile should be 0")
	}
}

// The sample plugs into Histogram/Timer and its Snapshot satisfies the
// *SampleSnapshot type assertion made by StandardHistogram.Snapshot.
func TestTimeWindowSampleHistogramIntegration(t *testing.T) {
	s := NewTimeWindowSample(time.Second, 10)
	var _ Sample = s
	h := NewHistogram(s)
	h.Update(5)
	h.Update(5)
	h.Update(7)
	if c := h.Count(); c != 3 {
		t.Fatalf("histogram count: 3 != %d", c)
	}
	snap := h.Snapshot()
	if c := snap.Count(); c != 3 {
		t.Fatalf("snapshot count: 3 != %d", c)
	}
	if p := snap.Percentile(0.5); p < 5 || p > 7 {
		t.Fatalf("snapshot percentile: %v", p)
	}
	raw := s.Snapshot()
	if _, ok := raw.(*SampleSnapshot); !ok {
		t.Fatalf("Snapshot() type: %T", raw)
	}
}

// Concurrent writers and window readers must be race-free and no Update may be
// lost from Count. Run with -race for the real guarantee.
func TestTimeWindowSampleConcurrent(t *testing.T) {
	s := NewTimeWindowSample(time.Millisecond, 32)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	var total int64

	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				for i := 0; i < 200; i++ {
					s.Update(int64(g*10000 + i))
				}
				atomic.AddInt64(&total, 200)
			}
		}(g)
	}

	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ps := []float64{0.5, 0.95, 0.99}
			for {
				select {
				case <-stop:
					return
				default:
				}
				_ = s.WindowPercentile(5*time.Millisecond, 0.99)
				_ = s.WindowPercentiles(50*time.Millisecond, ps)
				_ = s.Percentiles(ps)
				_ = s.Snapshot()
				_ = s.Values()
				_ = s.BucketCounts()
				_ = s.ActiveBucketCount()
			}
		}()
	}

	time.Sleep(200 * time.Millisecond)
	close(stop)
	wg.Wait()

	if got := s.Count(); got != atomic.LoadInt64(&total) {
		t.Fatalf("Count lost updates: recorded %d, wanted %d", got, total)
	}
}

func sortInt64s(v []int64) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j-1] > v[j]; j-- {
			v[j-1], v[j] = v[j], v[j-1]
		}
	}
}

func BenchmarkTimeWindowSampleUpdate(b *testing.B) {
	// Microsecond buckets make the ring rotate during the run so the benchmark
	// covers stale-slot reset as well as the fast append path.
	s := NewTimeWindowSample(time.Microsecond, 60)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Update(int64(i))
	}
}

func BenchmarkTimeWindowSampleUpdateParallel(b *testing.B) {
	s := NewTimeWindowSample(time.Microsecond, 60)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		v := int64(0)
		for pb.Next() {
			s.Update(v)
			v++
		}
	})
}

func BenchmarkTimeWindowSampleWindowPercentile(b *testing.B) {
	s := NewTimeWindowSample(time.Millisecond, 60)
	base := time.Unix(1700000000, 0)
	for i := 0; i < 60000; i++ {
		s.UpdateAt(base.Add(time.Duration(i)*time.Microsecond*100), int64(i))
	}
	now := base.Add(60000 * 100 * time.Microsecond)
	s.now = func() time.Time { return now }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.WindowPercentile(time.Second, 0.99)
	}
}
