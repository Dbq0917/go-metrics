package metrics

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const rescaleThreshold = time.Hour

// Samples maintain a statistically-significant selection of values from
// a stream.
type Sample interface {
	Clear()
	Count() int64
	Max() int64
	Mean() float64
	Min() int64
	Percentile(float64) float64
	Percentiles([]float64) []float64
	Size() int
	Snapshot() Sample
	StdDev() float64
	Sum() int64
	Update(int64)
	Values() []int64
	Variance() float64
}

// ExpDecaySample is an exponentially-decaying sample using a forward-decaying
// priority reservoir.  See Cormode et al's "Forward Decay: A Practical Time
// Decay Model for Streaming Systems".
//
// <http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf>
//
// The single reservoir is used while writes are uncontended.  If concurrent
// writers collide, the sample escalates once to a fixed set of independent
// shards whose capacities add up to the configured reservoir size.  Writers
// then touch only one shard and readers merge the shards, so the write path no
// longer has a process-wide sampling lock.
type ExpDecaySample struct {
	alpha         float64
	count         int64
	mutex         sync.Mutex
	reservoirSize int
	t0, t1        time.Time
	values        *expDecaySampleHeap

	escalateMutex sync.Mutex
	epoch         int64
	generation    int64
	blockSeq      uint64
	sharded       int32
	clearing      int32
	shards        []*expDecaySampleShard
	pickerPool    sync.Pool
}

// NewExpDecaySample constructs a new exponentially-decaying sample with the
// given reservoir size and alpha.
func NewExpDecaySample(reservoirSize int, alpha float64) Sample {
	if UseNilMetrics {
		return NilSample{}
	}
	s := &ExpDecaySample{
		alpha:         alpha,
		reservoirSize: reservoirSize,
		t0:            time.Now(),
		values:        newExpDecaySampleHeap(reservoirSize),
	}
	s.t1 = s.t0.Add(rescaleThreshold)
	return s
}

// Clear clears all samples.
func (s *ExpDecaySample) Clear() {
	s.escalateMutex.Lock()
	defer s.escalateMutex.Unlock()

	atomic.StoreInt32(&s.clearing, 1)
	atomic.AddInt64(&s.generation, 1)
	epoch := atomic.AddInt64(&s.epoch, 1)

	s.mutex.Lock()
	s.count = 0
	s.t0 = time.Now()
	s.t1 = s.t0.Add(rescaleThreshold)
	t0, t1 := s.t0, s.t1
	s.values.Clear()
	s.mutex.Unlock()

	for _, shard := range s.shards {
		shard.mutex.Lock()
		shard.epoch = epoch
		shard.count = 0
		shard.t0 = t0
		shard.t1 = t1
		shard.values.Clear()
		shard.mutex.Unlock()
	}
	atomic.StoreInt32(&s.clearing, 0)
}

// Count returns the number of samples recorded, which may exceed the
// reservoir size.
func (s *ExpDecaySample) Count() int64 {
	count, _ := s.snapshot()
	return count
}

// Max returns the maximum value in the sample, which may not be the maximum
// value ever to be part of the sample.
func (s *ExpDecaySample) Max() int64 {
	return SampleMax(s.Values())
}

// Mean returns the mean of the values in the sample.
func (s *ExpDecaySample) Mean() float64 {
	return SampleMean(s.Values())
}

// Min returns the minimum value in the sample, which may not be the minimum
// value ever to be part of the sample.
func (s *ExpDecaySample) Min() int64 {
	return SampleMin(s.Values())
}

// Percentile returns an arbitrary percentile of values in the sample.
func (s *ExpDecaySample) Percentile(p float64) float64 {
	return SamplePercentile(s.Values(), p)
}

// Percentiles returns a slice of arbitrary percentiles of values in the
// sample.
func (s *ExpDecaySample) Percentiles(ps []float64) []float64 {
	return SamplePercentiles(s.Values(), ps)
}

// Size returns the size of the sample, which is at most the reservoir size.
func (s *ExpDecaySample) Size() int {
	_, values := s.snapshot()
	return len(values)
}

// Snapshot returns a read-only copy of the sample.
func (s *ExpDecaySample) Snapshot() Sample {
	count, values := s.snapshot()
	return &SampleSnapshot{
		count:  count,
		values: values,
	}
}

// StdDev returns the standard deviation of the values in the sample.
func (s *ExpDecaySample) StdDev() float64 {
	return SampleStdDev(s.Values())
}

// Sum returns the sum of the values in the sample.
func (s *ExpDecaySample) Sum() int64 {
	return SampleSum(s.Values())
}

// Update samples a new value.
func (s *ExpDecaySample) Update(v int64) {
	s.updateAt(time.Now(), v)
}

// update samples a new value at a particular timestamp.  This is a method all
// its own to facilitate testing.
func (s *ExpDecaySample) update(t time.Time, v int64) {
	s.updateAt(t, v)
}

func (s *ExpDecaySample) updateAt(t time.Time, v int64) {
	generation := atomic.LoadInt64(&s.generation)
	if atomic.LoadInt32(&s.sharded) != 0 {
		s.updateSharded(t, v, generation)
		return
	}
	if s.mutex.TryLock() {
		if atomic.LoadInt32(&s.sharded) != 0 {
			s.mutex.Unlock()
			s.updateSharded(t, v, generation)
			return
		}
		if atomic.LoadInt64(&s.generation) != generation || atomic.LoadInt32(&s.clearing) != 0 {
			s.mutex.Unlock()
			return
		}
		s.updateLocked(t, v)
		s.mutex.Unlock()
		return
	}

	s.escalate()
	s.updateSharded(t, v, generation)
}

// Values returns a copy of the values in the sample.
func (s *ExpDecaySample) Values() []int64 {
	_, values := s.snapshot()
	return values
}

// Variance returns the variance of the values in the sample.
func (s *ExpDecaySample) Variance() float64 {
	return SampleVariance(s.Values())
}

func (s *ExpDecaySample) updateLocked(t time.Time, v int64) {
	s.count++
	if s.values.Size() == s.reservoirSize {
		s.values.Pop()
	}
	s.values.Push(expDecaySample{
		k: math.Exp(t.Sub(s.t0).Seconds()*s.alpha) / rand.Float64(),
		v: v,
	})
	if t.After(s.t1) {
		values := s.values.Values()
		t0 := s.t0
		s.values.Clear()
		s.t0 = t
		s.t1 = s.t0.Add(rescaleThreshold)
		for _, v := range values {
			v.k = v.k * math.Exp(-s.alpha*s.t0.Sub(t0).Seconds())
			s.values.Push(v)
		}
	}
}

func (s *ExpDecaySample) updateSharded(t time.Time, v int64, generation int64) {
	if atomic.LoadInt32(&s.clearing) != 0 {
		return
	}
	epoch := atomic.LoadInt64(&s.epoch)
	shard := s.shards[nextShardIndex(&s.pickerPool, &s.blockSeq, len(s.shards))]
	shard.mutex.Lock()
	if shard.epoch != epoch || atomic.LoadInt32(&s.clearing) != 0 || atomic.LoadInt64(&s.generation) != generation {
		shard.mutex.Unlock()
		return
	}
	shard.count++
	shard.addLocked(t, v, s.alpha)
	shard.mutex.Unlock()
}

// escalate converts the sample to a fixed set of independent reservoirs.  It
// runs once, after two writers collide on the single-reservoir fast path.  The
// existing reservoir is split across the shards so memory remains bounded by
// the configured reservoir size.
func (s *ExpDecaySample) escalate() {
	s.escalateMutex.Lock()
	defer s.escalateMutex.Unlock()
	if atomic.LoadInt32(&s.sharded) != 0 {
		return
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()

	n := shardCountFor(s.reservoirSize)
	epoch := atomic.LoadInt64(&s.epoch) + 1
	legacy := s.values.Values()
	shards := make([]*expDecaySampleShard, n)
	for i := range shards {
		shards[i] = &expDecaySampleShard{
			epoch:         epoch,
			reservoirSize: shardCapacity(s.reservoirSize, n, i),
			t0:            s.t0,
			t1:            s.t1,
			values:        newExpDecaySampleHeap(shardCapacity(s.reservoirSize, n, i)),
			rng:           shardRand(mix64(uint64(i)+1) ^ uint64(time.Now().UnixNano())),
		}
	}

	seq := uint64(0)
	for _, value := range legacy {
		shard := shards[shardIndex(n, seq)]
		seq++
		shard.mutex.Lock()
		if shard.values.Size() == shard.reservoirSize {
			shard.values.Pop()
		}
		shard.values.Push(value)
		shard.mutex.Unlock()
	}

	s.values.Clear()
	s.shards = shards
	atomic.StoreUint64(&s.blockSeq, seq)
	for i := 0; i < n; i++ {
		s.pickerPool.Put(new(shardPicker))
	}
	atomic.StoreInt64(&s.epoch, epoch)
	atomic.StoreInt32(&s.sharded, 1)
}

func (s *ExpDecaySample) snapshot() (int64, []int64) {
	for {
		generation := atomic.LoadInt64(&s.generation)
		if atomic.LoadInt32(&s.clearing) != 0 {
			runtime.Gosched()
			continue
		}
		epoch := atomic.LoadInt64(&s.epoch)

		s.mutex.Lock()
		values := make([]int64, 0, s.values.Size()+s.reservoirSize)
		for _, value := range s.values.Values() {
			values = append(values, value.v)
		}
		count := s.count
		s.mutex.Unlock()

		if atomic.LoadInt32(&s.sharded) != 0 {
			for _, shard := range s.shards {
				shard.mutex.Lock()
				if shard.epoch == epoch {
					count += shard.count
					for _, value := range shard.values.Values() {
						values = append(values, value.v)
					}
				}
				shard.mutex.Unlock()
			}
		}

		if atomic.LoadInt32(&s.clearing) == 0 &&
			atomic.LoadInt64(&s.generation) == generation &&
			atomic.LoadInt64(&s.epoch) == epoch {
			return count, values
		}
	}
}

type expDecaySampleShard struct {
	mutex         sync.Mutex
	epoch         int64
	count         int64
	reservoirSize int
	t0, t1        time.Time
	values        *expDecaySampleHeap
	rng           shardRand
}

func (s *expDecaySampleShard) addLocked(t time.Time, v int64, alpha float64) {
	if s.values.Size() == s.reservoirSize {
		s.values.Pop()
	}
	s.values.Push(expDecaySample{
		k: math.Exp(t.Sub(s.t0).Seconds()*alpha) / s.rng.float64(),
		v: v,
	})
	if t.After(s.t1) {
		values := s.values.Values()
		t0 := s.t0
		s.values.Clear()
		s.t0 = t
		s.t1 = s.t0.Add(rescaleThreshold)
		for _, v := range values {
			v.k = v.k * math.Exp(-alpha*s.t0.Sub(t0).Seconds())
			s.values.Push(v)
		}
	}
}

// NilSample is a no-op Sample.
type NilSample struct{}

// Clear is a no-op.
func (NilSample) Clear() {}

// Count is a no-op.
func (NilSample) Count() int64 { return 0 }

// Max is a no-op.
func (NilSample) Max() int64 { return 0 }

// Mean is a no-op.
func (NilSample) Mean() float64 { return 0.0 }

// Min is a no-op.
func (NilSample) Min() int64 { return 0 }

// Percentile is a no-op.
func (NilSample) Percentile(p float64) float64 { return 0.0 }

// Percentiles is a no-op.
func (NilSample) Percentiles(ps []float64) []float64 {
	return make([]float64, len(ps))
}

// Size is a no-op.
func (NilSample) Size() int { return 0 }

// Sample is a no-op.
func (NilSample) Snapshot() Sample { return NilSample{} }

// StdDev is a no-op.
func (NilSample) StdDev() float64 { return 0.0 }

// Sum is a no-op.
func (NilSample) Sum() int64 { return 0 }

// Update is a no-op.
func (NilSample) Update(v int64) {}

// Values is a no-op.
func (NilSample) Values() []int64 { return []int64{} }

// Variance is a no-op.
func (NilSample) Variance() float64 { return 0.0 }

// SampleMax returns the maximum value of the slice of int64.
func SampleMax(values []int64) int64 {
	if 0 == len(values) {
		return 0
	}
	var max int64 = math.MinInt64
	for _, v := range values {
		if max < v {
			max = v
		}
	}
	return max
}

// SampleMean returns the mean value of the slice of int64.
func SampleMean(values []int64) float64 {
	if 0 == len(values) {
		return 0.0
	}
	return float64(SampleSum(values)) / float64(len(values))
}

// SampleMin returns the minimum value of the slice of int64.
func SampleMin(values []int64) int64 {
	if 0 == len(values) {
		return 0
	}
	var min int64 = math.MaxInt64
	for _, v := range values {
		if min > v {
			min = v
		}
	}
	return min
}

// SamplePercentiles returns an arbitrary percentile of the slice of int64.
func SamplePercentile(values int64Slice, p float64) float64 {
	return SamplePercentiles(values, []float64{p})[0]
}

// SamplePercentiles returns a slice of arbitrary percentiles of the slice of
// int64.
func SamplePercentiles(values int64Slice, ps []float64) []float64 {
	scores := make([]float64, len(ps))
	size := len(values)
	if size > 0 {
		sort.Sort(values)
		for i, p := range ps {
			pos := p * float64(size+1)
			if pos < 1.0 {
				scores[i] = float64(values[0])
			} else if pos >= float64(size) {
				scores[i] = float64(values[size-1])
			} else {
				lower := float64(values[int(pos)-1])
				upper := float64(values[int(pos)])
				scores[i] = lower + (pos-math.Floor(pos))*(upper-lower)
			}
		}
	}
	return scores
}

// SampleSnapshot is a read-only copy of another Sample.
type SampleSnapshot struct {
	count  int64
	values []int64
}

func NewSampleSnapshot(count int64, values []int64) *SampleSnapshot {
	return &SampleSnapshot{
		count:  count,
		values: values,
	}
}

// Clear panics.
func (*SampleSnapshot) Clear() {
	panic("Clear called on a SampleSnapshot")
}

// Count returns the count of inputs at the time the snapshot was taken.
func (s *SampleSnapshot) Count() int64 { return s.count }

// Max returns the maximal value at the time the snapshot was taken.
func (s *SampleSnapshot) Max() int64 { return SampleMax(s.values) }

// Mean returns the mean value at the time the snapshot was taken.
func (s *SampleSnapshot) Mean() float64 { return SampleMean(s.values) }

// Min returns the minimal value at the time the snapshot was taken.
func (s *SampleSnapshot) Min() int64 { return SampleMin(s.values) }

// Percentile returns an arbitrary percentile of values at the time the
// snapshot was taken.
func (s *SampleSnapshot) Percentile(p float64) float64 {
	return SamplePercentile(s.values, p)
}

// Percentiles returns a slice of arbitrary percentiles of values at the time
// the snapshot was taken.
func (s *SampleSnapshot) Percentiles(ps []float64) []float64 {
	return SamplePercentiles(s.values, ps)
}

// Size returns the size of the sample at the time the snapshot was taken.
func (s *SampleSnapshot) Size() int { return len(s.values) }

// Snapshot returns the snapshot.
func (s *SampleSnapshot) Snapshot() Sample { return s }

// StdDev returns the standard deviation of values at the time the snapshot was
// taken.
func (s *SampleSnapshot) StdDev() float64 { return SampleStdDev(s.values) }

// Sum returns the sum of values at the time the snapshot was taken.
func (s *SampleSnapshot) Sum() int64 { return SampleSum(s.values) }

// Update panics.
func (*SampleSnapshot) Update(int64) {
	panic("Update called on a SampleSnapshot")
}

// Values returns a copy of the values in the sample.
func (s *SampleSnapshot) Values() []int64 {
	values := make([]int64, len(s.values))
	copy(values, s.values)
	return values
}

// Variance returns the variance of values at the time the snapshot was taken.
func (s *SampleSnapshot) Variance() float64 { return SampleVariance(s.values) }

// SampleStdDev returns the standard deviation of the slice of int64.
func SampleStdDev(values []int64) float64 {
	return math.Sqrt(SampleVariance(values))
}

// SampleSum returns the sum of the slice of int64.
func SampleSum(values []int64) int64 {
	var sum int64
	for _, v := range values {
		sum += v
	}
	return sum
}

// SampleVariance returns the variance of the slice of int64.
func SampleVariance(values []int64) float64 {
	if 0 == len(values) {
		return 0.0
	}
	m := SampleMean(values)
	var sum float64
	for _, v := range values {
		d := float64(v) - m
		sum += d * d
	}
	return sum / float64(len(values))
}

// A uniform sample using Vitter's Algorithm R.
//
// <http://www.cs.umd.edu/~samir/498/vitter.pdf>
//
// Like ExpDecaySample, concurrent writers are moved onto a fixed set of
// bounded shards and readers merge those shards.
type UniformSample struct {
	count         int64
	mutex         sync.Mutex
	reservoirSize int
	values        []int64

	escalateMutex sync.Mutex
	epoch         int64
	generation    int64
	blockSeq      uint64
	sharded       int32
	clearing      int32
	shards        []*uniformSampleShard
	pickerPool    sync.Pool
}

// NewUniformSample constructs a new uniform sample with the given reservoir
// size.
func NewUniformSample(reservoirSize int) Sample {
	if UseNilMetrics {
		return NilSample{}
	}
	return &UniformSample{
		reservoirSize: reservoirSize,
		values:        make([]int64, 0, reservoirSize),
	}
}

// Clear clears all samples.
func (s *UniformSample) Clear() {
	s.escalateMutex.Lock()
	defer s.escalateMutex.Unlock()

	atomic.StoreInt32(&s.clearing, 1)
	atomic.AddInt64(&s.generation, 1)
	epoch := atomic.AddInt64(&s.epoch, 1)

	s.mutex.Lock()
	s.count = 0
	s.values = make([]int64, 0, s.reservoirSize)
	s.mutex.Unlock()

	for _, shard := range s.shards {
		shard.mutex.Lock()
		shard.epoch = epoch
		shard.count = 0
		shard.seen = 0
		shard.values = shard.values[:0]
		shard.mutex.Unlock()
	}
	atomic.StoreInt32(&s.clearing, 0)
}

// Count returns the number of samples recorded, which may exceed the
// reservoir size.
func (s *UniformSample) Count() int64 {
	count, _ := s.snapshot()
	return count
}

// Max returns the maximum value in the sample, which may not be the maximum
// value ever to be part of the sample.
func (s *UniformSample) Max() int64 {
	return SampleMax(s.Values())
}

// Mean returns the mean of the values in the sample.
func (s *UniformSample) Mean() float64 {
	return SampleMean(s.Values())
}

// Min returns the minimum value in the sample, which may not be the minimum
// value ever to be part of the sample.
func (s *UniformSample) Min() int64 {
	return SampleMin(s.Values())
}

// Percentile returns an arbitrary percentile of values in the sample.
func (s *UniformSample) Percentile(p float64) float64 {
	return SamplePercentile(s.Values(), p)
}

// Percentiles returns a slice of arbitrary percentiles of values in the
// sample.
func (s *UniformSample) Percentiles(ps []float64) []float64 {
	return SamplePercentiles(s.Values(), ps)
}

// Size returns the size of the sample, which is at most the reservoir size.
func (s *UniformSample) Size() int {
	_, values := s.snapshot()
	return len(values)
}

// Snapshot returns a read-only copy of the sample.
func (s *UniformSample) Snapshot() Sample {
	count, values := s.snapshot()
	return &SampleSnapshot{
		count:  count,
		values: values,
	}
}

// StdDev returns the standard deviation of the values in the sample.
func (s *UniformSample) StdDev() float64 {
	return SampleStdDev(s.Values())
}

// Sum returns the sum of the values in the sample.
func (s *UniformSample) Sum() int64 {
	return SampleSum(s.Values())
}

// Update samples a new value.
func (s *UniformSample) Update(v int64) {
	generation := atomic.LoadInt64(&s.generation)
	if atomic.LoadInt32(&s.sharded) != 0 {
		s.updateSharded(v, generation)
		return
	}
	if s.mutex.TryLock() {
		if atomic.LoadInt32(&s.sharded) != 0 {
			s.mutex.Unlock()
			s.updateSharded(v, generation)
			return
		}
		if atomic.LoadInt64(&s.generation) != generation || atomic.LoadInt32(&s.clearing) != 0 {
			s.mutex.Unlock()
			return
		}
		s.updateLocked(v)
		s.mutex.Unlock()
		return
	}

	s.escalate()
	s.updateSharded(v, generation)
}

func (s *UniformSample) updateLocked(v int64) {
	s.count++
	if len(s.values) < s.reservoirSize {
		s.values = append(s.values, v)
	} else {
		r := rand.Int63n(s.count)
		if r < int64(len(s.values)) {
			s.values[int(r)] = v
		}
	}
}

func (s *UniformSample) updateSharded(v int64, generation int64) {
	if atomic.LoadInt32(&s.clearing) != 0 {
		return
	}
	epoch := atomic.LoadInt64(&s.epoch)
	shard := s.shards[nextShardIndex(&s.pickerPool, &s.blockSeq, len(s.shards))]
	shard.mutex.Lock()
	if shard.epoch != epoch || atomic.LoadInt32(&s.clearing) != 0 || atomic.LoadInt64(&s.generation) != generation {
		shard.mutex.Unlock()
		return
	}
	shard.addLocked(v, true)
	shard.mutex.Unlock()
}

func (s *UniformSample) escalate() {
	s.escalateMutex.Lock()
	defer s.escalateMutex.Unlock()
	if atomic.LoadInt32(&s.sharded) != 0 {
		return
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()

	n := shardCountFor(s.reservoirSize)
	epoch := atomic.LoadInt64(&s.epoch) + 1
	shards := make([]*uniformSampleShard, n)
	for i := range shards {
		shards[i] = &uniformSampleShard{
			epoch:         epoch,
			reservoirSize: shardCapacity(s.reservoirSize, n, i),
			values:        make([]int64, 0, shardCapacity(s.reservoirSize, n, i)),
			rng:           shardRand(mix64(uint64(i)+1) ^ uint64(time.Now().UnixNano())),
		}
	}

	seq := uint64(0)
	for _, value := range s.values {
		shard := shards[shardIndex(n, seq)]
		seq++
		shard.mutex.Lock()
		shard.addLocked(value, false)
		shard.mutex.Unlock()
	}
	oldCount := s.count
	for i, shard := range shards {
		weighted := oldCount / int64(n)
		if int64(i) < oldCount%int64(n) {
			weighted++
		}
		if shard.seen < weighted {
			shard.seen = weighted
		}
	}

	s.values = make([]int64, 0, s.reservoirSize)
	s.shards = shards
	atomic.StoreUint64(&s.blockSeq, seq)
	for i := 0; i < n; i++ {
		s.pickerPool.Put(new(shardPicker))
	}
	atomic.StoreInt64(&s.epoch, epoch)
	atomic.StoreInt32(&s.sharded, 1)
}

func (s *UniformSample) snapshot() (int64, []int64) {
	for {
		generation := atomic.LoadInt64(&s.generation)
		if atomic.LoadInt32(&s.clearing) != 0 {
			runtime.Gosched()
			continue
		}
		epoch := atomic.LoadInt64(&s.epoch)

		s.mutex.Lock()
		values := make([]int64, 0, len(s.values)+s.reservoirSize)
		values = append(values, s.values...)
		count := s.count
		s.mutex.Unlock()

		if atomic.LoadInt32(&s.sharded) != 0 {
			for _, shard := range s.shards {
				shard.mutex.Lock()
				if shard.epoch == epoch {
					count += shard.count
					values = append(values, shard.values...)
				}
				shard.mutex.Unlock()
			}
		}

		if atomic.LoadInt32(&s.clearing) == 0 &&
			atomic.LoadInt64(&s.generation) == generation &&
			atomic.LoadInt64(&s.epoch) == epoch {
			return count, values
		}
	}
}

type uniformSampleShard struct {
	mutex         sync.Mutex
	epoch         int64
	count         int64
	seen          int64
	reservoirSize int
	values        []int64
	rng           shardRand
}

func (s *uniformSampleShard) addLocked(v int64, record bool) {
	s.seen++
	if record {
		s.count++
	}
	if len(s.values) < s.reservoirSize {
		s.values = append(s.values, v)
	} else {
		r := s.rng.int63n(s.seen)
		if r < int64(len(s.values)) {
			s.values[int(r)] = v
		}
	}
}

// Values returns a copy of the values in the sample.
func (s *UniformSample) Values() []int64 {
	_, values := s.snapshot()
	return values
}

// Variance returns the variance of the values in the sample.
func (s *UniformSample) Variance() float64 {
	return SampleVariance(s.Values())
}

// shardCountFor bounds the number of write shards by both the reservoir size
// and the amount of parallelism available.  A power of two keeps shard
// selection to an add and a mask.
func shardCountFor(reservoirSize int) int {
	n := 32
	if p := runtime.GOMAXPROCS(0) * 2; p > n {
		n = p
	}
	if n > 256 {
		n = 256
	}
	if reservoirSize < n {
		n = reservoirSize
	}
	if n < 1 {
		return 1
	}
	power := 1
	for power*2 <= n {
		power *= 2
	}
	return power
}

func shardCapacity(reservoirSize, shards, i int) int {
	capacity := reservoirSize / shards
	if i < reservoirSize%shards {
		capacity++
	}
	if capacity < 1 {
		capacity = 1
	}
	return capacity
}

// shardIndex assigns consecutive updates in blocks of n.  Every block visits
// each shard exactly once, but the block's phase is randomized so a periodic
// input pattern cannot repeatedly hit only one shard.
func shardIndex(n int, seq uint64) int {
	if n <= 1 {
		return 0
	}
	offset := mix64(seq / uint64(n))
	return int((seq + offset) & uint64(n-1))
}

type shardPicker struct {
	next      uint64
	remaining int
}

func nextShardIndex(pool *sync.Pool, blockSeq *uint64, n int) int {
	picker, _ := pool.Get().(*shardPicker)
	if picker == nil {
		return shardIndex(n, mix64(uint64(time.Now().UnixNano())))
	}
	if picker.remaining == 0 {
		picker.next = atomic.AddUint64(blockSeq, uint64(n)) - uint64(n)
		picker.remaining = n
	}
	seq := picker.next
	picker.next++
	picker.remaining--
	index := shardIndex(n, seq)
	pool.Put(picker)
	return index
}

func mix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

type shardRand uint64

func (r *shardRand) uint64() uint64 {
	*r += 0x9e3779b97f4a7c15
	return mix64(uint64(*r))
}

func (r *shardRand) float64() float64 {
	return float64((r.uint64()>>11)+1) / float64(uint64(1)<<53)
}

func (r *shardRand) int63n(n int64) int64 {
	return int64(r.uint64() % uint64(n))
}

// expDecaySample represents an individual sample in a heap.
type expDecaySample struct {
	k float64
	v int64
}

func newExpDecaySampleHeap(reservoirSize int) *expDecaySampleHeap {
	return &expDecaySampleHeap{make([]expDecaySample, 0, reservoirSize)}
}

// expDecaySampleHeap is a min-heap of expDecaySamples.
// The internal implementation is copied from the standard library's container/heap
type expDecaySampleHeap struct {
	s []expDecaySample
}

func (h *expDecaySampleHeap) Clear() {
	h.s = h.s[:0]
}

func (h *expDecaySampleHeap) Push(s expDecaySample) {
	n := len(h.s)
	h.s = h.s[0 : n+1]
	h.s[n] = s
	h.up(n)
}

func (h *expDecaySampleHeap) Pop() expDecaySample {
	n := len(h.s) - 1
	h.s[0], h.s[n] = h.s[n], h.s[0]
	h.down(0, n)

	n = len(h.s)
	s := h.s[n-1]
	h.s = h.s[0 : n-1]
	return s
}

func (h *expDecaySampleHeap) Size() int {
	return len(h.s)
}

func (h *expDecaySampleHeap) Values() []expDecaySample {
	return h.s
}

func (h *expDecaySampleHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !(h.s[j].k < h.s[i].k) {
			break
		}
		h.s[i], h.s[j] = h.s[j], h.s[i]
		j = i
	}
}

func (h *expDecaySampleHeap) down(i, n int) {
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && !(h.s[j1].k < h.s[j2].k) {
			j = j2 // = 2*i + 2  // right child
		}
		if !(h.s[j].k < h.s[i].k) {
			break
		}
		h.s[i], h.s[j] = h.s[j], h.s[i]
		i = j
	}
}

type int64Slice []int64

func (p int64Slice) Len() int           { return len(p) }
func (p int64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p int64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
