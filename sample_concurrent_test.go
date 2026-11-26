package metrics

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

func forceSharded(s Sample) {
	switch sample := s.(type) {
	case *UniformSample:
		sample.escalate()
	case *ExpDecaySample:
		sample.escalate()
	}
}

func TestSampleConcurrentUpdates(t *testing.T) {
	const workers = 8
	const perWorker = 4000
	for _, s := range []Sample{
		NewUniformSample(1028),
		NewExpDecaySample(1028, 0.015),
	} {
		var wg sync.WaitGroup
		start := make(chan struct{})
		for worker := 0; worker < workers; worker++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				<-start
				for i := 0; i < perWorker; i++ {
					s.Update(int64(worker*perWorker + i))
				}
			}(worker)
		}
		close(start)
		wg.Wait()

		if got, want := s.Count(), int64(workers*perWorker); got != want {
			t.Errorf("%T Count(): %d != %d", s, got, want)
		}
		if got := s.Size(); got > 1028 {
			t.Errorf("%T Size(): %d > 1028", s, got)
		}
		if got, want := len(s.Values()), s.Size(); got != want {
			t.Errorf("%T len(Values()): %d != %d", s, got, want)
		}
	}
}

func TestSampleClearIsolation(t *testing.T) {
	for _, s := range []Sample{
		NewUniformSample(1028),
		NewExpDecaySample(1028, 0.015),
	} {
		forceSharded(s)
		for i := 0; i < 10000; i++ {
			s.Update(1 << 40)
		}

		s.Clear()
		if got := s.Count(); got != 0 {
			t.Errorf("%T Count after Clear: %d != 0", s, got)
		}
		if got := s.Size(); got != 0 {
			t.Errorf("%T Size after Clear: %d != 0", s, got)
		}
		if got := len(s.Values()); got != 0 {
			t.Errorf("%T len(Values()) after Clear: %d != 0", s, got)
		}

		for i := 0; i < 10000; i++ {
			s.Update(1)
		}
		if got := s.Max(); got != 1 {
			t.Errorf("%T old values leaked after Clear: Max()=%d", s, got)
		}
		if got := s.Percentile(0.99); got != 1 {
			t.Errorf("%T old values leaked after Clear: p99=%v", s, got)
		}
		if got := s.Count(); got != 10000 {
			t.Errorf("%T Count after reset: %d != 10000", s, got)
		}
	}
}

func TestSampleShardMemoryIsBounded(t *testing.T) {
	for _, s := range []Sample{
		NewUniformSample(1028),
		NewExpDecaySample(1028, 0.015),
	} {
		forceSharded(s)
		shardCount := 0
		capacity := 0
		switch sample := s.(type) {
		case *UniformSample:
			shardCount = len(sample.shards)
			for _, shard := range sample.shards {
				capacity += cap(shard.values)
			}
		case *ExpDecaySample:
			shardCount = len(sample.shards)
			for _, shard := range sample.shards {
				capacity += shard.reservoirSize
			}
		}
		if capacity > 1028 {
			t.Errorf("%T shard capacity: %d > 1028", s, capacity)
		}

		var wg sync.WaitGroup
		start := make(chan struct{})
		for worker := 0; worker < 256; worker++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				<-start
				for i := 0; i < 100; i++ {
					s.Update(int64(worker*100 + i))
				}
			}(worker)
		}
		close(start)
		wg.Wait()

		switch sample := s.(type) {
		case *UniformSample:
			if len(sample.shards) != shardCount {
				t.Errorf("%T shard count grew with goroutines: %d != %d", s, len(sample.shards), shardCount)
			}
		case *ExpDecaySample:
			if len(sample.shards) != shardCount {
				t.Errorf("%T shard count grew with goroutines: %d != %d", s, len(sample.shards), shardCount)
			}
		}
	}
}

func TestSampleShardedPercentileAccuracy(t *testing.T) {
	const n = 100000
	const reservoir = 4096
	values := make([]int64, n)
	rng := rand.New(rand.NewSource(1))
	for i := range values {
		values[i] = int64(rng.ExpFloat64() * 10)
		if i%1000 == 0 {
			values[i] = 1000000
		}
	}
	exact := append([]int64(nil), values...)
	truth := SamplePercentile(exact, 0.99)
	if truth <= 0 {
		t.Fatal("bad reference percentile")
	}

	for _, s := range []Sample{
		NewUniformSample(reservoir),
		NewExpDecaySample(reservoir, 0.015),
	} {
		forceSharded(s)
		for _, value := range values {
			s.Update(value)
		}
		got := s.Percentile(0.99)
		if relativeError := math.Abs(got-truth) / truth; relativeError > 0.12 {
			t.Errorf("%T p99: got %v, want %v (relative error %.3f)", s, got, truth, relativeError)
		}
	}
}

func BenchmarkUniformSample1028Parallel(b *testing.B) {
	s := NewUniformSample(1028)
	b.RunParallel(func(pb *testing.PB) {
		value := int64(0)
		for pb.Next() {
			s.Update(value)
			value++
		}
	})
}

func BenchmarkExpDecaySample1028Parallel(b *testing.B) {
	s := NewExpDecaySample(1028, 0.015)
	b.RunParallel(func(pb *testing.PB) {
		value := int64(0)
		for pb.Next() {
			s.Update(value)
			value++
		}
	})
}
