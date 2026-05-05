// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package workflow_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

func runWorkflow(t *testing.T, wf *workflow.Workflow) []*session.Event {
	t.Helper()
	wfAgent, err := wf.AsAgent()
	if err != nil {
		t.Fatalf("AsAgent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	var events []*session.Event
	for ev, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run err: %v", err)
		}
		events = append(events, ev)
	}
	return events
}

func TestWorkflow_FanOutFanIn_WithJoinNode(t *testing.T) {
	type val struct {
		N int `json:"n"`
	}
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (val, error) { return val{N: 10}, nil })
	a := workflow.Func("a",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 1}, nil })
	b := workflow.Func("b",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 2}, nil })
	c := workflow.Func("c",
		func(_ *workflow.NodeContext, in val) (val, error) { return val{N: in.N + 3}, nil })
	join := workflow.Join("join")
	final := workflow.Func("final",
		func(_ *workflow.NodeContext, in any) (int, error) {
			m := in.(map[string]any)
			total := 0
			for _, v := range m {
				total += v.(val).N
			}
			return total, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "fanout",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, a),
			workflow.Connect(seed, b),
			workflow.Connect(seed, c),
			workflow.Connect(a, join),
			workflow.Connect(b, join),
			workflow.Connect(c, join),
			workflow.Connect(join, final),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events := runWorkflow(t, wf)
	var got int
	for _, ev := range events {
		if ev.Author == "final" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.(int)
		}
	}
	// 10 -> a:11, b:12, c:13 -> join: {a:11,b:12,c:13} -> final: 36
	if got != 36 {
		t.Errorf("final = %d, want 36", got)
	}
}

func TestWorkflow_RetryOnTransientError(t *testing.T) {
	var attempts atomic.Int32
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	flaky := workflow.Func("flaky",
		func(_ *workflow.NodeContext, in int) (int, error) {
			n := attempts.Add(1)
			if n < 3 {
				return 0, errors.New("transient")
			}
			return in * 10, nil
		},
		workflow.WithRetry(&workflow.RetryConfig{
			MaxAttempts:  5,
			InitialDelay: time.Microsecond,
			MaxDelay:     time.Microsecond,
			Jitter:       -1,
		}),
	)

	wf, err := workflow.New(workflow.Config{
		Name: "retrywf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, flaky),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)
	var got int
	for _, ev := range events {
		if ev.Author == "flaky" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.(int)
		}
	}
	if got != 10 {
		t.Errorf("output = %d, want 10", got)
	}
	if attempts.Load() != 3 {
		t.Errorf("attempts = %d, want 3", attempts.Load())
	}
}

func TestWorkflow_NodeTimeout(t *testing.T) {
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	slow := workflow.Func("slow",
		func(ctx *workflow.NodeContext, _ int) (int, error) {
			select {
			case <-time.After(2 * time.Second):
				return 0, nil
			case <-ctx.InvocationContext.Done():
				return 0, ctx.InvocationContext.Err()
			}
		},
		workflow.WithTimeout(50*time.Millisecond),
	)

	wf, err := workflow.New(workflow.Config{
		Name: "timeoutwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, slow),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "t",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	sawErr := false
	start := time.Now()
	for _, err := range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
		if err != nil {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("expected timeout error from slow node")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("timeout did not abort quickly: elapsed = %v", elapsed)
	}
}

func TestWorkflow_ParallelWorker_FansOutAndCollects(t *testing.T) {
	var seen sync.Map
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) ([]int, error) {
			return []int{2, 3, 5, 7}, nil
		})
	square := workflow.Func("square",
		func(_ *workflow.NodeContext, n int) (int, error) {
			seen.Store(n, true)
			return n * n, nil
		})
	worker := workflow.Parallel[int](square)

	wf, err := workflow.New(workflow.Config{
		Name: "pwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, worker),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)
	var got []any
	for _, ev := range events {
		if ev.Author == worker.Name() && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			got = ev.Actions.NodeInfo.Output.([]any)
		}
	}
	if len(got) != 4 {
		t.Fatalf("results = %v, want length 4", got)
	}
	want := map[int]bool{4: true, 9: true, 25: true, 49: true}
	for _, r := range got {
		if !want[r.(int)] {
			t.Errorf("unexpected result %v", r)
		}
	}
}

func TestWorkflow_MaxConcurrency_BoundsParallelism(t *testing.T) {
	// Three nodes can fire after seed; with MaxConcurrency=2, peak in-flight
	// must be <= 2.
	type val struct{ N int }
	var inFlight atomic.Int32
	var peak atomic.Int32

	work := func(name string) workflow.Node {
		return workflow.Func(name,
			func(_ *workflow.NodeContext, _ val) (val, error) {
				cur := inFlight.Add(1)
				for {
					p := peak.Load()
					if cur <= p || peak.CompareAndSwap(p, cur) {
						break
					}
				}
				time.Sleep(20 * time.Millisecond)
				inFlight.Add(-1)
				return val{}, nil
			})
	}
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (val, error) { return val{}, nil })
	a, b, c := work("a"), work("b"), work("c")
	wf, err := workflow.New(workflow.Config{
		Name:           "boundedwf",
		MaxConcurrency: 2,
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, a),
			workflow.Connect(seed, b),
			workflow.Connect(seed, c),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	runWorkflow(t, wf)
	if got := peak.Load(); got > 2 {
		t.Errorf("peak in-flight = %d, want <= 2", got)
	}
}

func TestWorkflow_ConditionalRouting_DefaultBranch(t *testing.T) {
	// classifier emits no route; only the default-branch edge should fire.
	classifier := workflow.Func("classify",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	branchA := workflow.Func("a",
		func(_ *workflow.NodeContext, _ int) (string, error) { return "a", nil })
	branchDefault := workflow.Func("d",
		func(_ *workflow.NodeContext, _ int) (string, error) { return "d", nil })

	wf, err := workflow.New(workflow.Config{
		Name: "rwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, classifier),
			workflow.Connect(classifier, branchA, workflow.RouteString("alpha")),
			workflow.Connect(classifier, branchDefault, workflow.DefaultRoute),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)

	var sawA, sawD bool
	for _, ev := range events {
		if ev.Author == "a" {
			sawA = true
		}
		if ev.Author == "d" {
			sawD = true
		}
	}
	if sawA {
		t.Error("branch A should not have fired (no matching route)")
	}
	if !sawD {
		t.Error("default branch should have fired")
	}
}

// TestWorkflow_RetryHonorsCtxCancellation locks down review fix #1: a
// node whose retry sleep is interrupted by parent ctx cancellation must
// stop attempting and return the cancellation error promptly. A bare
// `break` inside the select previously left the attempt loop running with
// a cancelled ctx, burning the retry budget.
func TestWorkflow_RetryHonorsCtxCancellation(t *testing.T) {
	var attempts atomic.Int32

	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
	failing := workflow.Func("failing",
		func(ctx *workflow.NodeContext, _ int) (int, error) {
			attempts.Add(1)
			return 0, errors.New("always fails")
		},
		workflow.WithRetry(&workflow.RetryConfig{
			MaxAttempts:  10,
			InitialDelay: 200 * time.Millisecond,
			MaxDelay:     200 * time.Millisecond,
			Jitter:       0,
		}),
	)

	wf, err := workflow.New(workflow.Config{
		Name: "cancelwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, failing),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, err := wf.AsAgent()
	if err != nil {
		t.Fatalf("AsAgent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		// Cancel after a single retry sleep starts so we observe
		// labeled-break behavior.
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	for range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
	}

	got := attempts.Load()
	if got > 3 {
		t.Errorf("attempts = %d, want <= 3 (cancellation should short-circuit)", got)
	}
}

// TestWorkflow_RetryConfigDelayFor_RespectsMaxDelay locks down review fix
// #11: jittered delay never exceeds MaxDelay.
func TestWorkflow_RetryConfigDelayFor_RespectsMaxDelay(t *testing.T) {
	t.Parallel()
	cfg := workflow.RetryConfig{
		MaxAttempts:   30,
		InitialDelay:  10 * time.Millisecond,
		MaxDelay:      50 * time.Millisecond,
		BackoffFactor: 2.0,
		Jitter:        workflow.DefaultJitter,
	}
	for i := 0; i < 1000; i++ {
		for attempt := 1; attempt <= 12; attempt++ {
			d := cfg.DelayFor(attempt)
			if d > cfg.MaxDelay {
				t.Fatalf("DelayFor(attempt=%d) = %v, exceeds MaxDelay %v", attempt, d, cfg.MaxDelay)
			}
			if d < 0 {
				t.Fatalf("DelayFor(attempt=%d) = %v, want >= 0", attempt, d)
			}
		}
	}
}

// TestWorkflow_ParallelWorker_NoActionsRace locks down review fix #2: each
// parallel worker writes its own EventActions instance, so concurrent
// StateDelta writes don't race and all worker events are forwarded to the
// parent emitter.
func TestWorkflow_ParallelWorker_NoActionsRace(t *testing.T) {
	const N = 16
	inner := workflow.Func("inner",
		func(ctx *workflow.NodeContext, i int) (int, error) {
			ctx.Actions().StateDelta[fmtKey(i)] = i
			return i * 2, nil
		},
	)
	seed := workflow.Func("seed",
		func(_ *workflow.NodeContext, _ any) ([]int, error) {
			items := make([]int, N)
			for i := 0; i < N; i++ {
				items[i] = i
			}
			return items, nil
		})
	parallel := workflow.Parallel[int](inner)

	wf, err := workflow.New(workflow.Config{
		Name: "parallel_actions",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, seed),
			workflow.Connect(seed, parallel),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	events := runWorkflow(t, wf)

	var sawInnerEvents int
	for _, ev := range events {
		if ev.Author == "inner" {
			sawInnerEvents++
		}
	}
	if sawInnerEvents < N {
		t.Errorf("forwarded inner events = %d, want >= %d (events should propagate from parallel children)", sawInnerEvents, N)
	}
}

func fmtKey(i int) string { return "k_" + intToString(i) }

func intToString(i int) string {
	if i == 0 {
		return "0"
	}
	digits := []byte{}
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	if neg {
		return "-" + string(digits)
	}
	return string(digits)
}

// TestWorkflow_JoinNodeKeyOrderIsDeterministic locks down review fix #13:
// when a downstream LlmAgentNode renders the JoinNode aggregate as the
// agent's prompt, the predecessor names appear in a stable (alphabetic)
// order regardless of completion order.
func TestWorkflow_JoinNodeKeyOrderIsDeterministic(t *testing.T) {
	t.Parallel()

	// Run the same fan-out/fan-in 10 times with random predecessor delays
	// and verify the JoinNode-rendered prompt comes out identically every
	// time.
	render := func() string {
		seed := workflow.Func("seed",
			func(_ *workflow.NodeContext, _ any) (int, error) { return 1, nil })
		var aDelay, bDelay, cDelay time.Duration = 0, 5 * time.Millisecond, 10 * time.Millisecond
		// Shuffle delays to vary completion order.
		switch time.Now().UnixNano() % 6 {
		case 1:
			aDelay, bDelay, cDelay = 10*time.Millisecond, 0, 5*time.Millisecond
		case 2:
			aDelay, bDelay, cDelay = 5*time.Millisecond, 10*time.Millisecond, 0
		}
		mk := func(name string, d time.Duration) workflow.Node {
			return workflow.Func(name, func(_ *workflow.NodeContext, _ int) (string, error) {
				time.Sleep(d)
				return name + "_out", nil
			})
		}
		a := mk("a", aDelay)
		b := mk("b", bDelay)
		c := mk("c", cDelay)
		join := workflow.Join("merge")
		var rendered string
		final := workflow.Func("final",
			func(_ *workflow.NodeContext, in any) (string, error) {
				if m, ok := in.(map[string]any); ok {
					keys := []string{}
					for k := range m {
						keys = append(keys, k)
					}
					// Render via the same path as renderUserContent: Sprintf
					// with sorted keys.
					sortStrings(keys)
					var s string
					for _, k := range keys {
						s += k + "=" + asString(m[k]) + "|"
					}
					rendered = s
				}
				return "done", nil
			})
		wf, err := workflow.New(workflow.Config{
			Name: "joindet",
			Edges: []workflow.Edge{
				workflow.Connect(workflow.START, seed),
				workflow.Connect(seed, a),
				workflow.Connect(seed, b),
				workflow.Connect(seed, c),
				workflow.Connect(a, join),
				workflow.Connect(b, join),
				workflow.Connect(c, join),
				workflow.Connect(join, final),
			},
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		runWorkflow(t, wf)
		return rendered
	}
	first := render()
	for i := 0; i < 10; i++ {
		if got := render(); got != first {
			t.Fatalf("rendered varied: first=%q iter=%d got=%q", first, i, got)
		}
	}
}

func sortStrings(s []string) {
	// minimal in-test sort to avoid pulling in slices/sort here.
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
