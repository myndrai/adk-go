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

package llminternal

import (
	"context"
	"iter"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

// TestRenderTaskInput_DeterministicKeyOrder locks down review fix #12:
// the rendered prompt must be stable regardless of map insertion order,
// because Go map iteration is randomized and replay/eval/prompt-caching
// rely on deterministic input.
func TestRenderTaskInput_DeterministicKeyOrder(t *testing.T) {
	t.Parallel()
	values := map[string]int{"alpha": 1, "beta": 2, "gamma": 3, "delta": 4}
	build := func(order []string) session.TaskRequest {
		input := map[string]any{}
		for _, k := range order {
			input[k] = values[k]
		}
		return session.TaskRequest{AgentName: "x", Input: input}
	}

	r1 := renderTaskInput(build([]string{"alpha", "beta", "gamma", "delta"}))
	r2 := renderTaskInput(build([]string{"delta", "gamma", "beta", "alpha"}))
	r3 := renderTaskInput(build([]string{"gamma", "alpha", "delta", "beta"}))

	if r1 != r2 || r2 != r3 {
		t.Fatalf("renderTaskInput varied across insertion orders:\n%q\n%q\n%q", r1, r2, r3)
	}

	// Verify alphabetic order: alpha appears before beta which appears
	// before delta which appears before gamma.
	wantOrder := []string{"alpha", "beta", "delta", "gamma"}
	for i := 1; i < len(wantOrder); i++ {
		idxPrev := strings.Index(r1, wantOrder[i-1])
		idxCur := strings.Index(r1, wantOrder[i])
		if idxPrev < 0 || idxCur < 0 || idxPrev > idxCur {
			t.Errorf("expected %q before %q in rendered output:\n%s", wantOrder[i-1], wantOrder[i], r1)
		}
	}
}

// TestTaskDepthCtx_HelpersRoundTrip is a thin sanity test for the
// taskDepth / withTaskDepth context helpers introduced for review fix #9.
func TestTaskDepthCtx_HelpersRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	if got := taskDepth(ctx); got != 0 {
		t.Fatalf("initial depth = %d, want 0", got)
	}
	ctx = withTaskDepth(ctx, 3)
	if got := taskDepth(ctx); got != 3 {
		t.Fatalf("after withTaskDepth(3) = %d, want 3", got)
	}
	ctx = withTaskDepth(ctx, 7)
	if got := taskDepth(ctx); got != 7 {
		t.Fatalf("after withTaskDepth(7) = %d, want 7", got)
	}
}

// emittingAgent constructs a real agent.Agent (via agent.New so the
// sealed Agent interface is satisfied) whose Run yields a fixed event
// sequence. Sub-agents and FindAgent are wired through agent.New.
func emittingAgent(t *testing.T, name string, emit []*session.Event, subs ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:      name,
		SubAgents: subs,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, ev := range emit {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q): %v", name, err)
	}
	return a
}

func newTaskMeshCtx(t *testing.T, current agent.Agent, root agent.Agent) agent.InvocationContext {
	t.Helper()
	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatalf("parentmap.New: %v", err)
	}
	svc := session.InMemoryService()
	cr, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sid",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	return icontext.NewInvocationContext(
		parentmap.ToContext(t.Context(), parents),
		icontext.InvocationContextParams{
			Agent:        current,
			Session:      cr.Session,
			InvocationID: "inv-1",
		},
	)
}

func eventWithFinishTask(callID, agentName string, output map[string]any) *session.Event {
	ev := session.NewEvent("inv-1")
	ev.Author = agentName
	ev.Actions.FinishTask = map[string]session.TaskResult{
		callID: {AgentName: agentName, Output: output},
	}
	return ev
}

// TestRunTaskRequests_SingleTask_SynthesizesFunctionResponse verifies the
// happy path: one delegated task whose agent emits FinishTask, the runner
// synthesizes a FunctionResponse keyed by the original call ID.
func TestRunTaskRequests_SingleTask_SynthesizesFunctionResponse(t *testing.T) {
	worker := emittingAgent(t, "worker", []*session.Event{
		eventWithFinishTask("call-1", "worker", map[string]any{"answer": 42}),
	})
	coord := emittingAgent(t, "coord", nil, worker)

	ctx := newTaskMeshCtx(t, coord, coord)

	ev := session.NewEvent("inv-1")
	ev.Actions.RequestTask = map[string]session.TaskRequest{
		"call-1": {AgentName: "worker", Input: map[string]any{"q": "?"}},
	}

	var yielded []*session.Event
	yield := func(e *session.Event, err error) bool {
		if err != nil {
			t.Errorf("unexpected yield err: %v", err)
			return false
		}
		yielded = append(yielded, e)
		return true
	}
	f := &Flow{}
	if !f.runTaskRequests(ctx, ev, yield) {
		t.Fatal("runTaskRequests returned false (yield aborted)")
	}

	// Expect: child's FinishTask event (1) + synthetic FunctionResponse (1).
	if len(yielded) != 2 {
		t.Fatalf("yielded %d events, want 2", len(yielded))
	}
	fr := yielded[1]
	if fr.Author != "coord" {
		t.Errorf("synth Author = %q, want coord", fr.Author)
	}
	if fr.Content == nil || len(fr.Content.Parts) == 0 || fr.Content.Parts[0].FunctionResponse == nil {
		t.Fatalf("synth event missing FunctionResponse: %+v", fr)
	}
	got := fr.Content.Parts[0].FunctionResponse
	if got.ID != "call-1" {
		t.Errorf("FunctionResponse.ID = %q, want call-1", got.ID)
	}
	if got.Name != "worker" {
		t.Errorf("FunctionResponse.Name = %q, want worker", got.Name)
	}
	if v, ok := got.Response["answer"].(int); !ok || v != 42 {
		t.Errorf("FunctionResponse.Response[answer] = %v, want 42", got.Response["answer"])
	}
}

// TestRunTaskRequests_AgentNotFound surfaces an error rather than crashing
// when a coordinator references a non-existent task agent.
func TestRunTaskRequests_AgentNotFound(t *testing.T) {
	coord := emittingAgent(t, "coord", nil)
	ctx := newTaskMeshCtx(t, coord, coord)

	ev := session.NewEvent("inv-1")
	ev.Actions.RequestTask = map[string]session.TaskRequest{
		"call-1": {AgentName: "missing", Input: nil},
	}

	var gotErr error
	yield := func(_ *session.Event, err error) bool {
		gotErr = err
		return false
	}
	f := &Flow{}
	if f.runTaskRequests(ctx, ev, yield) {
		t.Error("runTaskRequests returned true for missing agent; want false (aborted)")
	}
	if gotErr == nil {
		t.Fatal("expected error for missing agent")
	}
	if !strings.Contains(gotErr.Error(), "missing") {
		t.Errorf("err = %v, want it to mention agent name", gotErr)
	}
}

// TestRunTaskRequests_OrphanTask synthesizes an empty payload when the
// task agent never emits FinishTask.
func TestRunTaskRequests_OrphanTask(t *testing.T) {
	silent := emittingAgent(t, "silent", nil) // emits no events
	coord := emittingAgent(t, "coord", nil, silent)
	ctx := newTaskMeshCtx(t, coord, coord)

	ev := session.NewEvent("inv-1")
	ev.Actions.RequestTask = map[string]session.TaskRequest{
		"call-x": {AgentName: "silent", Input: nil},
	}

	var yielded []*session.Event
	yield := func(e *session.Event, err error) bool {
		if err != nil {
			t.Errorf("unexpected err: %v", err)
		}
		yielded = append(yielded, e)
		return true
	}
	f := &Flow{}
	if !f.runTaskRequests(ctx, ev, yield) {
		t.Fatal("runTaskRequests returned false")
	}
	// Only the synthetic FunctionResponse is yielded (no child events).
	if len(yielded) != 1 {
		t.Fatalf("yielded %d events, want 1", len(yielded))
	}
	got := yielded[0]
	if got.Content == nil || got.Content.Parts[0].FunctionResponse == nil {
		t.Fatalf("expected synthetic FunctionResponse, got %+v", got)
	}
	resp := got.Content.Parts[0].FunctionResponse.Response
	if len(resp) != 0 {
		t.Errorf("expected empty payload, got %v", resp)
	}
}

// TestRunTaskRequests_MultiTask exercises the loop path: two distinct
// call IDs in one Actions.RequestTask map. Each gets its own
// FunctionResponse keyed by the matching call ID.
func TestRunTaskRequests_MultiTask(t *testing.T) {
	a := emittingAgent(t, "a", []*session.Event{eventWithFinishTask("c1", "a", map[string]any{"v": 1})})
	b := emittingAgent(t, "b", []*session.Event{eventWithFinishTask("c2", "b", map[string]any{"v": 2})})
	coord := emittingAgent(t, "coord", nil, a, b)
	ctx := newTaskMeshCtx(t, coord, coord)

	ev := session.NewEvent("inv-1")
	ev.Actions.RequestTask = map[string]session.TaskRequest{
		"c1": {AgentName: "a"},
		"c2": {AgentName: "b"},
	}

	var fr []*genai.FunctionResponse
	yield := func(e *session.Event, err error) bool {
		if err != nil {
			t.Errorf("err: %v", err)
		}
		if e == nil || e.Content == nil {
			return true
		}
		if p := e.Content.Parts; len(p) > 0 && p[0].FunctionResponse != nil {
			fr = append(fr, p[0].FunctionResponse)
		}
		return true
	}
	f := &Flow{}
	if !f.runTaskRequests(ctx, ev, yield) {
		t.Fatal("runTaskRequests returned false")
	}
	if len(fr) != 2 {
		t.Fatalf("got %d FunctionResponses, want 2", len(fr))
	}
	// Each call ID must map to its own response. Map iteration order
	// is non-deterministic so just check that both IDs appear once.
	seen := map[string]bool{}
	for _, r := range fr {
		seen[r.ID] = true
	}
	if !seen["c1"] || !seen["c2"] {
		t.Errorf("expected call IDs c1 and c2, got %v", seen)
	}
}
