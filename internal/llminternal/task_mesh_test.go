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
	"strings"
	"testing"

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
