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
	"fmt"
	"slices"
	"strings"

	"google.golang.org/genai"

	icontext "google.golang.org/adk/internal/context"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// taskDepthKey is the context.Context key that tracks how many task
// delegations are nested in the current call chain.
type taskDepthKey struct{}

// DefaultMaxTaskDepth caps how deep coordinator→task→coordinator chains
// can recurse before the runtime synthesizes an error response instead of
// invoking the next task agent. Override via WithMaxTaskDepth on the
// invocation context.
const DefaultMaxTaskDepth = 4

// taskDepth returns the current task-delegation depth pulled from ctx,
// or 0 if unset.
func taskDepth(ctx context.Context) int {
	if v := ctx.Value(taskDepthKey{}); v != nil {
		if d, ok := v.(int); ok {
			return d
		}
	}
	return 0
}

// withTaskDepth returns a context whose task-depth counter is one higher.
func withTaskDepth(ctx context.Context, d int) context.Context {
	return context.WithValue(ctx, taskDepthKey{}, d)
}

// runTaskRequests handles any session.TaskRequest entries the coordinator
// produced via task.NewRequestTaskTool. For each (function_call_id,
// TaskRequest), the named task agent is located in the agent tree and
// invoked with a rendered user message; events from the task agent are
// forwarded to yield. After the task agent emits a session.TaskResult
// (via task.NewFinishTaskTool), a synthetic FunctionResponse event is
// produced and yielded so the coordinator's next turn observes the
// result keyed by the original call ID.
//
// The implementation is conservative: it consumes events serially per
// task to keep ordering deterministic. Parallel delegation across
// multiple call IDs is enabled by simply scheduling each task in the
// merged event one after another — the LLM still sees them as one
// batch because they share the model turn that originated them.
func (f *Flow) runTaskRequests(
	ctx agent.InvocationContext,
	ev *session.Event,
	yield func(*session.Event, error) bool,
) bool {
	if ev == nil || len(ev.Actions.RequestTask) == 0 {
		return true
	}
	depth := taskDepth(ctx)
	maxDepth := DefaultMaxTaskDepth
	parents := parentmap.FromContext(ctx)
	for callID, req := range ev.Actions.RequestTask {
		if depth >= maxDepth {
			// Synthesize a guard FunctionResponse instead of running the
			// task agent so the coordinator can react gracefully (e.g.
			// fall back to a different plan or surface the limit to the
			// user) rather than the runtime hanging in a recursion loop.
			payload := map[string]any{
				"error":    fmt.Sprintf("task delegation depth %d exceeded max %d", depth, maxDepth),
				"agent":    req.AgentName,
				"callID":   callID,
				"maxDepth": maxDepth,
			}
			fr := session.NewEvent(ctx.InvocationID())
			fr.Author = ctx.Agent().Name()
			fr.Branch = ctx.Branch()
			fr.LLMResponse = model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleUser,
					Parts: []*genai.Part{{
						FunctionResponse: &genai.FunctionResponse{
							ID:       callID,
							Name:     req.AgentName,
							Response: payload,
						},
					}},
				},
			}
			if !yield(fr, nil) {
				return false
			}
			continue
		}
		taskAgent := findAgentByName(ctx.Agent(), parents, req.AgentName)
		if taskAgent == nil {
			yield(nil, fmt.Errorf("task: agent %q not found in tree", req.AgentName))
			return false
		}
		// Build a child invocation context whose UserContent renders the
		// task input and whose context value increments the delegation
		// depth so coordinator-task-coordinator chains terminate.
		childCtx := newTaskChildContext(ctx, taskAgent, req, depth+1)
		var finish *session.TaskResult
		for childEv, err := range taskAgent.Run(childCtx) {
			if !yield(childEv, err) {
				return false
			}
			if err != nil {
				return false
			}
			if childEv != nil && len(childEv.Actions.FinishTask) > 0 {
				// Pick the latest FinishTask entry; tasks normally produce one.
				for _, r := range childEv.Actions.FinishTask {
					r := r
					finish = &r
				}
			}
		}
		// Synthesize a FunctionResponse for the coordinator. If the task
		// never called finish_task, synthesize an empty result so the
		// coordinator can decide how to proceed rather than stalling.
		respPayload := map[string]any{}
		if finish != nil && finish.Output != nil {
			respPayload = finish.Output
		}
		fr := session.NewEvent(ctx.InvocationID())
		fr.Author = ctx.Agent().Name()
		fr.Branch = ctx.Branch()
		fr.LLMResponse = model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     req.AgentName,
						Response: respPayload,
					},
				}},
			},
		}
		if !yield(fr, nil) {
			return false
		}
	}
	return true
}

// findAgentByName searches the agent tree (root via parents) for the
// named agent. Returns nil if not found.
func findAgentByName(start agent.Agent, parents parentmap.Map, name string) agent.Agent {
	if start == nil {
		return nil
	}
	root := start
	for {
		p := parents[root.Name()]
		if p == nil {
			break
		}
		root = p
	}
	return root.FindAgent(name)
}

// newTaskChildContext builds a sub-invocation context for a task agent.
// Mirrors the scaffolding in agent.Run: a fresh InvocationContext with
// the same session/services, but UserContent rendered from the task
// input so the task agent's contents-builder picks it up naturally.
//
// The newDepth value is propagated through ctx.Value(taskDepthKey{}) so
// nested coordinator→task→coordinator chains observe the running depth
// and terminate at DefaultMaxTaskDepth.
func newTaskChildContext(parent agent.InvocationContext, taskAgent agent.Agent, req session.TaskRequest, newDepth int) agent.InvocationContext {
	rendered := renderTaskInput(req)
	uc := &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{Text: rendered}},
	}
	child := icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Artifacts:    parent.Artifacts(),
		Memory:       parent.Memory(),
		Session:      parent.Session(),
		Agent:        taskAgent,
		UserContent:  uc,
		RunConfig:    parent.RunConfig(),
		InvocationID: parent.InvocationID(),
	})
	return child.WithContext(withTaskDepth(child, newDepth))
}

// renderTaskInput formats a TaskRequest's input as a human-readable
// string. Mirrors adk-python's render_task_input — labels each field.
//
// Keys are sorted alphabetically before formatting so the rendered prompt
// is deterministic across runs (Go map iteration is randomized). Replay,
// eval, and prompt caching all rely on stable input rendering.
func renderTaskInput(req session.TaskRequest) string {
	var b strings.Builder
	b.WriteString("[Delegated Task]\n")
	keys := make([]string, 0, len(req.Input))
	for k := range req.Input {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "%s: %v\n", k, req.Input[k])
	}
	return b.String()
}
