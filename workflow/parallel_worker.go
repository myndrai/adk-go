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

package workflow

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"google.golang.org/adk/session"
)

// ParallelWorker fans out an inner Node across each element of an input
// slice and collects the per-element outputs. Mirrors adk-python's
// _parallel_worker._ParallelWorker.
//
// Concurrency is bounded by MaxConcurrency (zero = unbounded). On the
// first error the worker cancels remaining attempts and returns the error.
//
// The slice element type is captured by the T type parameter so user code
// stays type-safe; the engine still trafficks in any across edges.
type ParallelWorker[T any] struct {
	Base
	inner          Node
	maxConcurrency int
}

// ParallelOption customizes a ParallelWorker.
type ParallelOption func(*parallelOpts)

type parallelOpts struct {
	name           string
	description    string
	maxConcurrency int
}

// ParallelWithName overrides the default node name (defaults to
// "parallel_<inner>").
func ParallelWithName(name string) ParallelOption {
	return func(o *parallelOpts) { o.name = name }
}

// ParallelWithDescription sets the node description.
func ParallelWithDescription(desc string) ParallelOption {
	return func(o *parallelOpts) { o.description = desc }
}

// ParallelMaxConcurrency caps the number of inner-node attempts that may
// run simultaneously. 0 = unbounded.
func ParallelMaxConcurrency(n int) ParallelOption {
	return func(o *parallelOpts) { o.maxConcurrency = n }
}

// Parallel wraps inner so that it fans out across each item of the input
// slice. The wrapped node's input must be a []T; the wrapper's output is
// a []any of per-element results.
func Parallel[T any](inner Node, opts ...ParallelOption) *ParallelWorker[T] {
	if inner == nil {
		panic("workflow: Parallel requires a non-nil inner node")
	}
	o := parallelOpts{name: "parallel_" + inner.Name()}
	for _, opt := range opts {
		opt(&o)
	}
	w := &ParallelWorker[T]{inner: inner, maxConcurrency: o.maxConcurrency}
	if err := w.SetMetadata(o.name, o.description, NodeSpec{}); err != nil {
		panic(err)
	}
	return w
}

// RunImpl runs the inner node once per element of the input slice. Errors
// from any worker abort the lot via context cancellation.
//
// Concurrency model:
//   - Each worker gets its own *session.EventActions, populated through a
//     fresh per-worker NodeContext. Workers never share the parent's actions
//     map, so their concurrent StateDelta / ArtifactDelta writes can't race.
//   - After all workers finish, the parent goroutine merges per-worker
//     actions into the parent's actions deterministically by input index.
//   - Each worker's collectingEmitter buffers events; after the merge step
//     the parent forwards every buffered event to the parent emitter, also
//     in input-index order, so the workflow's iter.Seq2 sees a stable
//     sequence per parallel batch.
func (p *ParallelWorker[T]) RunImpl(ctx *NodeContext, input any, em EventEmitter) error {
	items, err := coerceParallelInput[T](input)
	if err != nil {
		return fmt.Errorf("parallel %q: %w", p.Name(), err)
	}
	if len(items) == 0 {
		return em.Output([]any{})
	}

	// Cancellation propagation across workers.
	cctx, cancel := context.WithCancel(ctx.InvocationContext)
	defer cancel()

	results := make([]any, len(items))
	errs := make([]error, len(items))
	subEms := make([]*collectingEmitter, len(items))
	subActions := make([]*session.EventActions, len(items))

	var sem chan struct{}
	if p.maxConcurrency > 0 {
		sem = make(chan struct{}, p.maxConcurrency)
	}

	var wg sync.WaitGroup
	for i := range items {
		i := i
		if sem != nil {
			select {
			case sem <- struct{}{}:
			case <-cctx.Done():
				goto wait
			}
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			if sem != nil {
				defer func() { <-sem }()
			}
			if cctx.Err() != nil {
				return
			}
			workerActions := &session.EventActions{
				StateDelta:    map[string]any{},
				ArtifactDelta: map[string]int64{},
			}
			subActions[i] = workerActions
			child := &NodeContext{
				InvocationContext: ctx.InvocationContext,
				nodePath:          fmt.Sprintf("%s/[%d]", ctx.NodePath(), i),
				runID:             fmt.Sprintf("%d", i),
				actions:           workerActions,
			}
			subEm := newCollectingEmitter(p.inner, ctx.InvocationContext, i, child.NodePath())
			subEm.ctx = child
			subEms[i] = subEm
			if err := p.inner.RunImpl(child, items[i], subEm); err != nil {
				errs[i] = err
				cancel()
				return
			}
			if len(subEm.outputs) > 0 {
				results[i] = subEm.outputs[len(subEm.outputs)-1]
			}
		}()
	}
wait:
	wg.Wait()

	for _, e := range errs {
		if e != nil {
			return e
		}
	}

	// Merge worker actions into parent in input-index order. State delta
	// keys collide last-write-wins by index.
	if ctx.actions != nil {
		for _, a := range subActions {
			if a == nil {
				continue
			}
			if ctx.actions.StateDelta == nil {
				ctx.actions.StateDelta = map[string]any{}
			}
			for k, v := range a.StateDelta {
				ctx.actions.StateDelta[k] = v
			}
			if ctx.actions.ArtifactDelta == nil {
				ctx.actions.ArtifactDelta = map[string]int64{}
			}
			for k, v := range a.ArtifactDelta {
				ctx.actions.ArtifactDelta[k] = v
			}
		}
	}

	// Forward buffered worker events to the parent emitter in input-index
	// order so the workflow's iter.Seq2 consumer sees a stable sequence.
	for _, sub := range subEms {
		if sub == nil {
			continue
		}
		for _, ev := range sub.events {
			if err := em.Event(ev); err != nil {
				return err
			}
		}
	}

	return em.Output(results)
}

// coerceParallelInput accepts a []T directly or any slice of compatible
// element type, returning a []T copy.
func coerceParallelInput[T any](data any) ([]T, error) {
	if data == nil {
		return nil, nil
	}
	if v, ok := data.([]T); ok {
		return v, nil
	}
	// Fallback: []any whose elements are T.
	if anys, ok := data.([]any); ok {
		out := make([]T, 0, len(anys))
		for i, item := range anys {
			t, ok := item.(T)
			if !ok {
				return nil, fmt.Errorf("parallel: element %d has type %T, want %T", i, item, *new(T))
			}
			out = append(out, t)
		}
		return out, nil
	}
	return nil, errors.New("parallel: input must be a slice")
}
