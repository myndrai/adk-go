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

// Package compaction provides opt-in event-history compaction for adk
// runners: older events are periodically summarized into a single
// compaction event, and internal/compaction.Fold substitutes the summary
// when history is rebuilt for the model. Myndr fork extension; upstream
// v2 has no equivalent.
//
// Session-backend requirement: compaction relies on the session service
// persisting Event.Actions.Compaction verbatim across reloads. The
// in-memory and database (session/database) services round-trip
// EventActions in full and are safe to use with compaction. The VertexAI
// session service (session/vertexai) maps only StateDelta into its
// persisted EventActions representation and drops Compaction, so after a
// reload the compaction history is lost: MaybeRun re-summarizes the entire
// event history on every subsequent turn (a cost leak) and Fold never
// folds because no persisted compaction event survives to be found. Do not
// enable compaction against a VertexAI-backed session service.
//
// Isolation-scope interaction: events carrying a non-empty
// session.Event.IsolationScope (see internal/llminternal's exact-match
// scope filtering) are excluded from compaction entirely. They are never
// included in a compaction window or passed to the Summarizer, and
// internal/compaction.Fold always lets them pass through unfolded — a
// scoped agent's raw history is preserved as-is rather than being
// summarized into (and thereby leaked into) another agent's unscoped
// compaction summary.
package compaction

import (
	"errors"
	"fmt"

	icompaction "google.golang.org/adk/v2/internal/compaction"
)

// Summarizer produces a compaction event from a span of session events.
type Summarizer = icompaction.Summarizer

// Config configures compaction. At least one trigger (sliding window via
// CompactionInterval, or token threshold via TokenThreshold +
// EventRetentionSize) must be set. Field semantics are unchanged from the
// v1 fork's app.EventsCompactionConfig.
type Config struct {
	// Summarizer generates the summary event. Required.
	Summarizer Summarizer
	// CompactionInterval is the sliding-window size in events. When >0,
	// compaction fires each time this many new events accumulate.
	CompactionInterval int
	// OverlapSize is how many events of the previous window are re-included
	// for continuity. Must be < CompactionInterval when the window trigger
	// is used.
	OverlapSize int
	// TokenThreshold, when set, fires compaction once the latest known
	// prompt token count exceeds it.
	TokenThreshold *int
	// EventRetentionSize, when set with TokenThreshold, is how many recent
	// events are always kept uncompacted.
	EventRetentionSize *int
}

// Validate reports whether the config is usable. Mirrors the v1 fork's
// app.EventsCompactionConfig.validate. The sliding-window and token-threshold
// triggers can each be configured independently; however, TokenThreshold and
// EventRetentionSize must be set together (one cannot be set without the other).
func (c Config) Validate() error {
	if c.Summarizer == nil {
		return errors.New("compaction: Summarizer is required")
	}
	hasWindow := c.CompactionInterval != 0
	hasToken := c.TokenThreshold != nil
	hasRetention := c.EventRetentionSize != nil
	if !hasWindow && !hasToken {
		return errors.New("compaction: configure CompactionInterval and/or TokenThreshold")
	}
	if (c.TokenThreshold == nil) != (c.EventRetentionSize == nil) {
		return errors.New("compaction: TokenThreshold and EventRetentionSize must be set together")
	}
	if c.CompactionInterval < 0 {
		return fmt.Errorf("compaction: CompactionInterval must be >= 0, got %d", c.CompactionInterval)
	}
	if c.OverlapSize < 0 {
		return fmt.Errorf("compaction: OverlapSize must be >= 0, got %d", c.OverlapSize)
	}
	if hasWindow && c.OverlapSize >= c.CompactionInterval {
		return fmt.Errorf("compaction: OverlapSize (%d) must be smaller than CompactionInterval (%d)", c.OverlapSize, c.CompactionInterval)
	}
	if hasToken && *c.TokenThreshold <= 0 {
		return fmt.Errorf("compaction: TokenThreshold must be > 0, got %d", *c.TokenThreshold)
	}
	if hasRetention && *c.EventRetentionSize <= 0 {
		return fmt.Errorf("compaction: EventRetentionSize must be > 0, got %d", *c.EventRetentionSize)
	}
	return nil
}
