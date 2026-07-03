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

package compaction

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

func TestNewPluginValidatesConfig(t *testing.T) {
	if _, err := NewPlugin("compaction", nil, Config{}); err == nil {
		t.Fatal("NewPlugin with invalid config: want error, got nil")
	}
	if _, err := NewPlugin("compaction", nil, Config{Summarizer: nopSummarizer{}, CompactionInterval: 10, OverlapSize: 2}); err == nil {
		t.Fatal("NewPlugin with nil session.Service: want error, got nil")
	}
}

// fakePluginSummarizer is a deterministic summarizer for the behavioral
// plugin test; it always produces a summary event.
type fakePluginSummarizer struct{}

func (fakePluginSummarizer) MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error) {
	if len(events) == 0 {
		return nil, nil
	}
	out := session.NewEvent(ctx, "plugin-compaction")
	out.Author = "user"
	out.Actions.Compaction = &session.EventCompaction{
		StartTimestamp:   events[0].Timestamp,
		EndTimestamp:     events[len(events)-1].Timestamp,
		CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "summary"}}},
	}
	return out, nil
}

func TestNewPluginAfterRunCallbackTriggersCompaction(t *testing.T) {
	svc := session.InMemoryService()
	created, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName: "app", UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Seed two full invocations (user+model event pairs) so the
	// CompactionInterval=2 sliding-window trigger fires.
	t0 := time.Unix(100, 0)
	events := []*session.Event{
		userEvent("inv-1", "hello", t0),
		modelEvent("inv-1", "hi there", t0.Add(time.Second)),
		userEvent("inv-2", "how are you", t0.Add(2*time.Second)),
		modelEvent("inv-2", "doing well", t0.Add(3*time.Second)),
	}
	for _, e := range events {
		if err := svc.AppendEvent(context.Background(), created.Session, e); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}

	p, err := NewPlugin("compaction", svc, Config{
		Summarizer:         fakePluginSummarizer{},
		CompactionInterval: 2,
		OverlapSize:        0,
	})
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	ictx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Session: created.Session,
	})

	p.AfterRunCallback()(ictx)

	evs := created.Session.Events()
	if evs.Len() == 0 {
		t.Fatal("expected events in session")
	}
	last := evs.At(evs.Len() - 1)
	if last.Actions.Compaction == nil {
		t.Fatal("expected last event to carry a Compaction, got nil")
	}
}

func userEvent(invID, text string, ts time.Time) *session.Event {
	e := session.NewEvent(context.Background(), invID)
	e.Author = "user"
	e.Timestamp = ts
	e.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
	return e
}

func modelEvent(invID, text string, ts time.Time) *session.Event {
	e := session.NewEvent(context.Background(), invID)
	e.Author = "model"
	e.Timestamp = ts
	e.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{Text: text}}}
	return e
}
