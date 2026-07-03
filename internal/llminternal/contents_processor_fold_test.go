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

package llminternal_test

import (
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestContentsRequestProcessor_FoldsCompactedEvents pins that the processor
// itself (not just internal/compaction.Fold in isolation) applies event
// compaction: a compaction event's summary replaces the raw events inside
// its window, and req.Contents ends up with the summary text but not the
// compacted originals. Myndr fork extension (event compaction has no
// upstream v2 equivalent).
//
// The synthesized seed event Fold produces carries Author "model" (a
// generic marker, not a real agent name — see internal/compaction.Fold),
// so the processor's isOtherAgentReply check treats it as a foreign-agent
// reply and reframes it via ConvertForeignEvent ("For context: [model]
// said: ..."), same as it would for any other agent's history. The
// assertion below reflects that real, already-existing processor
// behavior rather than fighting it.
func TestContentsRequestProcessor_FoldsCompactedEvents(t *testing.T) {
	t.Parallel()
	base := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)

	old1 := &session.Event{
		Timestamp: base,
		Author:    "user",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("old question", "user"),
		},
	}
	old2 := &session.Event{
		Timestamp: base.Add(1 * time.Minute),
		Author:    "testAgent",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("old answer", "model"),
		},
	}
	comp := &session.Event{
		Timestamp: base.Add(2 * time.Minute),
		Author:    "testAgent",
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   base,
				EndTimestamp:     base.Add(1 * time.Minute),
				CompactedContent: genai.NewContentFromText("summary of old exchange", "model"),
			},
		},
	}
	fresh := &session.Event{
		Timestamp: base.Add(3 * time.Minute),
		Author:    "user",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("new question", "user"),
		},
	}

	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:  "testAgent",
		Model: &testModel{},
	}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: []*session.Event{old1, old2, comp, fresh}},
	})

	req := &model.LLMRequest{}
	for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("ContentsRequestProcessor generated an unexpected event")
		}
		if err != nil {
			t.Fatalf("ContentsRequestProcessor failed: %v", err)
		}
	}

	want := wantWithContinuation([]*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "For context:"},
				{Text: "[model] said: summary of old exchange"},
			},
		},
		genai.NewContentFromText("new question", "user"),
	})
	if diff := cmp.Diff(want, req.Contents); diff != "" {
		t.Errorf("req.Contents mismatch (-want +got):\n%s", diff)
	}

	// Regardless of framing, the raw compacted originals must not leak
	// into the model-bound contents, and the summary text must be present
	// exactly once.
	var allText string
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			allText += p.Text
		}
	}
	for _, unwanted := range []string{"old question", "old answer"} {
		if strings.Contains(allText, unwanted) {
			t.Errorf("req.Contents leaked compacted original text %q:\n%v", unwanted, req.Contents)
		}
	}
	if !strings.Contains(allText, "summary of old exchange") {
		t.Errorf("req.Contents missing compaction summary text:\n%v", req.Contents)
	}
}
