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

package session

import (
	"testing"
	"time"

	"google.golang.org/genai"
)

func TestEventActionsCompactionField(t *testing.T) {
	start := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add(time.Hour)
	a := EventActions{Compaction: &EventCompaction{
		StartTimestamp:   start,
		EndTimestamp:     end,
		CompactedContent: genai.NewContentFromText("summary", "model"),
	}}
	if a.Compaction == nil || !a.Compaction.StartTimestamp.Equal(start) || !a.Compaction.EndTimestamp.Equal(end) {
		t.Fatalf("Compaction field round-trip failed: %+v", a.Compaction)
	}
	if got := a.Compaction.CompactedContent.Parts[0].Text; got != "summary" {
		t.Fatalf("CompactedContent = %q, want %q", got, "summary")
	}
}
