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
