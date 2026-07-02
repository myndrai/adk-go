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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// The prompt-cache token counters (myndr fork extension) are reported by
// providers on a single chunk of a stream; the aggregated final response
// must retain them rather than silently zeroing them, or downstream cost
// rollups undercount cached tokens. Internal-package test because the
// counters only enter via adapter-produced LLMResponses, which the
// exported genai-based entry point cannot carry.
func TestAggregatedResponseCarriesCacheTokens(t *testing.T) {
	agg := NewStreamingResponseAggregator()

	// Chunk 1: content plus the cache counters (Anthropic reports these
	// on message_start/message_delta, not on every chunk).
	agg.aggregateResponse(&model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{genai.NewPartFromText("hel")},
			Role:  genai.RoleModel,
		},
		CachedInputTokens:   1200,
		CacheCreationTokens: 34,
	})
	// Chunk 2: trailing content with zero-valued counters — must not
	// clobber the recorded values.
	agg.aggregateResponse(&model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{genai.NewPartFromText("lo")},
			Role:  genai.RoleModel,
		},
		FinishReason: genai.FinishReasonStop,
	})

	final := agg.Close()
	if final == nil {
		t.Fatal("Close() returned nil final response")
	}
	if final.CachedInputTokens != 1200 || final.CacheCreationTokens != 34 {
		t.Fatalf("final response cache tokens = %d,%d; want 1200,34",
			final.CachedInputTokens, final.CacheCreationTokens)
	}
}
