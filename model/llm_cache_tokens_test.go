package model

import "testing"

// The cache-token fields are plain data carried by LLMResponse; this test
// pins their existence and zero-value semantics so an upstream sync that
// drops them fails loudly.
func TestLLMResponseCacheTokenFields(t *testing.T) {
	var r LLMResponse
	if r.CachedInputTokens != 0 || r.CacheCreationTokens != 0 {
		t.Fatalf("zero value: got CachedInputTokens=%d CacheCreationTokens=%d, want 0,0",
			r.CachedInputTokens, r.CacheCreationTokens)
	}
	r.CachedInputTokens = 1200
	r.CacheCreationTokens = 34
	if r.CachedInputTokens != 1200 || r.CacheCreationTokens != 34 {
		t.Fatalf("assignment: got %d,%d want 1200,34", r.CachedInputTokens, r.CacheCreationTokens)
	}
}
