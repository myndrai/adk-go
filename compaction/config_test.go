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

	"google.golang.org/adk/v2/session"
)

type nopSummarizer struct{}

func (nopSummarizer) MaybeSummarize(ctx context.Context, events []*session.Event) (*session.Event, error) {
	return nil, nil
}

func TestConfigValidate(t *testing.T) {
	tt := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"no summarizer", Config{CompactionInterval: 10}, true},
		{"no trigger configured", Config{Summarizer: nopSummarizer{}}, true},
		{"sliding window ok", Config{Summarizer: nopSummarizer{}, CompactionInterval: 10, OverlapSize: 2}, false},
		{"overlap >= interval", Config{Summarizer: nopSummarizer{}, CompactionInterval: 5, OverlapSize: 5}, true},
		{"negative interval", Config{Summarizer: nopSummarizer{}, CompactionInterval: -1}, true},
		{"token threshold ok", Config{Summarizer: nopSummarizer{}, TokenThreshold: ptr(4096), EventRetentionSize: ptr(6)}, false},
	}
	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Fatalf("Validate() err=%v, wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func ptr(i int) *int { return &i }
