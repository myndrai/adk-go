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
	"errors"
	"log"

	"google.golang.org/adk/v2/agent"
	icompaction "google.golang.org/adk/v2/internal/compaction"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

// NewPlugin returns a runner plugin that runs one round of best-effort
// compaction after every invocation (the v2 runner invokes the after-run
// callback at the end of Run and RunLive). Compaction failures are logged,
// never propagated — a failed summary must not fail the user's turn.
func NewPlugin(name string, svc session.Service, cfg Config) (*plugin.Plugin, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if svc == nil {
		return nil, errors.New("compaction: session.Service is required")
	}
	return plugin.New(plugin.Config{
		Name: name,
		AfterRunCallback: func(ictx agent.InvocationContext) {
			sess := ictx.Session()
			if sess == nil {
				return
			}
			agentName := ""
			if a := ictx.Agent(); a != nil {
				agentName = a.Name()
			}
			if _, err := icompaction.MaybeRun(ictx, icompaction.MaybeRunInput{
				Summarizer:         cfg.Summarizer,
				CompactionInterval: cfg.CompactionInterval,
				OverlapSize:        cfg.OverlapSize,
				TokenThreshold:     cfg.TokenThreshold,
				EventRetentionSize: cfg.EventRetentionSize,
				Session:            sess,
				SessionService:     svc,
				AppName:            sess.AppName(),
				UserID:             sess.UserID(),
				SessionID:          sess.ID(),
				CurrentBranch:      ictx.Branch(),
				AgentName:          agentName,
			}); err != nil {
				log.Printf("compaction error (non-fatal): %v", err)
			}
		},
	})
}
