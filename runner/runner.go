// Copyright 2025 Google LLC
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

// Package runner provides a runtime for ADK agents.
package runner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/app"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/internal/artifact"
	"google.golang.org/adk/internal/compaction"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/plugininternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
)

// Config is used to create a [Runner].
//
// Either Agent (v1 form) or App (v2 form) must be supplied — but not both.
// When App is set, AppName, Agent, and PluginConfig are read from it; the
// equivalent top-level fields are then optional and ignored if also set.
type Config struct {
	AppName string
	// Root agent which starts the execution. v1 form. Mutually exclusive
	// with App. Future Phase 2 will allow workflow.Node here too.
	Agent          agent.Agent
	SessionService session.Service

	// App is the v2 container that pairs the root agent with shared plugins
	// and runtime configurations (event compaction, context cache,
	// resumability). Mutually exclusive with the top-level Agent field.
	App *app.App

	// optional
	ArtifactService artifact.Service
	// optional
	MemoryService memory.Service
	// optional. Ignored when App is set; use App.Plugins instead.
	PluginConfig PluginConfig
	// optional
	AutoCreateSession bool
}

type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
}

type RunOption func(*runOptions)

type runOptions struct {
	stateDelta map[string]any
}

// WithStateDelta sets a state delta for the run invocation.
func WithStateDelta(delta map[string]any) RunOption {
	return func(o *runOptions) {
		o.stateDelta = delta
	}
}

// New creates a new [Runner].
//
// Accepts both v1-style (Config.Agent + Config.PluginConfig) and v2-style
// (Config.App) construction. When Config.App is set, it takes precedence
// over the v1 fields except for SessionService and ArtifactService /
// MemoryService, which always come from Config.
func New(cfg Config) (*Runner, error) {
	if cfg.App != nil && cfg.Agent != nil {
		return nil, errors.New("runner: set either Config.Agent or Config.App, not both")
	}

	rootAgent := cfg.Agent
	appName := cfg.AppName
	plugins := cfg.PluginConfig.Plugins
	closeTimeout := cfg.PluginConfig.CloseTimeout

	if cfg.App != nil {
		rootAgent = cfg.App.RootAgent
		if appName == "" {
			appName = cfg.App.Name
		}
		plugins = cfg.App.Plugins
		// closeTimeout stays from PluginConfig if caller supplied it
		// alongside the App; App itself doesn't model a timeout today.
	}

	if rootAgent == nil {
		return nil, fmt.Errorf("root agent is required")
	}

	if cfg.SessionService == nil {
		return nil, fmt.Errorf("session service is required")
	}

	parents, err := parentmap.New(rootAgent)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent tree: %w", err)
	}

	pluginManager, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins:      plugins,
		CloseTimeout: closeTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin manager: %w", err)
	}

	return &Runner{
		appName:           appName,
		rootAgent:         rootAgent,
		sessionService:    cfg.SessionService,
		artifactService:   cfg.ArtifactService,
		memoryService:     cfg.MemoryService,
		parents:           parents,
		pluginManager:     pluginManager,
		autoCreateSession: cfg.AutoCreateSession,
		appCfg:            cfg.App,
	}, nil
}

// Runner manages the execution of the agent within a session, handling message
// processing, event generation, and interaction with various services like
// artifact storage, session management, and memory.
type Runner struct {
	appName         string
	rootAgent       agent.Agent
	sessionService  session.Service
	artifactService artifact.Service
	memoryService   memory.Service

	parents           parentmap.Map
	pluginManager     *plugininternal.PluginManager
	autoCreateSession bool

	// appCfg is set when Runner was constructed via Config.App. It carries
	// app-level configuration (event compaction, context cache, resumability)
	// to runtime hooks introduced in later Phase 1 tracks.
	appCfg *app.App
}

// ErrNotResumable is returned by Run/Resume when no new message is provided
// and the runner's App does not have ResumabilityConfig.IsResumable set.
// Mirrors adk-python runners.py:884.
var ErrNotResumable = errors.New("runner: running an agent requires a new_message or a resumable app")

// Resume re-enters an existing session without appending a new user message.
// The root agent gets a chance to rehydrate from prior session events and
// continue from the first non-completed point. Resume only succeeds when the
// runner was constructed via Config.App with ResumabilityConfig.IsResumable
// set; otherwise it yields ErrNotResumable.
//
// In this foundation PR Resume is the API surface only — the actual workflow
// rehydration semantics arrive with the workflow package in a later PR.
func (r *Runner) Resume(ctx context.Context, userID, sessionID string, cfg agent.RunConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
	if !r.isResumable() {
		return func(yield func(*session.Event, error) bool) {
			yield(nil, ErrNotResumable)
		}
	}
	return r.Run(ctx, userID, sessionID, nil, cfg, opts...)
}

// isResumable reports whether the runner's App is configured to permit
// resume invocations (msg=nil).
func (r *Runner) isResumable() bool {
	return r.appCfg != nil && r.appCfg.ResumabilityConfig != nil && r.appCfg.ResumabilityConfig.IsResumable
}

// Run runs the agent for the given user input, yielding events from agents.
// For each user message it finds the proper agent within an agent tree to
// continue the conversation within the session.
//
// Calling Run with msg == nil is only valid when the runner was constructed
// from an App whose ResumabilityConfig.IsResumable is true. Otherwise Run
// yields ErrNotResumable. Mirrors adk-python runners.py:884.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*session.Event, error) bool) {
		if msg == nil && !r.isResumable() {
			yield(nil, ErrNotResumable)
			return
		}
		options := runOptions{}
		for _, opt := range opts {
			opt(&options)
		}

		var storedSession session.Session
		getResp, err := r.sessionService.Get(ctx, &session.GetRequest{
			AppName:   r.appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			if !r.autoCreateSession {
				yield(nil, err)
				return
			}
			createResp, err := r.sessionService.Create(ctx, &session.CreateRequest{
				AppName:   r.appName,
				UserID:    userID,
				SessionID: sessionID,
			})
			if err != nil {
				yield(nil, err)
				return
			}
			storedSession = createResp.Session
		} else {
			storedSession = getResp.Session
		}

		agentToRun, err := r.findAgentToRun(storedSession, msg)
		if err != nil {
			yield(nil, err)
			return
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})
		ctx = plugininternal.ToContext(ctx, r.pluginManager)

		var artifacts agent.Artifacts
		if r.artifactService != nil {
			artifacts = &artifactinternal.Artifacts{
				Service:   r.artifactService,
				SessionID: storedSession.ID(),
				AppName:   storedSession.AppName(),
				UserID:    storedSession.UserID(),
			}
		}

		var memoryImpl agent.Memory = nil
		if r.memoryService != nil {
			memoryImpl = &imemory.Memory{
				Service:   r.memoryService,
				SessionID: storedSession.ID(),
				UserID:    storedSession.UserID(),
				AppName:   storedSession.AppName(),
			}
		}

		ctx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:   artifacts,
			Memory:      memoryImpl,
			Session:     storedSession,
			Agent:       agentToRun,
			UserContent: msg,
			RunConfig:   &cfg,
		})
		ctx, err = r.appendMessageToSession(ctx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager, options.stateDelta)
		if err != nil {
			yield(nil, err)
			return
		}

		pluginManager := r.pluginManager
		if pluginManager != nil {
			// Defer the after run callbacks to perform global cleanup tasks or finalizing logs and metrics data.
			// This does NOT emit any event.
			defer pluginManager.RunAfterRunCallback(ctx)

			earlyExitResult, err := pluginManager.RunBeforeRunCallback(ctx)
			if earlyExitResult != nil || err != nil {
				// The user message has already been appended to the session by
				// appendMessageToSession above. Don't append a second user-authored
				// duplicate here. When the BeforeRun plugin produced an
				// early-exit content, surface it as an agent-authored event;
				// when it only returned an error, surface the error alone.
				// Mirrors adk-python runners.py:1166-1180.
				if earlyExitResult != nil {
					earlyExitEvent := session.NewEvent(ctx.InvocationID())
					earlyExitEvent.Author = agentToRun.Name()
					earlyExitEvent.Branch = ctx.Branch()
					earlyExitEvent.LLMResponse = model.LLMResponse{
						Content: earlyExitResult,
					}
					if appendErr := r.sessionService.AppendEvent(ctx, storedSession, earlyExitEvent); appendErr != nil {
						yield(nil, fmt.Errorf("failed to add event to session: %w", appendErr))
						return
					}
					yield(earlyExitEvent, err)
					return
				}
				yield(nil, err)
				return
			}
		}

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			if pluginManager != nil {
				modifiedEvent, err := pluginManager.RunOnEventCallback(ctx, event)
				if err != nil {
					if !yield(nil, err) {
						return
					}
					continue
				}
				if modifiedEvent != nil {
					event = modifiedEvent
				}
			}

			// only commit non-partial event to a session service
			if !event.LLMResponse.Partial {
				if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
					yield(nil, fmt.Errorf("failed to add event to session: %w", err))
					return
				}
			}

			if !yield(event, nil) {
				return
			}
		}

		// Post-invocation compaction (Phase 1E). Runs only when an App with
		// EventsCompactionConfig was supplied. Failures are logged but do not
		// abort the invocation since compaction is best-effort.
		if cc := r.compactionConfig(); cc != nil {
			if _, err := compaction.MaybeRun(ctx, compaction.MaybeRunInput{
				Summarizer:         cc.Summarizer,
				CompactionInterval: cc.CompactionInterval,
				OverlapSize:        cc.OverlapSize,
				TokenThreshold:     cc.TokenThreshold,
				EventRetentionSize: cc.EventRetentionSize,
				Session:            storedSession,
				SessionService:     r.sessionService,
				AppName:            r.appName,
				UserID:             storedSession.UserID(),
				SessionID:          storedSession.ID(),
				CurrentBranch:      ctx.Branch(),
				AgentName:          agentToRun.Name(),
			}); err != nil {
				log.Printf("compaction error (non-fatal): %v", err)
			}
		}
	}
}

type liveAgent interface {
	RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

// RunLive runs a live session for the agent, supporting bidirectional streaming.
type runnerLiveSession struct {
	sess          agent.LiveSession
	r             *Runner
	iCtx          agent.InvocationContext
	storedSession session.Session
}

func (s *runnerLiveSession) Send(req agent.LiveRequest) error {
	err := s.sess.Send(req)
	if err != nil {
		return err
	}

	// Save user text content to session history
	if req.Content != nil && len(req.Content.Parts) > 0 {
		// Skip function responses - they are handled separately
		isFunctionResponse := false
		for _, part := range req.Content.Parts {
			if part.FunctionResponse != nil {
				isFunctionResponse = true
				break
			}
		}

		if !isFunctionResponse {
			event := session.NewEvent(s.iCtx.InvocationID())
			event.Author = "user"
			event.LLMResponse = model.LLMResponse{
				Content: req.Content,
			}
			if err := s.r.sessionService.AppendEvent(s.iCtx, s.storedSession, event); err != nil {
				return fmt.Errorf("failed to add user event to session: %w", err)
			}
		}
	}

	return nil
}

func (s *runnerLiveSession) Close() error {
	return s.sess.Close()
}

type closedLiveSession struct{}

func (s *closedLiveSession) Send(req agent.LiveRequest) error {
	return fmt.Errorf("session is closed")
}

func (s *closedLiveSession) Close() error {
	return nil
}

func (r *Runner) RunLive(ctx context.Context, userID, sessionID string, cfg agent.LiveRunConfig, opts ...RunOption) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	options := runOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	var storedSession session.Session
	getResp, err := r.sessionService.Get(ctx, &session.GetRequest{
		AppName:   r.appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		if !r.autoCreateSession {
			return nil, nil, err
		}
		createResp, err := r.sessionService.Create(ctx, &session.CreateRequest{
			AppName:   r.appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			return nil, nil, err
		}
		storedSession = createResp.Session
	} else {
		storedSession = getResp.Session
	}

	// msg is nil for Live run as it's streaming
	agentToRun, err := r.findAgentToRun(storedSession, nil)
	if err != nil {
		return nil, nil, err
	}

	lAgent, ok := agentToRun.(liveAgent)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s does not support Live Run", agentToRun.Name())
	}

	ctx = parentmap.ToContext(ctx, r.parents)
	ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
		StreamingMode: runconfig.StreamingModeBidi, // Live is always bidirectional streaming
		Live:          &cfg,
	})
	ctx = plugininternal.ToContext(ctx, r.pluginManager)

	var artifacts agent.Artifacts
	if r.artifactService != nil {
		artifacts = &artifactinternal.Artifacts{
			Service:   r.artifactService,
			SessionID: storedSession.ID(),
			AppName:   storedSession.AppName(),
			UserID:    storedSession.UserID(),
		}
	}

	var memoryImpl agent.Memory = nil
	if r.memoryService != nil {
		memoryImpl = &imemory.Memory{
			Service:   r.memoryService,
			SessionID: storedSession.ID(),
			UserID:    storedSession.UserID(),
			AppName:   storedSession.AppName(),
		}
	}

	iCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:   artifacts,
		Memory:      memoryImpl,
		Session:     storedSession,
		Agent:       agentToRun,
		UserContent: nil,
	})

	if r.pluginManager != nil {
		earlyExitResult, err := r.pluginManager.RunBeforeRunCallback(iCtx)
		if err != nil {
			return nil, nil, err
		}
		if earlyExitResult != nil {
			earlyExitEvent := session.NewEvent(iCtx.InvocationID())
			earlyExitEvent.Author = agentToRun.Name()
			earlyExitEvent.LLMResponse = model.LLMResponse{
				Content: earlyExitResult,
			}
			if err := r.sessionService.AppendEvent(iCtx, storedSession, earlyExitEvent); err != nil {
				return nil, nil, fmt.Errorf("failed to add event to session: %w", err)
			}

			earlyExitIter := func(yield func(*session.Event, error) bool) {
				yield(earlyExitEvent, nil)
			}
			return &closedLiveSession{}, earlyExitIter, nil
		}
	}

	agentSess, innerIter, err := lAgent.RunLive(iCtx)
	if err != nil {
		return nil, nil, err
	}

	wrappedIter := func(yield func(*session.Event, error) bool) {
		if r.pluginManager != nil {
			defer r.pluginManager.RunAfterRunCallback(iCtx)
		}

		var bufferedEvents []*session.Event
		isTranscribing := false

		for event, err := range innerIter {
			if err != nil {
				if !yield(nil, err) {
					return
				}
				continue
			}

			if r.pluginManager != nil {
				modifiedEvent, pluginErr := r.pluginManager.RunOnEventCallback(iCtx, event)
				if pluginErr != nil {
					if !yield(nil, pluginErr) {
						return
					}
					continue
				}
				if modifiedEvent != nil {
					event = modifiedEvent
				}
			}

			// Chronological event buffering logic for Live streaming.
			// Holds back tool calls/responses if they arrive before the transcription finishes.
			if event.LLMResponse.Partial && (event.LLMResponse.InputTranscription != nil || event.LLMResponse.OutputTranscription != nil) {
				isTranscribing = true
			}

			isToolCallOrResp := false
			if event.LLMResponse.Content != nil {
				for _, part := range event.LLMResponse.Content.Parts {
					if part.FunctionCall != nil || part.FunctionResponse != nil {
						isToolCallOrResp = true
						break
					}
				}
			}

			if isTranscribing && isToolCallOrResp {
				bufferedEvents = append(bufferedEvents, event)
				continue
			}

			if !event.LLMResponse.Partial {
				if event.LLMResponse.InputTranscription != nil || event.LLMResponse.OutputTranscription != nil {
					isTranscribing = false

					if err := r.sessionService.AppendEvent(iCtx, storedSession, event); err != nil {
						if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
							return
						}
						continue
					}
					if !yield(event, nil) {
						return
					}

					for _, bufferedEvent := range bufferedEvents {
						if err := r.sessionService.AppendEvent(iCtx, storedSession, bufferedEvent); err != nil {
							if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
								return
							}
							continue
						}
						if !yield(bufferedEvent, nil) {
							return
						}
					}
					bufferedEvents = nil
					continue
				}
			}

			if !event.LLMResponse.Partial && !hasInlineData(event) {
				if err := r.sessionService.AppendEvent(iCtx, storedSession, event); err != nil {
					if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
						return
					}
					continue
				}
			}

			if !yield(event, nil) {
				return
			}
		}
	}

	return &runnerLiveSession{
		sess:          agentSess,
		r:             r,
		iCtx:          iCtx,
		storedSession: storedSession,
	}, wrappedIter, nil
}

// compactionConfig returns the App's EventsCompactionConfig, or nil if none.
func (r *Runner) compactionConfig() *app.EventsCompactionConfig {
	if r.appCfg == nil {
		return nil
	}
	return r.appCfg.EventsCompactionConfig
}

func (r *Runner) appendMessageToSession(ctx agent.InvocationContext, storedSession session.Session, msg *genai.Content, saveInputBlobsAsArtifacts bool, pluginManager *plugininternal.PluginManager, stateDelta map[string]any) (agent.InvocationContext, error) {
	if msg == nil {
		return ctx, nil
	}
	if pluginManager != nil {
		modifiedMsg, err := pluginManager.RunOnUserMessageCallback(ctx, msg)
		if err != nil {
			return ctx, fmt.Errorf("error running on run user message callback : %w", err)
		}
		if modifiedMsg != nil {
			msg = modifiedMsg
			// update ctx user message
			ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Artifacts:    ctx.Artifacts(),
				Memory:       ctx.Memory(),
				Session:      ctx.Session(),
				Agent:        ctx.Agent(),
				UserContent:  msg,
				RunConfig:    ctx.RunConfig(),
				InvocationID: ctx.InvocationID(),
			})
		}
	}

	artifactsService := ctx.Artifacts()
	if artifactsService != nil && saveInputBlobsAsArtifacts {
		for i, part := range msg.Parts {
			if part.InlineData == nil {
				continue
			}
			fileName := fmt.Sprintf("artifact_%s_%d", ctx.InvocationID(), i)
			if _, err := artifactsService.Save(ctx, fileName, part); err != nil {
				return ctx, fmt.Errorf("failed to save artifact %s: %w", fileName, err)
			}
			// Replace the part with a text placeholder
			msg.Parts[i] = &genai.Part{
				Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", fileName),
			}
		}
	}

	event := session.NewEvent(ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: msg,
	}
	if stateDelta != nil {
		event.Actions.StateDelta = stateDelta
	}

	if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return ctx, fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return ctx, nil
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session session.Session, msg *genai.Content) (agent.Agent, error) {
	if event := handleUserFunctionCallResponse(session.Events(), msg); event != nil {
		subAgent := r.rootAgent.FindAgent(event.Author)
		if subAgent != nil {
			return subAgent, nil
		}
		log.Printf("Function call from an unknown agent: %s, event id: %s", event.Author, event.ID)
	}

	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)

		if event.Author == "user" {
			continue
		}

		subAgent := r.rootAgent.FindAgent(event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID)
			continue
		}

		if r.isTransferableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.rootAgent, nil
}

// handleUserFunctionCallResponse finds the function call event that matches the function response id
// delivered by the user in the latest event.
func handleUserFunctionCallResponse(events session.Events, msg *genai.Content) *session.Event {
	if events.Len() == 0 {
		return nil
	}

	functionResponses := utils.FunctionResponses(msg)
	if len(functionResponses) == 0 {
		return nil
	}

	// This assumes that even if user provides multiple function responses, all the function calls
	// were made by the same agent. Otherwise it would be impossible to rearrange session events
	// such that every function response has a corresponding call filtering by author.
	callID := functionResponses[0].ID
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		for _, part := range utils.FunctionCalls(event.Content) {
			if part.ID == callID {
				return event
			}
		}
	}
	return nil
}

// checks if the agent and its parent chain allow transfer up the tree.
func (r *Runner) isTransferableAcrossAgentTree(agentToRun agent.Agent) bool {
	for curAgent := agentToRun; curAgent != nil; curAgent = r.parents[curAgent.Name()] {
		llmAgent, ok := curAgent.(llminternal.Agent)
		if !ok {
			return false
		}

		if llminternal.Reveal(llmAgent).DisallowTransferToParent {
			return false
		}
	}

	return true
}

func hasInlineData(event *session.Event) bool {
	if event.LLMResponse.Content == nil {
		return false
	}
	for _, part := range event.LLMResponse.Content.Parts {
		if part.InlineData != nil {
			return true
		}
	}
	return false
}
