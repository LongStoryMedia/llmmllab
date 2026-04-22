Both audits are in. Total removable footprint is ~5,500 lines of inference code (plus UI). Here's the plan — 5 phases, each a commit, biggest fish last. **I want your go-ahead before I start because Phase 10 (ModelProfile) has a real product decision baked in.**
## The hit list

| Concept | Lines | Current reality | Action |
|---|---|---|---|
| ComplexityLevel, IntelligentRouting | 0 | don't exist | — |
| ComplexityEstimate | 7 | pure dead enum | **delete** |
| ComputationalRequirements | 10 | phantom enum, 5 refs in orphaned schemas | **delete** |
| RequiredCapability | 25 | only ref'd in orphaned test | **delete** |
| RecoveryStrategy | 5 | enum in OOM data model, no runtime use | **delete** |
| CrashPrevention | 100 + UI | toggles wired but the preallocation test code doesn't exist | **delete** |
| ContextWindowConfig | 150 | instantiated in defaults, never read | **delete** |
| RefinementConfig | 40 | flags never checked | **delete** |
| CircuitBreakerConfig | 390 + UI + 3 tests | UI saves settings that are never read at runtime | **delete** |
| ParameterOptimizationConfig | 520 + UI | only BINARY_SEARCH branch is real; rest is enum theatre | **delete** |
| `intelligent_oom_recovery.py` | 1487 | sklearn Ridge regression for retrying OOM — overkill | **replace with ~30-line retry** |
| DynamicTools (storage + generator) | 947 + DB | \"dynamic\" is a misnomer — just hydrates static specs | **delete** |
| ModelProfile (schema, DB, routers, UI) | ~800 + 64 import sites + 5 endpoints + 15 UI files | per-user customization of system_prompt + params | **collapse into Model** |
| GPUConfig | 250 | actually functional (tensor split UI works) | **KEEP** |

## Proposed phases

**Phase 6 — \"No-op deletions\" (safe)**
Delete schemas + generated models + any imports of: ComplexityEstimate, ComputationalRequirements, RequiredCapability, RecoveryStrategy, ContextWindowConfig, RefinementConfig, CrashPrevention. Zero behavior change. `./regenerate_models.sh` run once at end.

**Phase 7 — Circuit breaker purge**
Delete CircuitBreakerConfig schema, model, UI pane, tests, perplexity/repetition guard code in token generation. Remove `circuit_breaker` field from ModelProfile (temporarily — ModelProfile dies in P10).

**Phase 8 — Parameter-opt + OOM-recovery purge**
Delete ParameterOptimizationConfig, PerformanceParameter, ParameterTuningStrategy. Replace `intelligent_oom_recovery.py` with a ~30-line exponential-backoff retry helper. Delete `/app/oom_recovery_data/` persistence code and UI pane.

**Phase 9 — Dynamic tools purge**
Delete `composer/tools/dynamic/`, `db/dynamic_tool_storage.py`, SQL, and all references. Pipelines fall back to the static tool registry that's already there.

**Phase 10 — ModelProfile collapse (big one)**
This is the ~2,500-line phase and the one with the product decision:

> Option A: **Radical** — add `system_prompt` + `parameters` + `gpu_config` directly onto `Model` schema. One model = one catalog entry with everything baked in. **Lose per-user customization.** Request-level overrides work via the existing OpenAI-style `temperature` field.
>
> Option B: **Flatten** — keep per-user overrides but store them as a single JSON blob on `UserConfig` keyed by task name. No more `model_profiles` table, no `ModelProfile` type, no REST endpoints. Still 19 task types.

The audit recommends B; your stated goal (\"configure hyperparameters on the model directly\") points at A. **A is much simpler; B preserves per-user tuning.** If you don't actually use per-user profiles (just you using the system), A is the right answer.

Also in P10: drop `ModelProfileType`'s 19-value enum (Primary, PrimarySummary, MasterSummary, BriefSummary, KeyPoints, SelfCritique, Improvement, MemoryRetrieval, Analysis, ResearchTask, ResearchPlan, ResearchConsolidation, ResearchAnalysis, Embedding, Formatting, ImageGenerationPrompt, Engineering, Reranking, ImageGeneration). Most workflows use 3-4 of these in practice. I'd collapse to: `chat`, `embedding`, `image_generation`, `reranking` — the rest get absorbed into `chat`.

## Risks / decisions I need from you

1. **Option A or B for ModelProfile?** (I recommend A given your goals. Say \"A\".)
2. **Model task types — collapse to 4?** (chat/embedding/image/reranking). Or keep more?
3. **UI cleanup** — do you want me to delete the ~1,400 lines of Settings UI (CircuitBreaker, ParameterOptimization, eventually ModelProfile pages) in these same phases, or punt UI to a separate branch? **(Punt is safer; UI builds independently.)**
4. **DB migration for existing deployments** — do you care about preserving existing `model_profiles` rows / `circuit_breaker` columns, or is this a clean-break personal project where we can drop tables?
5. **Commit style** — one per phase still? The ModelProfile phase might need 2-3 sub-commits because it touches so much.
6. **Checkpoint between P9 and P10?** I'd like to pause there so you can eyeball phases 6-9 before I start the big one.

If you say \"A, 4 types, punt UI, drop tables, one-commit-per-phase, checkpoint at P10\" I can just go. Otherwise tell me where to adjust.

--

--

## Checkpoint: Phases 6-9 complete

**Branch `simplify/cleanup` vs main**: 10 commits, 310 files changed, **-75,269 lines / +5,363 lines (~−70k net)**.

| Phase | Commit | Impact | What |
|---|---|---|---|
| 6 | `7cb54421` | -3,216 net | Delete `ComplexityEstimate`, `ComplexityLevel`, `ComputationalRequirement`, `RequiredCapability`, `RecoveryStrategy`, `AnalysisDepth`, `ToolAnalysisRequest`, `RoutingStrategy`, `CapabilityProfileMapping`, `ContextWindowConfig`, `RefinementConfig`, `IntentAnalysis` + `analysis_storage` DB table + intent-analysis docs |
| 7 | `2d0fa4f3` | -2,059 net | Delete `CircuitBreakerConfig` + perplexity/repetition guard refs + SQL column + UI pane |
| 8 | `192917e3` | -5,116 net | Delete `ParameterOptimizationConfig`, `PerformanceParameter`, `ParameterTuningStrategy`, `CrashPrevention`; delete `intelligent_oom_recovery.py` (1,487-line sklearn Ridge regression stack); `pipeline_cache` just fails on OOM now; delete UI `ParameterOptimizationSettings` + `ModelProfilesPage` (couldn't compile without the types) |
| 9 | `04ec683f` | -2,791 net | Delete entire `inference/composer/tools/dynamic/` + `dynamic_tool_storage` + SQL table + `engineering_agent` (whose only job was generating dynamic tools); rewrite `registry.py` 560→80 lines |

## What's still there heading into Phase 10

- `ModelProfile`, `ModelProfileType` (19 enum values), `ModelProfileConfig`, DB `model_profiles` table + 5 REST endpoints
- `GPUConfig` — still present, will be killed in P10 per your direction
- `ModelParameters` — needs to fold into `Model`
- The remaining **64 Python files** that reference `ModelProfile`
- `DialogGraphBuilder` still fetches 3 profiles by task type; will either get collapsed or wired to `Model` directly

## Known state for P10

- `inference/models/config.py` is **manually maintained** (schema-regen is broken by pre-existing missing `preferences_config.yaml` / `web_search_config.yaml` refs — the generator skips config.yaml). I'll need to keep manual-editing this file in P10 or fix those missing schemas.
- A handful of pre-existing Pyright errors remain (e.g. `get_model_profile_for_task` arg-type, `state.get` on WorkflowState) — carried over, not introduced.
- UI builds should still work after P8/P9 (Router + Navigation pruned to match).

Ready for Phase 10 when you are. Say \"go\" and I'll start the ModelProfile collapse per Option A: push `system_prompt` + hyperparameters onto `Model` directly, collapse 19 task types to {text, embedding, image, reranking, audio-in, audio-out, image-to-3d}, nuke `model_profiles` table + `/profiles` endpoints + UI, and rewrite the workflow builders to take a `Model` instead.