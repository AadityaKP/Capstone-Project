# Oracle v4 Implementation Breakdown

## Scope completed

This repo now includes the full `oracle_v4` and `oracle_v4_causal` implementation described in the supplied guide.

The following spec items were applied:

1. `oracle/schemas.py` full replacement
2. `oracle/context.py` full replacement
3. `oracle/memory.py` full replacement
4. `oracle/graph_store.py` new file
5. `oracle/oracle.py` full replacement
6. `oracle/prompt_builder.py` full replacement
7. `boardroom/boardroom.py` targeted causal-graph changes
8. `simulation_runner.py` targeted policy-registration and `end_episode()` changes

Note: the guide referenced `agents/simulation_runner.py`, but this codebase uses the root-level [simulation_runner.py](/C:/College/Capstone/CapstoneProject/simulation_runner.py), so the integration was applied there.

## What changed

### 1. Schema layer

Updated [oracle/schemas.py](/C:/College/Capstone/CapstoneProject/oracle/schemas.py) to include:

- `GraphShockRecord`
- `CausalChainSummary`
- `GraphContext`

These support passing causal graph evidence from the oracle into prompt generation.

### 2. Context helpers

Updated [oracle/context.py](/C:/College/Capstone/CapstoneProject/oracle/context.py) with new tier classifiers:

- `get_mrr_tier()`
- `get_churn_tier()`
- `get_innovation_tier()`

These are now reused by both memory retrieval and causal graph storage.

### 3. Memory retrieval fixes

Replaced [oracle/memory.py](/C:/College/Capstone/CapstoneProject/oracle/memory.py) with the full v4 logic:

- Per-episode recency decay using `episode_global_start`
- Shorter recency horizon via `RECENCY_DECAY_MONTHS = 30`
- Enriched memory documents and queries with phase/tier labels
- Outcome-alignment reranking bonus using current MRR trend
- Suppression of trivial startup states with `source_month < 3`

### 4. Oracle core updates

Replaced [oracle/oracle.py](/C:/College/Capstone/CapstoneProject/oracle/oracle.py) to add:

- `oracle_v4` and `oracle_v4_causal` memory behavior
- Episode-relative recency tracking
- Cache key without memory-signature poisoning
- Graph store initialization for `oracle_v4_causal`
- 4-value `get_context()` return:
  `trend_context, memories, current_global_month, graph_context`
- `end_episode(episode_metrics=...)` graph write support

### 5. Prompt enrichment

Replaced [oracle/prompt_builder.py](/C:/College/Capstone/CapstoneProject/oracle/prompt_builder.py) to add:

- `graph_context` input
- `CAUSAL GRAPH CONTEXT` section for `oracle_v4_causal`
- `oracle_v4` and `oracle_v4_causal` support in trend and memory blocks

### 6. Causal graph storage

Added [oracle/graph_store.py](/C:/College/Capstone/CapstoneProject/oracle/graph_store.py) with:

- Episode node writes
- Shock-event writes
- Deferred outcome writes
- Historical similar-shock queries
- Aggregated causal-chain summaries
- Graceful disable behavior when Neo4j is unavailable

### 7. Boardroom integration

Updated [boardroom/boardroom.py](/C:/College/Capstone/CapstoneProject/boardroom/boardroom.py) to:

- Track `_pending_outcome_writes`
- Reset pending writes at episode start
- Pass `active_shock_label` into `oracle.get_context()`
- Forward `graph_context` into `generate_brief()`
- Write shock events immediately for `oracle_v4_causal`
- Write outcomes after the 6-month maturation window

### 8. Simulation runner integration

Updated [simulation_runner.py](/C:/College/Capstone/CapstoneProject/simulation_runner.py) to:

- Register `oracle_v4`
- Register `oracle_v4_causal`
- Pass `episode_metrics=result` into `oracle.end_episode()`

## Important integration note

I verified all current `get_context()` unpack sites before changing the signature. In this repo, the only live unpack callers were inside:

- [boardroom/boardroom.py](/C:/College/Capstone/CapstoneProject/boardroom/boardroom.py)
- [oracle/oracle.py](/C:/College/Capstone/CapstoneProject/oracle/oracle.py)

No additional project call sites needed updates.

## Repo-specific hardening applied

Two small integration adjustments were made so the causal graph path behaves correctly in this repo:

1. `simulation_runner.py` was patched instead of `agents/simulation_runner.py` because that is the actual runner used here.
2. [oracle/graph_store.py](/C:/College/Capstone/CapstoneProject/oracle/graph_store.py) merges an `Episode` node before shock writes and keys episode persistence off `seed` when available, so shock events do not no-op before `end_episode()` runs and their later outcome writes attach to the same episode identity.
3. [oracle/weight_adapter.py](/C:/College/Capstone/CapstoneProject/oracle/weight_adapter.py) now treats `oracle_v4` and `oracle_v4_causal` like `oracle_v3` for confidence scaling and decline-response weighting, so the richer briefs influence decision weights the same way the existing memory-aware oracle already did.

## Environment and runtime notes

Created a local `.env` with Neo4j placeholders:

- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=your_password`

`neo4j` was already present in [requirements.txt](/C:/College/Capstone/CapstoneProject/requirements.txt), so no dependency file update was needed.

`oracle_v4_causal` will still run if Neo4j is down. It degrades to functional `oracle_v4` behavior until the connection is available.

For the ablation, use `oracle_frequency=3`. That remains important for the memory improvements to show up in retrieval volume.

## Validation performed

Validation completed:

- Syntactic validation with `py_compile` on all changed Python files
- Import check for project dependencies by using the repo `site-packages`
- Policy build smoke test for:
  - `oracle_v4`
  - `oracle_v4_causal`
- Oracle initialization smoke test for:
  - `Oracle(mode="oracle_v4")`
  - `Oracle(mode="oracle_v4_causal")`

Observed runtime behavior:

- `oracle_v4_causal` correctly soft-disabled Neo4j when `localhost:7687` was unavailable
- Chroma emitted telemetry connection warnings because outbound network is restricted in this environment

## Remaining setup before causal data appears

To get actual graph-backed causal retrieval, you still need:

1. A reachable Neo4j instance
2. Valid `NEO4J_*` credentials in `.env`
3. Enough `oracle_v4_causal` episodes for the graph to accumulate outcome history

Without that setup, the policy still runs safely as non-graph `oracle_v4`.
