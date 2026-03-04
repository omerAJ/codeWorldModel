# HumanEval Transition Dataset Tools

Generate and inspect line-level Python transition data for HumanEval/MBPP-style datasets.

## Scripts
- `dataPreparation/trace_generator.py`: traces execution and builds transition records.
- `dataPreparation/generate_humaneval_traces.py`: writes dataset JSONL (+ optional compaction).
- `dataPreparation/visualize_transition_dataset.py`: builds a local HTML viewer.
- `dataPreparation/run_transition_pipeline.py`: one-command tracked run (dataset + metadata + errors + viewer + run manifest).

## Record Fields
Core fields per row:
- `code_context`, `action`, `next_action`, `current_state`, `next_state`
- summary tokens: `<CTX_SUM>`, `<ACT_SUM>`, `<STATE_SUM>`
- optional (MBPP): `task_text` (problem statement from MBPP `text` field)

Control/quality fields:
- `line_no`, `next_line_no`, `next_event_type`, `changed_vars`
- `is_noop_state`, `is_branch_noop`, `is_pure_identity_noop`, `is_loop_progress_only`

## Generate
```bash
python3 dataPreparation/generate_humaneval_traces.py \
  --dataset-format humaneval \
  --limit 1 \
  --output data/humaneval_transition_1task.jsonl \
  --metadata-output data/humaneval_transition_1task.meta.json
```

## Compact (optional)
```bash
python3 dataPreparation/generate_humaneval_traces.py \
  --dataset-format humaneval \
  --output data/humaneval_transition_compact.jsonl \
  --max-examples-per-task 200 \
  --branch-noop-cap-per-path 2 \
  --loop-progress-cap-per-path 2 \
  --quota-state-changing 120 \
  --quota-branch-noop 40 \
  --quota-loop-progress 30 \
  --quota-identity-noop 10
```

## Knobs (simple)
Compaction is applied per task in this order:
1. category split (`state_changing`, `branch_noop`, `loop_progress`, `identity_noop`)
2. per-path caps
3. per-category quotas
4. final max-per-task overflow handling

Flag behavior:
- `--branch-noop-cap-per-path K`: keep at most `K` branch-noop rows for the same control-flow edge `(line_no -> next_line_no/event)`.
- `--loop-progress-cap-per-path K`: keep at most `K` loop-progress rows for the same edge.
- `--quota-state-changing N`: keep at most `N` state-changing rows per task.
- `--quota-branch-noop N`: keep at most `N` branch-noop rows per task.
- `--quota-loop-progress N`: keep at most `N` loop-progress rows per task.
- `--quota-identity-noop N`: keep at most `N` pure identity no-op rows per task.
- `--max-examples-per-task M`: hard upper bound per task after quotas.
- `--overflow-policy`: what happens if total rows still exceed `M`.
- `priority`: fill in category order (`state_changing` first, then branch, loop, identity).
- `proportional`: keep categories in roughly the same proportion.
- `error`: fail the task instead of trimming.
- `--sample-mode first|uniform`: when trimming, keep earliest rows or random rows.
- `--seed`: random seed used when `--sample-mode=uniform`.

Common effects:
- Lower path caps: removes repeated loop/branch transitions.
- Lower no-op quotas: pushes dataset toward state-changing rows.
- Lower `max-examples-per-task`: prevents single-task domination.
- `uniform` sampling: reduces temporal bias from always taking earliest trace steps.

## View
```bash
python3 dataPreparation/visualize_transition_dataset.py \
  --input data/humaneval_transition_1task.jsonl \
  --output data/humaneval_transition_1task.viewer.html
```
Open the generated `.viewer.html` in a browser.

## Tracked Runs (recommended)
```bash
python3 dataPreparation/run_transition_pipeline.py \
  --input /path/to/dataset.jsonl \
  --dataset-format auto
```
This creates `data/runs/<timestamp>_<dataset>/` with:
- `transition_dataset.jsonl`
- `transition_dataset.meta.json`
- `transition_dataset.errors.log`
- `transition_dataset.viewer.html`
- `run_manifest.json` (input, knobs, commands, timing, summary stats)

For MBPP:
```bash
python3 dataPreparation/run_transition_pipeline.py \
  --input /path/to/mbpp.jsonl \
  --dataset-format mbpp
```
