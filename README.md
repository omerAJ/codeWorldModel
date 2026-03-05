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

## Latent Transition Training (Maincoder-1B)

This repo now includes a configurable training stack under `src/cwmodel` for latent transition modeling:
- independent encoding passes for `code_context`, `action`, and `current_state`
- summary-token latent extraction (`<CTX_SUM>`, `<ACT_SUM>`, `<STATE_SUM>`)
- predictor head that outputs concatenated next latents (`next_state` + `next_action`)
- loss: `MSE(pred_next_state, target_next_state) + MSE(pred_next_action, target_next_action)`
- detached target latents computed from the same backbone on `next_state` and `next_action`

### 1) Environment setup (Conda)
```bash
conda env create -f environment.yml
conda activate cwmodel
```

### 2) Pull model snapshot locally
```bash
python scripts/pull_hf_snapshot.py \
  --model-id Maincode/Maincoder-1B \
  --local-dir models/maincode-maincoder-1b
```

This writes `models/maincode-maincoder-1b/snapshot_meta.json`.
To refresh later:
```bash
python scripts/pull_hf_snapshot.py \
  --model-id Maincode/Maincoder-1B \
  --local-dir models/maincode-maincoder-1b \
  --refresh
```

### 3) Train (1-epoch default config)
```bash
python scripts/train.py --config configs/train.maincoder_1b.yaml
```

Artifacts are written to `outputs/<run_name_or_timestamp>/`:
- `resolved_config.yaml`
- `run_summary.json`
- `metrics.jsonl`
- `tensorboard/`
- `checkpoints/step_*/`

### 4) Common config overrides
```bash
python scripts/train.py \
  --config configs/train.maincoder_1b.yaml \
  --set model.local_path=models/maincode-maincoder-1b \
  --set training.epochs=3 \
  --set data.batch_size=1 \
  --set output.run_name=maincoder_debug
```

### 5) AR benchmark eval with bigcode-evaluation-harness (LoRA adapters)
Clone harness once:
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness tools/bigcode-evaluation-harness
```

Then run evaluation from a compatible Python env (must include `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `safetensors`):
```bash
python scripts/eval_bigcode_harness.py \
  --harness-dir tools/bigcode-evaluation-harness \
  --run-dir outputs/run_20260304T103005Z \
  --checkpoint-step 30 \
  --tasks humaneval \
  --limit 1 \
  --n-samples 1 \
  --batch-size 1 \
  --max-length-generation 512 \
  --load-in-4bit \
  --trust-remote-code \
  --allow-code-execution \
  --save-generations
```

Notes:
- `--run-dir` + `--checkpoint-step` auto-resolves the adapter at `checkpoints/step_xxxxxx/backbone/`.
- If `--model` is omitted, it is inferred from `adapter_config.json`.
- The wrapper applies an adapter/base vocab-size compatibility fix before PEFT load.
- For full HumanEval pass@k runs, remove `--limit` and increase `--n-samples` (e.g. 200).

### Project layout
- `configs/train.maincoder_1b.yaml`: end-to-end defaults for local training
- `scripts/pull_hf_snapshot.py`: local-only HF pull/update flow
- `scripts/train.py`: config-driven training CLI
- `scripts/eval_bigcode_harness.py`: wrapper for AR benchmark eval via bigcode-evaluation-harness + PEFT adapters
- `src/cwmodel/data/*`: schema, dataset, collator
- `src/cwmodel/modeling/*`: HF setup, latent predictor, world model
- `src/cwmodel/training/*`: training loop and logging
- `tests/*`: token/dataset/forward/smoke coverage
