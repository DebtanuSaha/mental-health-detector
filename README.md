# Mental Health & Crisis Risk Detection — Text-Only Research Prototype

> **This is a research/screening prototype, not a clinical diagnostic tool.**
> It never claims to diagnose depression, suicidality, or any mental-health
> condition — see `docs/` (added in a later phase) for the full ethical/
> limitations statement.

## Status

- ✅ Phase 0 — Research (datasets, models, resources selected; see
  project notes / prior deliverable)
- ✅ Phase 1 — Dataset exploration
- ✅ Phase 2 — TF-IDF + Logistic Regression baseline
- ✅ Phase 3 — RoBERTa fine-tuning (real fine-tuning completed on full data)
- ✅ Phase 4 — Risk scoring
- ✅ Phase 5 — Explainability
- ✅ Phase 6 — Resource recommendation (this README section)
- ⬜ Phase 7 — FastAPI
- ⬜ Phase 8 — UI
- ⬜ Phase 9 — Tests (expanded beyond Phase 1's unit tests)

## Phase 1 — Dataset Exploration

### What this phase builds

- `ml/data/dataset_loader.py` — loads the Komati (suicide-watch) and
  Dreaddit raw CSVs into a normalized `(text, label, source)` schema.
  Each dataset's label keeps its own meaning (Komati: suicide/non-suicide;
  Dreaddit: stress/no-stress) — they are **not** merged into one shared
  label here. That mapping decision belongs to Phase 4 (risk scoring).
- `ml/data/dataset_statistics.py` — record counts, class distribution,
  missing-value counts, duplicate-text counts, character/word length
  distributions, and a small stratified sample-inspection helper.
- `scripts/run_phase1_exploration.py` — CLI that runs the above against
  either the real downloaded data or small synthetic fixtures.
- `tests/test_dataset_loader.py` — 9 unit tests covering normal loading,
  missing files, unknown label values, duplicate/missing detection, and
  stratified sampling.
- `configs/dataset_config.yaml` — all paths, column names, and label
  mappings live here, not hard-coded in the Python.

### Why it's needed

Before any model training, we need to actually know what's in the data:
class balance (is it skewed?), how long posts typically are (affects
tokenizer max-length choice in Phase 3), whether there are duplicates or
empty rows that would leak into train/test splits, and a small manual
read of real samples to catch anything the label schema doesn't capture.

### Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running it

**Against the real data** (after following `datasets/README.md` to
download both datasets):

```bash
python scripts/run_phase1_exploration.py
```

**Against small synthetic fixtures**, to check the pipeline itself works
before doing the real (larger, gated) downloads:

```bash
python scripts/run_phase1_exploration.py --use-fixtures
```

### Expected output

Console output for each dataset: total rows, missing text/label counts,
duplicate-text count, class distribution (counts + %), and character/word
length summary stats (mean/median/min/max/std). It also writes
`datasets/processed/<name>_sample_inspection.csv` — a small stratified,
truncated preview of real text per class, for manual review only (see
the privacy note in `datasets/README.md` — delete this file when done
reviewing it).

Example (from the synthetic fixtures — real-data numbers will differ
substantially, especially class balance and length distributions):

```
============================================================
Dataset: komati
============================================================
Total rows:              12
Missing/empty text:      0
Missing label:           0
Duplicate text rows:     0

Class distribution:
                1:       6 (50.0%)
                0:       6 (50.0%)
...
```

### Running the tests

```bash
pytest tests/ -v
```

All 9 tests should pass; they run entirely against
`tests/fixtures/sample_komati.csv` / `sample_dreaddit.csv` (small,
synthetic, non-graphic placeholder text), so no download is required
just to verify the code itself is correct.

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `DatasetLoadError: Komati raw file not found at ...` | Haven't downloaded the real CSV yet | Follow `datasets/README.md`, or run with `--use-fixtures` to test the pipeline first |
| `DatasetLoadError: ... expected column(s) [...] not found` | Downloaded a differently-structured version of the dataset | Open the CSV and check actual column names; update `text_column`/`label_column` in `configs/dataset_config.yaml` — don't hard-code a fix in the loader |
| `DatasetLoadError: komati: found label value(s) {...} not present in label_map` | The raw `class` column has a value other than `suicide`/`non-suicide` (e.g., a stray `depression` label from an older dataset version) | Add the missing value to `label_map` in the config with an explicit 0/1 mapping — don't guess in code |
| `ModuleNotFoundError: No module named 'yaml'` | Dependencies not installed | `pip install -r requirements.txt` |

### What's next (Phase 2)

TF-IDF + Logistic Regression baseline, trained on the Komati data with
proper train/val/test splitting (checking for user-level leakage where
possible) and class-imbalance handling, evaluated with precision/recall/
F1 per class — not accuracy alone — to give the project a real baseline
before RoBERTa fine-tuning in Phase 3.

## Phase 2 — TF-IDF + Logistic Regression Baseline

### What this phase builds

- `ml/preprocessing/text_cleaner.py` — conservative, config-driven text
  cleaning. Unicode/whitespace normalization and URL replacement are on
  by default; lowercasing, mention-stripping, and emoji-stripping are
  **off** by default, since punctuation, capitalization, and emojis can
  carry emotional signal in this domain (brief §11).
- `ml/splitting/dataset_split.py` — stratified train/val/test splitting
  (70/15/15 by default), plus a group-based splitter that's implemented
  and tested but currently unused: **neither Komati nor Dreaddit exposes
  a real author/user ID in the public CSV**, so group-based splitting
  isn't actually possible with the data as distributed — see the
  module docstring and `configs/training.yaml` for the full explanation.
  This is flagged rather than silently skipped, per the brief's leakage
  requirement (§12).
- `ml/models/baseline.py` — TF-IDF + Logistic Regression as a single
  sklearn `Pipeline` (vectorizer and classifier always saved/loaded
  together), with `class_weight="balanced"` as the default imbalance
  strategy (brief §13 — oversampling is deliberately NOT used by
  default; blind oversampling before a Transformer fine-tune (Phase 3)
  can distort embeddings, so it's left as a config option, not the
  default).
- `ml/training/evaluate.py` — precision/recall/F1 per class, macro F1,
  weighted F1, ROC-AUC, PR-AUC, confusion matrix, and false-negative /
  false-positive rate on the positive (crisis/stress) class specifically
  — never accuracy alone (brief §15).
- `ml/training/train_baseline.py` + `scripts/run_phase2_baseline.py` —
  orchestrates load → preprocess → report class distribution → split →
  fit (train only, to avoid TF-IDF vocabulary leakage) → evaluate → save
  model + metrics JSON. Dataset-agnostic — Komati or Dreaddit is a
  config/flag choice, not a code branch, since Phase 3 will need to run
  the same comparison.
- `tests/test_phase2_baseline.py` — 16 unit tests covering preprocessing,
  both splitters, pipeline fit/save/load round-tripping, and evaluation
  metrics (including a deliberately-wrong-predictions case to confirm
  false-negative-rate reporting works).

### Why it's needed

This is Experiment 1 from the brief's research-experiments design
(§16) — the baseline that RoBERTa (Phase 3) has to actually beat to
justify the extra complexity and compute. Without it, "RoBERTa gets
0.85 F1" is a number with nothing to compare against.

### Running it

```bash
pip install -r requirements.txt

# Real data (after datasets/README.md download steps):
python scripts/run_phase2_baseline.py                     # trains on komati (default)
python scripts/run_phase2_baseline.py --dataset dreaddit   # trains on dreaddit instead

# Synthetic fixtures (pipeline sanity check, no download needed):
python scripts/run_phase2_baseline.py --use-fixtures
python scripts/run_phase2_baseline.py --use-fixtures --dataset dreaddit
```

Outputs land in `models/baseline/`: the fitted pipeline
(`<dataset>_baseline_pipeline.joblib`) and a metrics JSON
(`<dataset>_baseline_metrics.json`) with train/val/test sizes, full
test-set metrics, and the config used — kept so Phase 3's RoBERTa run
can be compared against the exact same split ratios and preprocessing.

### Expected output

Console shows class distribution at each stage (full dataset → train →
val → test — watch these stay roughly consistent, which confirms
stratification worked), then the held-out test-set metrics block:
accuracy, macro/weighted F1, ROC-AUC, PR-AUC, per-class precision/
recall/F1, and the false-negative/false-positive rate on the positive
class with an explicit note that false negatives are the costlier
error here. **On the tiny 12–16 row synthetic fixtures these numbers
are not meaningful** — they're there to confirm the code runs, not to
represent real model quality; real numbers only mean something once
run against the actual downloaded datasets.

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: empty vocabulary; perhaps the documents only contain stop words` | TF-IDF's default token pattern drops single-character tokens; happens if preprocessing over-strips text or input is degenerate | Check `datasets/processed/*_sample_inspection.csv` from Phase 1 to confirm text looks reasonable; loosen `tfidf.min_df` in `configs/training.yaml` if it's a real (non-degenerate) small dataset |
| `ValueError: train/val/test ratios must sum to 1.0` | `split` ratios in `configs/training.yaml` were edited inconsistently | Make sure `train_ratio + val_ratio + test_ratio == 1.0` |
| `FileNotFoundError: No saved baseline pipeline at ...` | Trying to load a model before training it | Run `run_phase2_baseline.py` first |
| Test-set metrics look suspiciously perfect (1.0 everywhere) | Expected on the tiny synthetic fixtures (too few rows to be a real signal) — not expected on the real ~230K-row Komati data | If it happens on real data too, check for a data-leakage bug (e.g. duplicate rows across splits — Phase 1's duplicate-detection stats are the first thing to check) |

### What's next (Phase 3)

RoBERTa fine-tuning (`mental/mental-roberta-base`, benchmarked against
plain `roberta-base`) using the exact same split/preprocessing config
this baseline used, so the Phase 2 vs. Phase 3 comparison is apples-to-
apples.

### Recorded baseline result (real data, `dataset: komati`)

Run against the full 232,074-row real Komati dataset (70/15/15 split,
`min_df=5`, `max_features=40000`, `class_weight=balanced`):

| Metric | Value |
|---|---|
| Test set size | 34,812 |
| Accuracy | 0.9429 |
| Macro F1 | 0.9429 |
| ROC-AUC | 0.9846 |
| PR-AUC | 0.9838 |
| Precision (class 1, suicide) | 0.9507 |
| **Recall (class 1, suicide)** | **0.9342** |
| F1 (class 1, suicide) | 0.9424 |
| **False-negative rate (class 1)** | **6.58%** (1,145 / 17,406) |
| False-positive rate (class 0→1) | 4.84% (843 / 17,406) |

**This is the number Phase 3 has to beat — specifically recall/FNR on
class 1, not overall accuracy**, which is high enough here that it's
plausibly picking up some subreddit-sourced artifacts (posting
community/phrasing conventions) in addition to genuine crisis-language
signal, since labels come from subreddit origin rather than clinical
judgment. Before treating 94% as a ceiling that's purely about
crisis-language understanding, inspect the actual errors:

```bash
python scripts/export_baseline_errors.py
```

This reproduces the exact deterministic test split, loads the saved
model, and writes `datasets/processed/komati_false_negatives.csv` and
`komati_false_positives.csv` — truncated real-text previews of exactly
which posts the model got wrong, for manual read-through. (Same privacy
handling as Phase 1's sample inspection: local review only, delete when
done, never commit or share.)

### Recorded baseline result (real data, `dataset: dreaddit`)

Run against the real Dreaddit dataset (3,532 rows after deduping 21
exact-duplicate rows, 70/15/15 split):

| Metric | Value |
|---|---|
| Test set size | 530 |
| Accuracy | 0.7472 |
| Macro F1 | 0.7469 |
| ROC-AUC | 0.8423 |
| PR-AUC | 0.8544 |
| Precision (class 1, stress) | 0.7638 |
| **Recall (class 1, stress)** | **0.7473** |
| F1 (class 1, stress) | 0.7555 |
| **False-negative rate (class 1)** | **25.27%** (70/277) |
| False-positive rate (class 0→1) | 25.30% (64/253) |

Notably weaker than Komati (94% vs. 75% accuracy), and expected to be:
Dreaddit is a subtler task (expert-annotated stress vs. Komati's
coarser subreddit-origin labels) with ~30x less training data (2,472
vs. 162,451 rows). This actually reinforces the artifact caveat above
— if Komati's 94% were purely deep crisis-language understanding, it
should transfer more cleanly to Dreaddit's related task; the drop is a
soft signal that some of Komati's ease comes from coarser, more
learnable subreddit-level separability. The FNR jump (6.58% → 25.27%)
is the number to watch closing in Phase 3 — domain-pretrained
transfer learning tends to help most exactly when labeled data is this
scarce.

## Phase 3 — RoBERTa Fine-Tuning

### Architecture decision — a course-correction from Phase 0

Phase 0's research sketched a shared-encoder, two-head (sentiment +
crisis) architecture. In practice, **neither real dataset carries a
sentiment label**, and there's no single dataset with both a sentiment
AND a crisis/distress label to jointly train two heads from. Building
that architecture would mean fabricating sentiment pseudo-labels (a bad
idea for a screening tool) or alternating-task training across two
unrelated datasets purely for parameter-sharing's sake.

So Phase 3 fine-tunes a standard single-head
`AutoModelForSequenceClassification` **per dataset** — one crisis/
suicide classifier on Komati, one stress/distress classifier on
Dreaddit (same model class, different data, same code path via
`--dataset`). Sentiment as a secondary signal (the brief's own
requirement — "sentiment is not the crisis detector") is better served
by an off-the-shelf pretrained sentiment model called at inference time
in Phase 4's risk-scoring layer, not a jointly-trained head here.

### What this phase builds

- `ml/data/token_length_analysis.py` + `scripts/run_phase3_token_analysis.py`
  — tokenizes the real corpus with the actual model tokenizer and
  reports real BPE token-length percentiles and truncation rates at
  128/256/384/512 tokens, replacing the earlier word-count estimate.
- `ml/data/text_classification_dataset.py` — minimal torch `Dataset`
  wrapper for the HF `Trainer`.
- `ml/models/roberta_classifier.py` — tokenizer/model loading (Hub id
  or local path) and save/load, consistent interface with Phase 2's
  baseline wrapper.
- `ml/training/train_roberta.py` + `scripts/run_phase3_finetune.py` —
  full orchestration: load → **dedupe + split with the exact same
  config/seed Phase 2 used** (so the two experiments are directly
  comparable) → tokenize → fine-tune via HF `Trainer` with early
  stopping on validation macro-F1 → evaluate with the **same
  `evaluate_binary()`** Phase 2 used → save model + metrics JSON.
- `scripts/build_tiny_test_model.py` (dev utility, not production code)
  — builds a tiny, randomly-initialized, architecturally-real RoBERTa
  checkpoint entirely from local files under `tests/fixtures/tiny_roberta/`,
  used to smoke-test the real training code path without needing
  network access or the ~450MB real download.
- `tests/test_phase3_roberta.py` — 7 tests, including a full
  end-to-end fine-tune-and-evaluate run against the tiny checkpoint.

### Installation

```bash
pip install -r requirements.txt
```

This now includes `torch`, `transformers`, `datasets`, `accelerate`,
and `tokenizers`. First real run of `mental/mental-roberta-base` will
download ~450MB from the Hugging Face Hub.

### Running it

**1. Check real token-length distribution first** (finalizes `max_seq_length`):

```bash
python scripts/run_phase3_token_analysis.py                    # komati, 20K-row sample (fast)
python scripts/run_phase3_token_analysis.py --sample-size 0     # tokenize everything (slow, exact)
python scripts/run_phase3_token_analysis.py --dataset dreaddit
```

**2. Fine-tune:**

```bash
python scripts/run_phase3_finetune.py                          # komati, mental-roberta-base
python scripts/run_phase3_finetune.py --dataset dreaddit
python scripts/run_phase3_finetune.py --model roberta-base      # ablation (brief §16)
```

**2b. Or do a fast initial pass first on a class-balanced subsample** —
useful before committing to a full run, to confirm the pipeline works
end-to-end on real data and get a rough sense of per-epoch time on your
hardware:

```bash
python scripts/run_phase3_finetune.py --max-train-samples 20000
python scripts/run_phase3_finetune.py --max-train-samples 20000 --max-val-samples 3000
```

Subsampling preserves class balance (stratified, not a naive head/tail
slice). The run logs an explicit `REDUCED-SCOPE` reminder, and
`eval_steps`/`save_steps`/`logging_steps` auto-clamp down if the
subsample is small enough that the configured `eval_steps: 500` would
otherwise never trigger within the run. **Treat any subsampled run's
metrics as directional, not final** — re-run without `--max-train-samples`
(or set `roberta.max_train_samples: null` in the config) for the real
Phase 3 numbers once you've confirmed the pipeline and timing.

**3. Pipeline smoke test** (no download, no GPU needed, runs in seconds):

```bash
python scripts/run_phase3_finetune.py --use-fixtures
```

Outputs land in `models/roberta/<dataset>/`: the fine-tuned model +
tokenizer (standard HF format, reloadable via `from_pretrained`) and
`metrics.json` in the same shape as Phase 2's, for direct comparison.

### Expected output

Training logs loss/accuracy/macro-F1 at each `eval_steps` checkpoint,
then the same metrics block format as Phase 2 (accuracy, macro/
weighted F1, ROC-AUC, PR-AUC, per-class precision/recall/F1, false-
negative/positive rate). **On the tiny fixture + random-init checkpoint
these numbers are meaningless** — same caveat as every other phase's
fixture run, there purely to confirm the code runs correctly.

Real fine-tuning on the full Komati dataset will take considerably
longer than Phase 2's baseline (minutes-to-hours depending on your
hardware, especially without a GPU) — `roberta.num_epochs`,
`batch_size`, and early stopping in `configs/training.yaml` are there
to tune that trade-off.

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `OSError: We couldn't connect to 'https://huggingface.co'` | No internet access, or `mental/mental-roberta-base` blocked by a firewall/proxy | Check connectivity; the model only needs to download once and is then cached locally (`~/.cache/huggingface/`) |
| `GatedRepoError` / `401 Unauthorized` on `mental/mental-roberta-base` | **This model is gated** — confirmed the first time this project actually ran against it. It's a one-click accept, not an approval wait, but you do need to: (1) log into huggingface.co and accept the model's terms at its Hub page, (2) create a Read-scope token under Settings → Access Tokens, (3) run `huggingface-cli login` locally and paste it. The project's error handling (`ml/models/roberta_classifier.py`) now surfaces these steps directly instead of a raw traceback |
| `AttributeError: module 'datasets' has no attribute 'Dataset'` | The real Hugging Face `datasets` pip package isn't installed, so Python falls back to this project's own `datasets/` folder (raw/processed data) as a namespace package of the same name | `pip install datasets` — this is a real dependency now in `requirements.txt`; if it still happens, confirm you're not manually adding the repo root to `PYTHONPATH` ahead of site-packages |
| `ImportError: Using the 'Trainer' with 'PyTorch' requires 'accelerate>=1.1.0'` | `accelerate` missing/outdated | `pip install "accelerate>=1.1.0"` (already in `requirements.txt`) |
| `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | Some transformers versions moved/renamed this kwarg | Already handled — this project computes `warmup_steps` from `warmup_ratio` manually rather than passing `warmup_ratio` directly, specifically to avoid this |
| CUDA out-of-memory / very slow on CPU | Real Komati fine-tuning at `batch_size=16`, `max_seq_length=512` is a lot of compute | Lower `batch_size` in `configs/training.yaml` (use `gradient_accumulation_steps` if you need to keep effective batch size up — not yet exposed in the config, add it if needed), or lower `max_seq_length` once you've seen the real truncation-rate numbers from step 1 above |

### What's next (Phase 4)

Risk-scoring layer: combine the fine-tuned model's crisis-classification
output with an off-the-shelf pretrained sentiment model (per the
architecture decision above) into the configurable LOW/MODERATE/HIGH/
CRITICAL risk categories from the brief (§6), with thresholds in a
config file, not hard-coded.

## Phase 4 — Risk Scoring

Combines the crisis-classifier probability (Komati-trained) and
distress-classifier probability (Dreaddit-trained) into one
probabilistic `risk_score` and LOW/MODERATE/HIGH/CRITICAL category.
Sentiment (off-the-shelf `cardiffnlp/twitter-roberta-base-sentiment-latest`,
loaded lazily) is reported alongside but deliberately never fused into
the score — enforced by a real test, not just documented.

```bash
python scripts/run_phase4_risk_assessment.py --text "some text here"
python scripts/run_phase4_risk_assessment.py --text "..." --country IN
python scripts/run_phase4_risk_assessment.py --text "..." --prefer baseline
```

Thresholds, signal weights, and per-level safety messages all live in
`configs/risk_config.yaml` — explicitly flagged there as starting
points requiring real-world validation, not settled cutoffs. Prediction
automatically uses the fine-tuned RoBERTa model if available, falling
back to the Phase 2 baseline with a logged warning otherwise (via
`ml/inference/model_predictor.py`).

**Sentiment model network note:** if the sentiment model download hits
a Windows-specific `WinError 10022` (IPv6-related), see the fix: update
`httpx`/`httpcore`/`huggingface_hub`, or disable IPv6 on the active
adapter, or download the model files manually into a local folder and
point `sentiment.model_name` in the config at that folder path instead
of the Hub id.

## Phase 5 — Explainability

Token-level attribution via Captum's `LayerIntegratedGradients` for the
fine-tuned RoBERTa models, with automatic fallback to linear-coefficient
attribution (TF-IDF weight × Logistic Regression coefficient) when only
the Phase 2 baseline is available — mirrors `model_predictor.py`'s own
fallback design.

```bash
python scripts/run_phase5_explain.py --text "some text" --signal crisis
python scripts/run_phase5_explain.py --text "some text" --signal distress
```

Implementation verified empirically against this project's actual
`RobertaForSequenceClassification` architecture (not assumed): the
target embedding layer is `model.get_input_embeddings()` (confirmed
identical to `model.roberta.embeddings.word_embeddings`), and the
reference baseline keeps structural `<s>`/`</s>` tokens while replacing
content tokens with `pad_token_id`. Every result includes explicit
caveats — what the method shows, what it does NOT prove, and its
computational cost — per the brief's explainability requirements (§8).

## Phase 6 — Resource Recommendation

`resources/verified_resources.json` is the single, hand-sourced (Phase
0) location resource data lives — the model never generates or edits
crisis-resource entries. `ml/resources/resource_service.py` implements
the policy decided back in Phase 0: since text carries no reliable
location signal, a request with no explicit country returns only the
international directory (Findahelpline); a country is only used if
explicitly passed in — never inferred from the text. `LOW` risk never
returns resources at all (brief §18 — general wellbeing info only at
that level).

```bash
python scripts/run_phase6_resources.py --all
python scripts/run_phase6_resources.py --risk-level CRITICAL
python scripts/run_phase6_resources.py --risk-level CRITICAL --country IN
```

Phase 4's `assess_risk()` now actually populates the `resources` field
(previously always empty, intentionally, pending this phase) — pass
`country_code=` explicitly if you have one from outside the text itself
(e.g. a future UI's location field), otherwise leave it `None`.

## Privacy & Ethics (summary — expanded in a later phase)

- No raw user-submitted text is persisted by default anywhere in this
  system once the live API (Phase 7) exists.
- The datasets used here contain real, sensitive text from vulnerable
  populations; see `datasets/README.md` for handling guidance.
- Outputs are explicitly framed as probabilistic risk *screening*
  signals, never a diagnosis — see the master project brief for the
  exact language requirements enforced in later phases' API responses.
