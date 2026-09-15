# Raw datasets

This directory is intentionally **empty of real data** in the delivered
project — the raw CSVs are large (hundreds of MB) and, in Komati's case,
gated behind a Kaggle account. Download them yourself into the paths
below; `configs/dataset_config.yaml` already points at these locations.

## 1. Komati — "Suicide and Depression Detection"

Target path: `datasets/raw/komati/suicide_detection.csv`

```bash
pip install kaggle
# requires a Kaggle account + API token at ~/.kaggle/kaggle.json
# (Account -> Create New Token on kaggle.com)
kaggle datasets download -d nikhileswarkomati/suicide-watch -p datasets/raw/komati
unzip datasets/raw/komati/suicide-watch.zip -d datasets/raw/komati
```

The unzipped file may be named `Suicide_Detection.csv` — rename it to
`suicide_detection.csv`, or update `komati.raw_path` in
`configs/dataset_config.yaml` to match whatever it's actually called.

Expected columns: `Unnamed: 0, text, class` (the loader only needs
`text` and `class`).

## 2. Dreaddit

Target paths:
`datasets/raw/dreaddit/dreaddit-train.csv`
`datasets/raw/dreaddit/dreaddit-test.csv`

Official author-hosted source (Columbia University):

```bash
wget https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip -O datasets/raw/dreaddit/dreaddit.zip
unzip datasets/raw/dreaddit/dreaddit.zip -d datasets/raw/dreaddit
```

If that host is ever unreachable, the same files are mirrored (unofficially)
on Hugging Face at `andreagasparini/dreaddit` and on Kaggle at
`ruchi798/stress-analysis-in-social-media` — prefer the Columbia source
when it's up, since it's the canonical release the paper describes.

Expected columns include `text`, `label` (0/1), plus post metadata and
100+ `lex_liwc_*` linguistic-feature columns (the loader keeps only
`text`/`label` unless you pass `keep_liwc_features=True`).

## Verifying your download

Once both files are in place, run:

```bash
python scripts/run_phase1_exploration.py
```

This prints class distribution, missing/duplicate counts, and text-length
stats for both datasets, and writes a small stratified sample-inspection
CSV to `datasets/processed/` for manual review.

To sanity-check the pipeline code itself *before* downloading anything
(e.g., to confirm your environment is set up correctly), run it against
the small synthetic fixtures instead:

```bash
python scripts/run_phase1_exploration.py --use-fixtures
```

## Privacy

These datasets contain real, sensitive, user-generated text from people
discussing mental-health struggles. Treat `datasets/raw/` and
`datasets/processed/` as sensitive local data:

- Both are already covered by `.gitignore` — never commit them.
- Don't re-upload or re-share the raw CSVs.
- `datasets/processed/*_sample_inspection.csv` contains truncated raw text
  previews for manual review — delete it when you're done inspecting,
  don't leave it lying around or attach it to bug reports/screenshots.
