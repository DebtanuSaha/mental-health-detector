"""
train_roberta.py

Orchestrates Phase 3: load a dataset (Phase 1 loader), dedupe + light
preprocess (same functions Phase 2 used — for split/leakage parity, NOT
because Transformers need TF-IDF-style cleaning), split train/val/test
with the SAME config/seed as the baseline (so the two experiments are
comparable), tokenize, fine-tune with the Hugging Face Trainer
(early stopping on validation macro-F1), evaluate on held-out test data
with the exact same evaluate_binary() used for the Phase 2 baseline, and
save the model + metrics JSON.

Note on preprocessing for a Transformer: unlike TF-IDF, RoBERTa's
subword tokenizer handles punctuation/capitalization/URLs natively and
doesn't strictly need the same cleaning a bag-of-words model does. The
same light `text_cleaner` config is still applied here for one concrete
reason — keeping the exact same train/val/test SPLIT (same rows in same
splits) as Phase 2, since preprocessing happens before splitting in both
pipelines. If Phase 2's preprocessing config changes, re-run both phases
together, not just one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from ml.data.dataset_loader import load_config as load_dataset_config
from ml.data.text_classification_dataset import TextClassificationDataset
from ml.models.roberta_classifier import load_tokenizer_and_model, save_model
from ml.preprocessing.text_cleaner import PreprocessConfig, clean_series
from ml.splitting.dataset_split import group_split, stratified_split, stratified_subsample
from ml.training.evaluate import evaluate_binary, print_evaluation
from ml.training.train_baseline import _load_dataset, load_training_config

logger = logging.getLogger(__name__)


def _tokenize(texts, tokenizer, max_length: int) -> dict:
    # padding=False here deliberately: padding every example to a fixed
    # max_length wastes most of the compute on padding tokens, since the
    # real data's median length (~80 tokens) is far below max_length
    # (512) for the vast majority of posts. Padding is instead done
    # dynamically per-batch by DataCollatorWithPadding in run_finetuning()
    # below — each batch only pads to its own longest example, not to
    # 512 every time. truncation still applies per-example at max_length.
    return dict(
        tokenizer(
            list(texts),
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    )


def _compute_metrics_fn(eval_pred):
    """HF Trainer callback: returns a flat dict (Trainer requires this shape);
    the richer evaluate_binary() report is generated separately after training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import f1_score, accuracy_score

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def run_finetuning(
    training_config_path: str | Path = "configs/training.yaml",
    dataset_config_path: str | Path = "configs/dataset_config.yaml",
    base_dir: str | Path = ".",
    model_name_or_path_override: str | None = None,
    dataset_override: str | None = None,
) -> dict:
    base_dir = Path(base_dir)
    train_cfg = load_training_config(base_dir / training_config_path)
    dataset_cfg = load_dataset_config(base_dir / dataset_config_path)

    roberta_cfg = train_cfg["roberta"]
    dataset_name = dataset_override or roberta_cfg["dataset"]
    model_name_or_path = model_name_or_path_override or roberta_cfg["model_name_or_path"]

    loaded = _load_dataset(dataset_name, dataset_cfg, base_dir)
    df = loaded.df.copy()
    logger.info("Loaded '%s': %d rows", dataset_name, len(df))

    # --- Dedup + preprocess (same as Phase 2 — see module docstring) ---
    if train_cfg.get("preprocessing", {}).get("drop_duplicate_text", True):
        before = len(df)
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
        if len(df) < before:
            logger.info("Dropped %d duplicate-text rows before splitting", before - len(df))

    pp_cfg = PreprocessConfig.from_dict(train_cfg["preprocessing"])
    df["text"] = clean_series(df["text"], pp_cfg)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # --- Split (identical config/seed to Phase 2 baseline) ---
    split_cfg = train_cfg["split"]
    if split_cfg["group_based"]:
        result = group_split(df, group_col=split_cfg["group_column"], label_col="label",
                              train_ratio=split_cfg["train_ratio"], val_ratio=split_cfg["val_ratio"],
                              test_ratio=split_cfg["test_ratio"], random_seed=split_cfg["random_seed"])
    else:
        result = stratified_split(df, label_col="label",
                                   train_ratio=split_cfg["train_ratio"], val_ratio=split_cfg["val_ratio"],
                                   test_ratio=split_cfg["test_ratio"], random_seed=split_cfg["random_seed"])

    # --- Optional stratified subsampling for a fast initial pass ---
    # (class balance preserved; see configs/training.yaml roberta.max_*_samples)
    seed = roberta_cfg["seed"]
    subsampled_any = False
    if roberta_cfg.get("max_train_samples"):
        before = len(result.train)
        result.train = stratified_subsample(result.train, roberta_cfg["max_train_samples"], "label", seed)
        if len(result.train) < before:
            logger.info("Subsampled train: %d -> %d rows", before, len(result.train))
            subsampled_any = True
    if roberta_cfg.get("max_val_samples"):
        before = len(result.val)
        result.val = stratified_subsample(result.val, roberta_cfg["max_val_samples"], "label", seed)
        if len(result.val) < before:
            logger.info("Subsampled val: %d -> %d rows", before, len(result.val))
    if roberta_cfg.get("max_test_samples"):
        before = len(result.test)
        result.test = stratified_subsample(result.test, roberta_cfg["max_test_samples"], "label", seed)
        if len(result.test) < before:
            logger.info("Subsampled test: %d -> %d rows", before, len(result.test))
    if subsampled_any:
        logger.info(
            "NOTE: this is a REDUCED-SCOPE run on a subsample of the real data — "
            "label any results accordingly, don't treat them as the final Phase 3 numbers."
        )

    for name, split_df in (("train", result.train), ("val", result.val), ("test", result.test)):
        counts = split_df["label"].value_counts().to_dict()
        logger.info("Class distribution (%s, n=%d): %s", name, len(split_df),
                     {str(k): v for k, v in counts.items()})

    # --- Load tokenizer + model ---
    logger.info("Loading model: %s", model_name_or_path)
    tokenizer, model = load_tokenizer_and_model(model_name_or_path, num_labels=2)

    max_length = roberta_cfg["max_seq_length"]
    train_encodings = _tokenize(result.train["text"], tokenizer, max_length)
    val_encodings = _tokenize(result.val["text"], tokenizer, max_length)
    test_encodings = _tokenize(result.test["text"], tokenizer, max_length)

    train_dataset = TextClassificationDataset(train_encodings, result.train["label"].tolist())
    val_dataset = TextClassificationDataset(val_encodings, result.val["label"].tolist())
    test_dataset = TextClassificationDataset(test_encodings, result.test["label"].tolist())

    output_dir = base_dir / roberta_cfg["output_dir"] / dataset_name

    # warmup_steps computed from warmup_ratio rather than passing
    # warmup_ratio directly to TrainingArguments — that kwarg has moved/
    # been renamed across transformers versions; warmup_steps is the
    # more stable, broadly-supported option.
    steps_per_epoch = max(1, -(-len(train_dataset) // roberta_cfg["batch_size"]))  # ceil division
    total_training_steps = steps_per_epoch * roberta_cfg["num_epochs"]
    warmup_steps = int(roberta_cfg["warmup_ratio"] * total_training_steps)

    # Clamp eval/save/logging step intervals to the actual number of
    # training steps — otherwise a small subsample (see
    # roberta.max_train_samples) could mean eval_steps=500 never
    # triggers within a short run, so early stopping and checkpointing
    # silently never fire.
    eval_steps = max(1, min(roberta_cfg["eval_steps"], total_training_steps))
    logging_steps = max(1, min(roberta_cfg["logging_steps"], total_training_steps))
    if eval_steps != roberta_cfg["eval_steps"]:
        logger.info(
            "Clamped eval_steps %d -> %d to fit total_training_steps=%d (small dataset/subsample)",
            roberta_cfg["eval_steps"], eval_steps, total_training_steps,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=roberta_cfg["num_epochs"],
        per_device_train_batch_size=roberta_cfg["batch_size"],
        per_device_eval_batch_size=roberta_cfg["eval_batch_size"],
        learning_rate=roberta_cfg["learning_rate"],
        weight_decay=roberta_cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=roberta_cfg["fp16"],
        seed=roberta_cfg["seed"],
        report_to=[],  # no wandb/tensorboard by default — keep this local-only
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=roberta_cfg["early_stopping_patience"])],
    )

    logger.info("Starting fine-tuning: %d train / %d val rows, max_length=%d", len(train_dataset), len(val_dataset), max_length)
    trainer.train()

    # --- Evaluate on held-out test set with the SAME metrics as the baseline ---
    test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_proba = _softmax_positive_class(logits)
    y_true = np.array(result.test["label"].tolist())

    eval_result = evaluate_binary(y_true, y_pred, y_proba, positive_label=1)
    print_evaluation(eval_result, positive_label_name=loaded.positive_label_name)

    # --- Save model + tokenizer + metrics ---
    model_path = save_model(trainer.model, tokenizer, output_dir)
    logger.info("Saved fine-tuned model to %s", model_path)

    metrics = {
        "dataset": dataset_name,
        "model_name_or_path": model_name_or_path,
        "n_train": len(result.train),
        "n_val": len(result.val),
        "n_test": len(result.test),
        "max_seq_length": max_length,
        "test_metrics": eval_result.to_dict(),
        "config": {k: v for k, v in roberta_cfg.items()},
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    return metrics


def _softmax_positive_class(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]
