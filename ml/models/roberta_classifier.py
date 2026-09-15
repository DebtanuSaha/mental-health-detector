"""
roberta_classifier.py

Thin wrapper around Hugging Face's AutoTokenizer/AutoModelForSequenceClassification
for binary crisis/distress classification.

Design note (course-correction from the Phase 0 research writeup): Phase 0
sketched a shared-encoder, two-head (sentiment + crisis) architecture. In
practice, neither real dataset available (Komati, Dreaddit) carries a
sentiment label, and there is no single dataset here with both a
sentiment AND a crisis/distress label to jointly train two heads from.
Building a shared-encoder-two-head model without paired labels would mean
either fabricating sentiment pseudo-labels (bad idea for a screening
tool) or alternating-task training across two unrelated datasets purely
for parameter-sharing's sake — extra complexity without a clear benefit
at this stage.

So Phase 3 trains a standard single-head AutoModelForSequenceClassification
per dataset (one crisis/suicide classifier fine-tuned on Komati, one
stress/distress classifier fine-tuned on Dreaddit — same model class,
different data). Sentiment as a secondary signal (per the brief's own
"sentiment is not the crisis detector" requirement) is better served by
an off-the-shelf pretrained sentiment model called at inference time in
Phase 4's risk-scoring layer, not a jointly-trained head — that's a
decision for Phase 4, not implemented here.
"""

from __future__ import annotations

from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_tokenizer_and_model(
    model_name_or_path: str, num_labels: int = 2
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a tokenizer + sequence-classification model from a Hugging Face
    Hub model id (e.g. "mental/mental-roberta-base") or a local path
    (e.g. "tests/fixtures/tiny_roberta", or a previously-saved
    models/roberta/<dataset>/ checkpoint from a prior training run).

    Note: mental/mental-roberta-base is a GATED model on the Hub — it
    requires accepting the model's terms on its Hub page and
    authenticating locally (`huggingface-cli login`) before it can be
    downloaded, even though it's free and the accept step is instant
    (not an approval-wait). This is caught below and re-raised with
    those steps spelled out, instead of surfacing a raw 401 traceback.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
    except OSError as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            raise OSError(
                f"'{model_name_or_path}' is a gated model on the Hugging Face Hub. "
                f"To fix:\n"
                f"  1. Log in at huggingface.co, visit "
                f"https://huggingface.co/{model_name_or_path}, and accept the model's terms.\n"
                f"  2. Create a Read-scope token: Settings -> Access Tokens -> New token.\n"
                f"  3. Run `huggingface-cli login` locally and paste the token.\n"
                f"  4. Re-run this script.\n"
                f"(Original error: {e})"
            ) from e
        raise
    return tokenizer, model


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
