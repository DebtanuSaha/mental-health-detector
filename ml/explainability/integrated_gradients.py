"""
integrated_gradients.py

Token-level attribution for the fine-tuned RoBERTa crisis/distress
classifiers, via Captum's LayerIntegratedGradients — the Phase 0
research decision (cheaper per-request than SHAP/LIME: one extra
backward pass instead of many resampled forward passes).

Implementation notes specific to this project's HF
RobertaForSequenceClassification models (verified empirically against
the project's own tokenizer/model, not assumed from memory):

- Target layer: model.get_input_embeddings() — the standard,
  version-stable HF API for the input embedding layer. Confirmed to be
  the exact same object as model.roberta.embeddings.word_embeddings for
  this project's checkpoints. Using get_input_embeddings() rather than
  hardcoding the module path is deliberate: it's the documented, stable
  API and doesn't depend on RoBERTa's internal attribute naming holding
  across transformers versions.
- Baseline (reference) input: structural tokens — the sequence-start
  token and sequence-end token — are kept as-is; every other (content)
  token position is replaced with pad_token_id. Standard "content
  removed, structure kept" reference used in Captum's own
  transformer-explainability examples.
- RoBERTa convention check: this project's tokenizer was empirically
  confirmed to leave cls_token_id/sep_token_id as None and rely on
  bos_token_id/eos_token_id instead (RoBERTa's actual convention, as
  opposed to BERT's cls/sep). The code below checks cls/sep first and
  falls back to bos/eos, to stay correct across either convention if a
  different RoBERTa checkpoint sets both.
- Target class: attributions are computed with respect to the POSITIVE
  class logit (index 1 — "suicide"/"stress" depending on which model),
  since that's the probability risk_scoring.py actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from captum.attr import LayerIntegratedGradients

IG_CAVEATS = [
    "Token attributions show which words most influenced THIS model's prediction, given its current parameters — not an independent judgment of which words matter.",
    "This does NOT prove the model's reasoning is clinically valid, or that highlighted words are inherently significant outside this model.",
    "Scores are relative to a padding-token reference baseline and can shift with a different baseline choice or model version.",
    "Computed via Integrated Gradients (Captum) — requires an additional backward pass through the model beyond the forward pass used for the risk score itself, so this is meaningfully more expensive per request than the base prediction alone.",
]


@dataclass
class TokenAttribution:
    token: str
    score: float


@dataclass
class IntegratedGradientsResult:
    target_class: int
    predicted_proba: float
    token_attributions: list  # list[TokenAttribution], in original token order
    caveats: list


def _build_baseline_input_ids(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Structural tokens (bos/cls, eos/sep) kept as-is; content tokens -> pad_token_id."""
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

    ref = input_ids.clone()
    for i in range(ref.shape[1]):
        tok_id = ref[0, i].item()
        if tok_id in (cls_id, sep_id, pad_id):
            continue
        ref[0, i] = pad_id
    return ref


def explain_with_integrated_gradients(
    text: str,
    tokenizer,
    model,
    target_class: int = 1,
    n_steps: int = 50,
    max_length: int = 512,
) -> IntegratedGradientsResult:
    """
    Compute per-token Integrated Gradients attributions for `text`
    against `target_class` (default 1 = the positive/crisis-or-distress
    class), using the tokenizer/model from an already-loaded
    RobertaPredictor (ml/inference/model_predictor.py).
    """
    model.eval()
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    baseline_input_ids = _build_baseline_input_ids(input_ids, tokenizer)

    def forward_func(input_ids_, attention_mask_):
        logits = model(input_ids=input_ids_, attention_mask=attention_mask_).logits
        return torch.softmax(logits, dim=-1)[:, target_class]

    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())

    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # attributions shape: (1, seq_len, embedding_dim) -> sum over embedding dim per token
    token_scores = attributions.sum(dim=-1).squeeze(0)
    norm = token_scores.norm()
    if norm > 0:
        token_scores = token_scores / norm  # normalize to a comparable scale across texts

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_attributions = [
        TokenAttribution(token=tok, score=round(float(score), 4))
        for tok, score in zip(tokens, token_scores.tolist())
    ]

    with torch.no_grad():
        proba = torch.softmax(
            model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1
        )[0, target_class]

    return IntegratedGradientsResult(
        target_class=target_class,
        predicted_proba=round(float(proba), 4),
        token_attributions=token_attributions,
        caveats=IG_CAVEATS,
    )
