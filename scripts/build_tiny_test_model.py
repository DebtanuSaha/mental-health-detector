"""
build_tiny_test_model.py (dev utility — NOT part of the production pipeline)

Builds a tiny, randomly-initialized RoBERTa-architecture checkpoint saved
entirely to local disk under tests/fixtures/tiny_roberta/, so the Phase 3
training/eval code can be smoke-tested with real
`AutoTokenizer.from_pretrained(path)` /
`AutoModelForSequenceClassification.from_pretrained(path)` calls against a
*local path* — no huggingface.co access required.

This exists because the actual `mental/mental-roberta-base` checkpoint
can only be fetched from the Hugging Face Hub, which isn't reachable in
this build environment. The training/eval CODE PATH is identical whether
it's pointed at this tiny local checkpoint or the real one — only the
config value `roberta.model_name_or_path` changes.

Run once (already run — output is committed under tests/fixtures/):
    python scripts/build_tiny_test_model.py
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "tiny_roberta"

TRAINING_TEXT = [
    "I feel hopeless and empty inside, nothing matters anymore.",
    "Just had a great day at the park with my friends, feeling refreshed.",
    "Work has been piling up and I feel overwhelmed and exhausted.",
    "Finished a fun movie night, really enjoyed the evening.",
    "I don't see the point in anything anymore, everything feels heavy.",
    "Saved up enough this month to finally book that trip I wanted.",
    "Every day feels harder than the last, I feel like giving up.",
    "Anyone have recommendations for a good book to read this weekend?",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Train a tiny BPE tokenizer locally (no network) ---
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        TRAINING_TEXT,
        vocab_size=500,
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    # Add a RoBERTa-style post-processor so this fixture actually
    # auto-inserts <s> (bos) / </s> (eos) around every encoded sequence,
    # matching real RoBERTa tokenizer behavior. Without this, the raw
    # tokenizer had no structural tokens at all, which made it a
    # misleading stand-in for testing Phase 5's bos/eos-aware
    # explainability code (that code specifically needs a fixture that
    # behaves like the real thing here).
    from tokenizers.processors import TemplateProcessing

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    tmp_tok_dir = OUT_DIR / "_raw_tokenizer"
    tmp_tok_dir.mkdir(exist_ok=True)
    tokenizer_json_path = tmp_tok_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        model_max_length=128,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast_tokenizer.save_pretrained(str(OUT_DIR))

    # --- Build a tiny, randomly-initialized RoBERTa classifier ---
    config = RobertaConfig(
        vocab_size=fast_tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=130,
        type_vocab_size=1,
        num_labels=2,
        pad_token_id=fast_tokenizer.pad_token_id,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
    )
    model = RobertaForSequenceClassification(config)  # random init, no download
    model.save_pretrained(str(OUT_DIR))

    print(f"Tiny test checkpoint written to {OUT_DIR}")
    print("This is a random-weight, architecturally-real RoBERTa checkpoint "
          "for pipeline smoke-testing only — never use it for actual predictions.")


if __name__ == "__main__":
    main()
