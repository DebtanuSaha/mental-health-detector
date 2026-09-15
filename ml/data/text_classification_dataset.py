"""
text_classification_dataset.py

Minimal torch Dataset wrapping pre-tokenized (but NOT pre-padded) text +
labels, for use with the Hugging Face Trainer + a DataCollatorWithPadding.

__getitem__ deliberately returns plain Python lists/ints, not torch
tensors: padding to a common length happens per-batch, in the collator,
not per-item here. Converting to tensors before the collator sees them
would fight with its variable-length padding logic — this is the
standard Hugging Face pattern for dynamic padding.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
