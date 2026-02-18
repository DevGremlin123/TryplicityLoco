"""
Data pipeline for Tryplicity training.
Loads JSONL data, tokenizes, creates batches with curriculum-aware mixing.
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from tokenizers import Tokenizer
from typing import Optional


class TextDataset(Dataset):
    """
    Simple dataset that loads pre-tokenized text chunks.
    Each item is a fixed-length sequence of token IDs.
    """

    def __init__(
        self,
        data_files: list,
        tokenizer_path: str,
        max_seq_len: int = 512,
        max_samples: int = None,
        data_mix: dict = None,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load and tokenize all text
        self.tokens = []
        self._load_data(data_files, max_samples, data_mix)

    def _load_data(self, data_files: list, max_samples: int, data_mix: dict):
        """Load text from JSONL files and tokenize into chunks."""
        all_tokens = []

        for file_path in data_files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"  Warning: {file_path} not found, skipping")
                continue

            print(f"  Loading {file_path.name}...")
            count = 0
            file_max = max_samples // len(data_files) if max_samples else None

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if file_max and count >= file_max:
                        break
                    data = json.loads(line)
                    text = data["text"]
                    encoded = self.tokenizer.encode(text)
                    all_tokens.extend(encoded.ids)
                    count += 1

            print(f"    Loaded {count:,} docs, {len(all_tokens):,} tokens so far")

        # Chunk into sequences of max_seq_len
        # Each chunk overlaps by 0 (no overlap for training efficiency)
        for i in range(0, len(all_tokens) - self.max_seq_len, self.max_seq_len):
            chunk = all_tokens[i:i + self.max_seq_len + 1]  # +1 for labels
            if len(chunk) == self.max_seq_len + 1:
                self.tokens.append(chunk)

        print(f"  Created {len(self.tokens):,} training sequences of length {self.max_seq_len}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for very large data.
    Reads JSONL files on the fly without loading everything to memory.
    """

    def __init__(
        self,
        data_files: list,
        tokenizer_path: str,
        max_seq_len: int = 512,
        shuffle_buffer: int = 10000,
    ):
        self.data_files = [Path(f) for f in data_files if Path(f).exists()]
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        token_buffer = []

        # Shuffle files
        files = list(self.data_files)
        random.shuffle(files)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = data["text"]
                    encoded = self.tokenizer.encode(text)
                    token_buffer.extend(encoded.ids)

                    # Yield complete sequences
                    while len(token_buffer) >= self.max_seq_len + 1:
                        chunk = token_buffer[:self.max_seq_len + 1]
                        token_buffer = token_buffer[self.max_seq_len:]
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        yield {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    max_seq_len: int = 512,
    batch_size: int = 8,
    max_samples: int = None,
    streaming: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from the data directory."""
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob("*.jsonl"))

    if not data_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    print(f"Found data files: {[f.name for f in data_files]}")

    if streaming:
        dataset = StreamingTextDataset(
            data_files=[str(f) for f in data_files],
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        dataset = TextDataset(
            data_files=[str(f) for f in data_files],
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


if __name__ == "__main__":
    print("Data pipeline smoke test...")
    loader = create_dataloader(
        max_seq_len=128,
        batch_size=4,
        max_samples=1000,
    )
    batch = next(iter(loader))
    print(f"  Batch input_ids: {batch['input_ids'].shape}")
    print(f"  Batch labels: {batch['labels'].shape}")
    print(f"  Total batches: {len(loader)}")
    print("  Data pipeline OK.")
