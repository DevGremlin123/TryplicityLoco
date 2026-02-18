"""
Build a BPE tokenizer on our training data.
32K vocab, trained from scratch on FineWeb-Edu + code + math.
"""

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def build_tokenizer(
    data_dir: str = "./data",
    output_path: str = "./tokenizer/tryplicity.json",
    vocab_size: int = 32000,
    max_samples: int = 50000,
):
    """Build and save a BPE tokenizer from our data."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Tokenizer already exists at {output_path}")
        return str(output_path)

    print(f"Building tokenizer (vocab_size={vocab_size})...")

    # Collect text samples from all data files
    texts = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        print(f"  Reading {jsonl_file.name}...")
        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_samples:
                    break
                data = json.loads(line)
                texts.append(data["text"])
                count += 1
        print(f"    Loaded {count:,} samples")

    print(f"  Total training samples: {len(texts):,}")

    # Build BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens
    special_tokens = [
        "<|pad|>",
        "<|eos|>",
        "<|bos|>",
        "<|unk|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train on collected texts
    print("  Training tokenizer...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor for BOS/EOS
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<|bos|>:0 $A:0 <|eos|>:0",
        pair=f"<|bos|>:0 $A:0 <|eos|>:0 <|bos|>:1 $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    # Save
    tokenizer.save(str(output_path))
    print(f"  Saved tokenizer to {output_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Quick test
    test_text = "Hello, this is Tryplicity! The answer to 2+2 is 4."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Test encode: '{test_text}' -> {len(encoded.ids)} tokens")
    print(f"  Test decode: '{decoded}'")

    return str(output_path)


if __name__ == "__main__":
    build_tokenizer()
