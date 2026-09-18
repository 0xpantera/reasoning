# reasoning

A package-oriented, from-scratch implementation of the models and reasoning
methods in Sebastian Raschka's *Build a Reasoning Model (From Scratch)*.

The sibling `reasoning-from-scratch` repository contains the book's notebooks
and reference code. This repository contains our implementation. The reference
package is not a runtime dependency.

## Setup

```bash
uv sync
```

## Chapter 2

Chapter 2's Qwen3 model, tokenizer, artifact downloader, KV cache, device
selection, and greedy generation routines are available now:

```python
from reasoning.artifacts import download_qwen3_small
from reasoning.generation import generate_text_basic_cache
from reasoning.qwen3 import QWEN_CONFIG_06_B, Qwen3Model, Qwen3Tokenizer

artifacts = download_qwen3_small(
    kind="base",
    tokenizer_only=True,
    out_dir="qwen3",
)
tokenizer = Qwen3Tokenizer(artifacts["tokenizer"])
model = Qwen3Model(QWEN_CONFIG_06_B)
```

The book's imports translate as follows:

| Book package | This package |
| --- | --- |
| `reasoning_from_scratch.qwen3` model types | `reasoning.qwen3` |
| `reasoning_from_scratch.qwen3.download_qwen3_small` | `reasoning.artifacts.download_qwen3_small` |
| `reasoning_from_scratch.ch02.get_device` | `reasoning.get_device` |
| `reasoning_from_scratch.ch02.generate_*` | `reasoning.generation.generate_*` |

## Architecture roadmap

The package is organized by responsibility rather than book chapter. Chapter
numbers remain in examples so it is easy to follow the text.

```text
src/reasoning/
├── artifacts/          # Model, tokenizer, and checkpoint downloads
├── data/               # Dataset loading and preprocessing
├── evaluation/         # Math verification and benchmark evaluation
├── generation/         # Greedy, cached, streaming, batched, and sampled decoding
├── methods/            # Self-consistency, scoring, and self-refinement
├── qwen3/              # Config, model, layers, RoPE, tokenizer, and KV cache
├── training/
│   ├── grpo/           # Rewards, rollouts, loss, and trainer
│   └── distillation/   # Data, loss, and trainer
└── device.py

examples/               # Thin, chapter-aligned executable examples
tests/
├── unit/               # Fast isolated behavior
├── parity/             # Comparison with the book implementation
└── integration/        # Network, checkpoint, and accelerator tests
```

Directories are populated as the corresponding chapter is implemented. Core
logic belongs in the package, not in notebooks or chapter scripts.

Planned progression:

1. Qwen3 and Chapter 2 generation (current)
2. Mathematical answer parsing and MATH-500 evaluation
3. Sampling, self-consistency, scoring, and self-refinement
4. GRPO training
5. Distillation

## Development

```bash
uv run pytest
uv run ruff check .
```

The implementation is derived in part from the Apache-2.0-licensed reference
repository. See `NOTICE.md` for attribution.
