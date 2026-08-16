from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

DATASET_ID = "monology/pile-uncopyrighted"
FIELD_TEXT = "text"


@dataclass
class CorpusConfig:
    seq_len: int = 1024
    skip_tokens: int = 0
    max_tokens: Optional[int] = None
    shuffle_buffer: int = 10_000
    seed: int = 0


def _raw_dataset(cfg: CorpusConfig):
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    return ds


def token_stream(tokenizer: PreTrainedTokenizerBase, cfg: CorpusConfig) -> Iterator[int]:
    """Генератор ID токенов: документ за документом, с eos между документами."""
    ds = _raw_dataset(cfg)
    n_yielded = 0
    for sample in ds:
        text = sample.get(FIELD_TEXT)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for tok in ids:
            yield tok
            n_yielded += 1
        if tokenizer.eos_token_id is not None:
            yield tokenizer.eos_token_id
            n_yielded += 1
        if cfg.max_tokens is not None and n_yielded >= cfg.max_tokens:
            return


def chunk_stream(tokenizer: PreTrainedTokenizerBase, cfg: CorpusConfig) -> Iterator[torch.Tensor]:
    """
    Режет непрерывный поток токенов на чанки длины cfg.seq_len.

    cfg.skip_tokens — сколько токенов пропустить в начале потока (resume
    после обрыва сессии).

    Последний неполный чанк отбрасывается — не паддим, чтобы паддинг-токены
    не искажали density-статистику.
    """
    stream = token_stream(tokenizer, cfg)
    if cfg.skip_tokens:
        stream = itertools.islice(stream, cfg.skip_tokens, None)

    buf: List[int] = []
    for tok in stream:
        buf.append(tok)
        if len(buf) == cfg.seq_len:
            yield torch.tensor(buf, dtype=torch.long)
            buf = []


def batch_chunks(chunk_iter: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    """Собирает чанки в батчи формы (batch_size, seq_len)."""
    buf: List[torch.Tensor] = []
    for chunk in chunk_iter:
        buf.append(chunk)
        if len(buf) == batch_size:
            yield torch.stack(buf, dim=0)
            buf = []
    if buf:
        yield torch.stack(buf, dim=0)
