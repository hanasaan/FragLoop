"""Dataset utilities for JSONL corpora."""

from __future__ import annotations

import json
import os
import hashlib
import random
import bisect
import math
import pickle
import logging
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from .tokenizer import Tokenizer

from tqdm import tqdm

try:  # optional dependency for parquet-backed datasets
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None

from accelerate.logging import get_logger


def _pad_sequence(seq: list[int], target_len: int, pad_id: int | None) -> list[int]:
    if len(seq) >= target_len:
        return seq
    if pad_id is None:
        return seq
    return seq + [pad_id] * (target_len - len(seq))


def _default_cache_dir(paths: list[str]) -> str:
    return os.path.expanduser("~/.cache/infiniteshader")


def _cache_key(paths: list[str]) -> str:
    stats = []
    for path in paths:
        try:
            st = os.stat(path)
            stats.append({"path": path, "size": st.st_size, "mtime": int(st.st_mtime)})
        except FileNotFoundError:
            stats.append({"path": path, "size": None, "mtime": None})
    payload = {"paths": stats, "format": "jsonl_raw_v1"}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _cache_path(paths: list[str], cache_dir: Optional[str] = None) -> str:
    key = _cache_key(paths)
    root = cache_dir or _default_cache_dir(paths)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"jsonl_{key}.pkl")


def _yield_sequences(
    tokens: list[int],
    block_size: int,
    stride: int,
    pad_to_block: bool,
    pad_id: int | None,
    use_cont: bool,
    cont_id: int | None,
) -> Iterator[list[int]]:
    target_len = block_size + 1
    if not tokens:
        return
    if len(tokens) <= target_len:
        if pad_to_block:
            yield _pad_sequence(tokens, target_len, pad_id)
        elif len(tokens) >= target_len:
            yield tokens[:target_len]
        return

    if use_cont and cont_id is not None:
        first = True
        remaining = tokens
        while remaining:
            if first:
                chunk = remaining[:target_len]
                remaining = remaining[target_len:]
                first = False
            else:
                chunk = [cont_id] + remaining[:block_size]
                remaining = remaining[block_size:]
            if len(chunk) < target_len:
                if pad_to_block:
                    chunk = _pad_sequence(chunk, target_len, pad_id)
                else:
                    if len(chunk) < 2:
                        break
            yield chunk
        return

    for i in range(0, len(tokens) - block_size, stride):
        chunk = tokens[i : i + target_len]
        if len(chunk) < target_len and pad_to_block:
            chunk = _pad_sequence(chunk, target_len, pad_id)
        if len(chunk) < 2:
            continue
        yield chunk


def _iter_jsonl_token_sequences(
    paths: list[str],
    tokenizer: Tokenizer,
    block_size: int,
    stride: int,
    add_bos: bool,
    add_eos: bool,
    max_samples: int,
    pad_to_block: bool,
    pad_id: int | None,
    use_cont: bool,
    cont_id: int | None,
) -> Iterator[torch.Tensor]:
    count = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "")
                if not text:
                    continue
                tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
                for seq in _yield_sequences(
                    tokens,
                    block_size=block_size,
                    stride=stride,
                    pad_to_block=pad_to_block,
                    pad_id=pad_id,
                    use_cont=use_cont,
                    cont_id=cont_id,
                ):
                    yield torch.tensor(seq, dtype=torch.long)
                    count += 1
                    if max_samples and count >= max_samples:
                        return


def _load_cached_entries(
    paths: list[str],
    cache_dir: Optional[str],
) -> Optional[tuple[list[tuple[str, str, int]], int]]:
    cache_path = _cache_path(paths, cache_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != 1:
            return None
        items = payload.get("items")
        raw_line_count = payload.get("raw_line_count")
        if isinstance(items, list):
            if not isinstance(raw_line_count, int):
                raw_line_count = len(items)
            return items, raw_line_count
    except Exception:
        return None
    return None


def _save_cached_entries(
    paths: list[str],
    cache_dir: Optional[str],
    items: list[tuple[str, str, int]],
    raw_line_count: int,
) -> None:
    cache_path = _cache_path(paths, cache_dir)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"version": 1, "items": items, "raw_line_count": raw_line_count},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass


def _read_jsonl_entries(
    paths: list[str],
    show_progress: bool = True,
) -> tuple[list[tuple[str, str, int]], int]:
    items: list[tuple[str, str, int]] = []
    raw_line_count = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            iterator = tqdm(f) if show_progress else f
            for line_idx, line in enumerate(iterator, start=1):
                if not line.strip():
                    continue
                raw_line_count += 1
                row = json.loads(line)
                text = row.get("text", "")
                items.append((text, path, line_idx))
    return items, raw_line_count


def _load_or_build_entries(
    paths: list[str],
    use_cache: bool,
    cache_dir: Optional[str],
    show_progress: bool = True,
) -> tuple[list[tuple[str, str, int]], int]:
    if use_cache:
        cached = _load_cached_entries(paths, cache_dir)
        if cached is not None:
            return cached
    items, raw_line_count = _read_jsonl_entries(paths, show_progress=show_progress)
    if use_cache:
        _save_cached_entries(paths, cache_dir, items, raw_line_count)
    return items, raw_line_count


def _build_samples_from_entries(
    entries: list[tuple[str, str, int]],
    tokenizer: Tokenizer,
    block_size: int,
    stride: int,
    add_bos: bool,
    add_eos: bool,
    max_samples: int,
    pad_to_block: bool,
    pad_id: int | None,
    use_cont: bool,
    cont_id: int | None,
    return_meta: bool,
    text_prefix_chars: int,
    show_progress: bool = True,
) -> tuple[list[torch.Tensor], Optional[list[dict]], int, int]:
    samples: list[torch.Tensor] = []
    meta: Optional[list[dict]] = [] if return_meta else None
    count = 0
    line_count = 0
    iterator = tqdm(entries) if show_progress else entries
    for text, source, line_idx in iterator:
        line_count += 1
        if not text:
            continue
        prefix = ""
        if return_meta and text_prefix_chars > 0:
            prefix = text[:text_prefix_chars]
        tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        for seq in _yield_sequences(
            tokens,
            block_size=block_size,
            stride=stride,
            pad_to_block=pad_to_block,
            pad_id=pad_id,
            use_cont=use_cont,
            cont_id=cont_id,
        ):
            samples.append(torch.tensor(seq, dtype=torch.long))
            if return_meta and meta is not None:
                meta.append({"source": source, "line": line_idx, "prefix": prefix})
            count += 1
            if max_samples and count >= max_samples:
                return samples, meta, count, line_count
    return samples, meta, count, line_count


class JsonlTokenDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: int = 0,
        pad_to_block: bool = False,
        use_cont: bool = False,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples
        self.pad_to_block = pad_to_block
        self.use_cont = use_cont
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.pad_id = tokenizer.special_id("<PAD>") if pad_to_block else None
        self.cont_id = tokenizer.special_id("<CONT>") if use_cont else None

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from _iter_jsonl_token_sequences(
            [self.path],
            self.tokenizer,
            self.block_size,
            self.stride,
            self.add_bos,
            self.add_eos,
            self.max_samples,
            self.pad_to_block,
            self.pad_id,
            self.use_cont,
            self.cont_id,
        )


class MultiJsonlTokenDataset(IterableDataset):
    def __init__(
        self,
        paths: list[str],
        tokenizer: Tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: int = 0,
        pad_to_block: bool = False,
        use_cont: bool = False,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples
        self.pad_to_block = pad_to_block
        self.use_cont = use_cont
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.pad_id = tokenizer.special_id("<PAD>") if pad_to_block else None
        self.cont_id = tokenizer.special_id("<CONT>") if use_cont else None

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from _iter_jsonl_token_sequences(
            self.paths,
            self.tokenizer,
            self.block_size,
            self.stride,
            self.add_bos,
            self.add_eos,
            self.max_samples,
            self.pad_to_block,
            self.pad_id,
            self.use_cont,
            self.cont_id,
        )


class JsonlTokenDatasetMap(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: int = 0,
        pad_to_block: bool = False,
        use_cont: bool = False,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        return_meta: bool = False,
        text_prefix_chars: int = 0,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples
        self.pad_to_block = pad_to_block
        self.use_cont = use_cont
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.return_meta = return_meta
        self.text_prefix_chars = text_prefix_chars
        self.pad_id = tokenizer.special_id("<PAD>") if pad_to_block else None
        self.cont_id = tokenizer.special_id("<CONT>") if use_cont else None
        self.samples: list[torch.Tensor] = []
        self.meta: Optional[list[dict]] = [] if return_meta else None
        self._load()

    def _load(self) -> None:
        entries, _ = _load_or_build_entries(
            [self.path],
            self.use_cache,
            self.cache_dir,
            show_progress=True,
        )
        samples, meta, _, _ = _build_samples_from_entries(
            entries,
            self.tokenizer,
            self.block_size,
            self.stride,
            self.add_bos,
            self.add_eos,
            self.max_samples,
            self.pad_to_block,
            self.pad_id,
            self.use_cont,
            self.cont_id,
            self.return_meta,
            self.text_prefix_chars,
            show_progress=True,
        )
        self.samples = samples
        if self.return_meta:
            self.meta = meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        if not self.return_meta:
            return sample
        meta = self.meta[idx] if self.meta is not None else {}
        meta_out = dict(meta)
        meta_out["index"] = idx
        return sample, meta_out


class MultiJsonlTokenDatasetMap(Dataset):
    def __init__(
        self,
        paths: list[str],
        tokenizer: Tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: int = 0,
        pad_to_block: bool = False,
        use_cont: bool = False,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        return_meta: bool = False,
        text_prefix_chars: int = 0,
    ) -> None:
        self.paths = paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples
        self.pad_to_block = pad_to_block
        self.use_cont = use_cont
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.return_meta = return_meta
        self.text_prefix_chars = text_prefix_chars
        self.pad_id = tokenizer.special_id("<PAD>") if pad_to_block else None
        self.cont_id = tokenizer.special_id("<CONT>") if use_cont else None
        self.samples: list[torch.Tensor] = []
        self.meta: Optional[list[dict]] = [] if return_meta else None
        self._load()

    def _load(self) -> None:
        entries, _ = _load_or_build_entries(
            self.paths,
            self.use_cache,
            self.cache_dir,
            show_progress=True,
        )
        samples, meta, count, line_count = _build_samples_from_entries(
            entries,
            self.tokenizer,
            self.block_size,
            self.stride,
            self.add_bos,
            self.add_eos,
            self.max_samples,
            self.pad_to_block,
            self.pad_id,
            self.use_cont,
            self.cont_id,
            self.return_meta,
            self.text_prefix_chars,
            show_progress=True,
        )
        self.samples = samples
        if self.return_meta:
            self.meta = meta
        print(f"Loaded {count} samples from {line_count} lines.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        if not self.return_meta:
            return sample
        meta = self.meta[idx] if self.meta is not None else {}
        meta_out = dict(meta)
        meta_out["index"] = idx
        return sample, meta_out


class ResampledDatasetMap(Dataset):
    """Resample multiple datasets according to weights."""

    def __init__(
        self,
        datasets: list[Dataset],
        weights: Optional[list[float]] = None,
        length: Optional[int] = None,
        seed: int = 1337,
        upsampling_factors: Optional[list[float]] = None,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty")
        self.datasets = datasets
        self.seed = seed
        self.lengths = [len(ds) for ds in datasets]

        if weights is None:
            if upsampling_factors is None:
                weights = [float(sz) for sz in self.lengths]
            else:
                if len(upsampling_factors) != len(datasets):
                    raise ValueError("upsampling_factors must match datasets length")
                weights = [
                    float(sz) * float(factor)
                    for sz, factor in zip(self.lengths, upsampling_factors)
                ]
        if len(weights) != len(datasets):
            raise ValueError("datasets and weights must have same length")

        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        if length is None:
            length = int(math.ceil(total))
        if length <= 0:
            raise ValueError("length must be > 0")
        self.length = length
        self.weights = [w / total for w in weights]
        cum = []
        acc = 0.0
        for w in self.weights:
            acc += w
            cum.append(acc)
        self.cum_weights = cum

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = random.Random(self.seed + idx)
        r = rng.random()
        ds_idx = bisect.bisect_left(self.cum_weights, r)
        ds_idx = min(ds_idx, len(self.datasets) - 1)
        ds_len = self.lengths[ds_idx]
        sample_idx = rng.randrange(ds_len)
        return self.datasets[ds_idx][sample_idx]


def _require_pyarrow() -> None:
    if pq is None:  # pragma: no cover - only hits when pyarrow missing
        raise SystemExit("pyarrow is required for parquet datasets (pip install pyarrow)")


def _parquet_num_rows(path: str) -> int:
    _require_pyarrow()
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    if meta is None:
        return 0
    return int(meta.num_rows or 0)


def _split_shards(
    shards: list[str],
    seed: int,
    worker_id: int,
    worker_count: int,
) -> list[str]:
    order = list(shards)
    rng = random.Random(seed)
    rng.shuffle(order)
    if worker_count <= 1:
        return order
    return [path for idx, path in enumerate(order) if idx % worker_count == worker_id]


def _assign_shards_by_worker(
    sources: list[list[str]],
    seed: int,
    worker_id: int,
    worker_count: int,
) -> list[list[str]]:
    pairs: list[tuple[str, int]] = []
    for source_idx, source in enumerate(sources):
        for path in source:
            pairs.append((path, source_idx))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    assigned: list[list[str]] = [[] for _ in sources]
    if worker_count <= 1:
        for path, source_idx in pairs:
            assigned[source_idx].append(path)
        return assigned
    for idx, (path, source_idx) in enumerate(pairs):
        if idx % worker_count == worker_id:
            assigned[source_idx].append(path)
    return assigned


def _get_worker_rank(process_index: int, num_processes: int) -> tuple[int, int]:
    from torch.utils.data import get_worker_info

    worker = get_worker_info()
    if worker is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker.id
        num_workers = worker.num_workers
    global_worker_id = process_index * num_workers + worker_id
    global_worker_count = max(1, num_processes * num_workers)
    return global_worker_id, global_worker_count


def _unwrap_logger(logger: object) -> logging.Logger:
    base = getattr(logger, "logger", None)
    if isinstance(base, logging.Logger):
        return base
    if isinstance(logger, logging.Logger):
        return logger
    return logging.getLogger(__name__)


def _ensure_logger_handler(logger: logging.Logger, level: str = "INFO") -> None:
    logger.setLevel(level.upper())
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class _ShardCycler:
    def __init__(self, shards: list[str], seed: int) -> None:
        if not shards:
            raise ValueError("shards must not be empty")
        self._shards = list(shards)
        self._rng = random.Random(seed)
        self._index = 0
        self._rng.shuffle(self._shards)

    def next(self) -> str:
        if self._index >= len(self._shards):
            self._rng.shuffle(self._shards)
            self._index = 0
        path = self._shards[self._index]
        self._index += 1
        return path


class ParquetTokenIterableDataset(IterableDataset):
    def __init__(
        self,
        sources: list[list[str]],
        tokenizer: Tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: int = 0,
        pad_to_block: bool = False,
        use_cont: bool = False,
        seed: int = 1337,
        upsampling_factors: Optional[list[float]] = None,
        text_col: str = "text",
        return_meta: bool = False,
        text_prefix_chars: int = 0,
        process_index: int = 0,
        num_processes: int = 1,
        logger: Optional[object] = None,   
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("sources must not be empty")
        _require_pyarrow()
        self.sources = sources
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples
        self.pad_to_block = pad_to_block
        self.use_cont = use_cont
        self.seed = seed
        self.upsampling_factors = upsampling_factors or []
        self.text_col = text_col
        self.return_meta = return_meta
        self.text_prefix_chars = text_prefix_chars
        self.process_index = process_index
        self.num_processes = num_processes
        self.pad_id = tokenizer.special_id("<PAD>") if pad_to_block else None
        self.cont_id = tokenizer.special_id("<CONT>") if use_cont else None
        self._source_rows = [self._count_rows(paths) for paths in sources]

        self.logger = logger or get_logger(__name__, log_level="DEBUG")
        base_logger = _unwrap_logger(self.logger)
        _ensure_logger_handler(base_logger, level="DEBUG")

        if self.upsampling_factors and len(self.upsampling_factors) != len(self.sources):
            raise ValueError("upsampling_factors must match number of sources")

    def _log_info(self, message: str) -> None:
        try:
            self.logger.info(message, main_process_only=False)
        except TypeError:
            self.logger.info(message)
        except RuntimeError:
            _unwrap_logger(self.logger).info(message)

    def _count_rows(self, paths: list[str]) -> int:
        total = 0
        for path in paths:
            total += _parquet_num_rows(path)
        if total <= 0:
            total = len(paths)
        return total

    def __len__(self) -> int:
        worker_id, worker_count = _get_worker_rank(self.process_index, self.num_processes)
        total = 0
        assigned_by_source = _assign_shards_by_worker(
            self.sources, self.seed, worker_id, worker_count
        )
        for assigned in assigned_by_source:
            for path in assigned:
                total += _parquet_num_rows(path)
        return total

    def _iter_shard_sequences(self, path: str) -> Iterator[torch.Tensor | tuple[torch.Tensor, dict]]:
        pf = pq.ParquetFile(path)
        row_index = 0
        self._log_info(
            f"[{self.process_index}] Reading parquet shard: {path} with {pf.num_row_groups} row groups"
        )
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=[self.text_col])
            col = table.column(0)
            for i in range(table.num_rows):
                row_index += 1
                text = col[i].as_py()
                if not isinstance(text, str) or not text:
                    continue
                prefix = ""
                if self.return_meta and self.text_prefix_chars > 0:
                    prefix = text[: self.text_prefix_chars]
                tokens = self.tokenizer.encode(text, add_bos=self.add_bos, add_eos=self.add_eos)
                for seq in _yield_sequences(
                    tokens,
                    block_size=self.block_size,
                    stride=self.stride,
                    pad_to_block=self.pad_to_block,
                    pad_id=self.pad_id,
                    use_cont=self.use_cont,
                    cont_id=self.cont_id,
                ):
                    sample = torch.tensor(seq, dtype=torch.long)
                    if not self.return_meta:
                        yield sample
                    else:
                        meta = {"source": path, "line": row_index, "prefix": prefix}
                        yield sample, meta

    def __iter__(self) -> Iterator[torch.Tensor | tuple[torch.Tensor, dict]]:
        worker_id, worker_count = _get_worker_rank(self.process_index, self.num_processes)
        shard_cyclers: list[_ShardCycler] = []
        active_weights: list[float] = []
        assigned_by_source = _assign_shards_by_worker(
            self.sources, self.seed, worker_id, worker_count
        )
        for idx, assigned in enumerate(assigned_by_source):
            if not assigned:
                continue
            shard_cyclers.append(_ShardCycler(assigned, self.seed + 1000 + idx))
            rows = 0
            for path in assigned:
                rows += _parquet_num_rows(path)
            if rows <= 0:
                rows = float(len(assigned))
            factor = 1.0
            if self.upsampling_factors:
                factor = float(self.upsampling_factors[idx])
            active_weights.append(float(rows) * factor)

        if not shard_cyclers:
            raise RuntimeError(
                "No shards assigned to this worker. Reduce number of processes/workers "
                "or increase number of shards."
            )

        total_weight = sum(active_weights)
        if total_weight <= 0:
            raise RuntimeError("source weights must sum to > 0")

        rng = random.Random(self.seed + 4242)
        emitted = 0
        while True:
            pick = rng.choices(range(len(shard_cyclers)), weights=active_weights, k=1)[0]
            shard_path = shard_cyclers[pick].next()
            for sample in self._iter_shard_sequences(shard_path):
                yield sample
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return
