"""Tokenizer utilities (rustbpe + tiktoken or HuggingFace tokenizers)."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import pre_tokenizers, decoders, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
except Exception as exc:  # pragma: no cover
    HFTokenizer = None
    pre_tokenizers = None
    decoders = None
    Regex = None
    BPE = None
    BpeTrainer = None
    _HF_IMPORT_ERROR = exc
else:
    _HF_IMPORT_ERROR = None

try:
    import rustbpe
except Exception as exc:  # pragma: no cover
    rustbpe = None
    _RUSTBPE_IMPORT_ERROR = exc
else:
    _RUSTBPE_IMPORT_ERROR = None

try:
    import tiktoken
except Exception as exc:  # pragma: no cover
    tiktoken = None
    _TIKTOKEN_IMPORT_ERROR = exc
else:
    _TIKTOKEN_IMPORT_ERROR = None


SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<PAD>", "<CONT>"]

# NOTE: copied from nanochat. Uses \p{N}{1,2} to reduce token waste on numbers.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class HuggingFaceTokenizer:
    """Wrapper around HuggingFace tokenizers for GPT-4 style BPE."""

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(
        cls, text_iterator: Iterable[str], vocab_size: int, min_freq: int = 0
    ) -> "HuggingFaceTokenizer":
        if HFTokenizer is None:
            raise SystemExit(
                f"tokenizers is required. Install with: pip install tokenizers ({_HF_IMPORT_ERROR})"
            )
        tokenizer = HFTokenizer(
            BPE(
                byte_fallback=True,
                unk_token=None,
                fuse_unk=False,
            )
        )
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=min_freq,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        if HFTokenizer is None:
            raise SystemExit(
                f"tokenizers is required. Install with: pip install tokenizers ({_HF_IMPORT_ERROR})"
            )
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, tokenizer_path: str) -> "HuggingFaceTokenizer":
        if HFTokenizer is None:
            raise SystemExit(
                f"tokenizers is required. Install with: pip install tokenizers ({_HF_IMPORT_ERROR})"
            )
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def encode_special(self, text: str) -> int:
        return self.tokenizer.token_to_id(text)

    def _encode_one(self, text: str, prepend=None, append=None) -> List[int]:
        assert isinstance(text, str)
        ids: List[int] = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode(self, text, prepend=None, append=None):
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        if isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> str:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
        return tokenizer_path


class RustBPETokenizer:
    """Wrapper around rustbpe for training and tiktoken for fast inference."""

    def __init__(self, enc) -> None:
        self.enc = enc
        self.bos_token_id = self.encode_special("<BOS>")
        self.eos_token_id = self.encode_special("<EOS>")

    @classmethod
    def train_from_iterator(
        cls, text_iterator: Iterable[str], vocab_size: int
    ) -> "RustBPETokenizer":
        if rustbpe is None:
            raise SystemExit(
                f"rustbpe is required. Install with: pip install rustbpe ({_RUSTBPE_IMPORT_ERROR})"
            )
        if tiktoken is None:
            raise SystemExit(
                f"tiktoken is required. Install with: pip install tiktoken ({_TIKTOKEN_IMPORT_ERROR})"
            )
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        if vocab_size_no_special < 256:
            raise ValueError(
                f"vocab_size must be >= {256 + len(SPECIAL_TOKENS)} (got {vocab_size})"
            )
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "RustBPETokenizer":
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    @classmethod
    def from_file(cls, pickle_path: str) -> "RustBPETokenizer":
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def encode(self, text, prepend=None, append=None, num_threads: int = 8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(ids)

    def save(self, tokenizer_dir: str) -> str:
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
        return pickle_path


def _is_hex_token(token: str) -> bool:
    if len(token) % 2 != 0:
        return False
    try:
        bytes.fromhex(token)
    except ValueError:
        return False
    return True


@dataclass
class LegacyByteBPETokenizer:
    id_to_token: List[str]
    merges: List[Tuple[str, str]]
    special_tokens: dict

    def __post_init__(self) -> None:
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}
        self.merge_ranks = {}
        self.merge_map = {}
        for rank, (a, b) in enumerate(self.merges):
            if a not in self.token_to_id or b not in self.token_to_id:
                continue
            pair = (self.token_to_id[a], self.token_to_id[b])
            merged = a + b
            if merged not in self.token_to_id:
                continue
            self.merge_ranks[pair] = rank
            self.merge_map[pair] = self.token_to_id[merged]

        self.byte_to_id = {}
        for tok, idx in self.token_to_id.items():
            if len(tok) == 2 and _is_hex_token(tok):
                self.byte_to_id[int(tok, 16)] = idx

    def get_vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def encode(self, text: str, prepend=None, append=None) -> List[int]:
        data = text.encode("utf-8", errors="ignore")
        tokens = [self.byte_to_id[b] for b in data]
        tokens = self._bpe(tokens)
        if prepend is not None and prepend in self.special_tokens:
            tokens = [self.special_tokens[prepend]] + tokens
        if append is not None and append in self.special_tokens:
            tokens = tokens + [self.special_tokens[append]]
        return tokens

    def encode_special(self, text: str) -> int:
        if text not in self.special_tokens:
            raise ValueError(f"Unknown special token: {text}")
        return self.special_tokens[text]

    def decode(self, ids: Sequence[int]) -> str:
        output = bytearray()
        for idx in ids:
            token = self.id_to_token[idx]
            if token in self.special_tokens:
                continue
            if not _is_hex_token(token):
                continue
            output.extend(bytes.fromhex(token))
        return output.decode("utf-8", errors="ignore")

    def _bpe(self, token_ids: List[int]) -> List[int]:
        if not self.merge_ranks or len(token_ids) < 2:
            return token_ids
        tokens = token_ids
        while True:
            best_rank = None
            best_pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pairs = [i]
                elif rank == best_rank:
                    best_pairs.append(i)
            if best_rank is None:
                break
            merged = []
            i = 0
            pair_positions = set(best_pairs)
            while i < len(tokens):
                if i in pair_positions and i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    merged.append(self.merge_map[pair])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            if len(tokens) < 2:
                break
        return tokens

    def save(self, path: str) -> str:
        if path.endswith(".json"):
            out_path = path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            out_path = os.path.join(path, "tokenizer.json")
        payload = {
            "id_to_token": self.id_to_token,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved legacy tokenizer to {out_path}")
        return out_path

    @classmethod
    def from_file(cls, path: str) -> "LegacyByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            id_to_token=payload["id_to_token"],
            merges=[tuple(pair) for pair in payload.get("merges", [])],
            special_tokens=payload.get("special_tokens", {}),
        )


class Tokenizer:
    """Facade that hides backend differences."""

    def __init__(self, impl, backend: str) -> None:
        self.impl = impl
        self.backend = backend

    @property
    def vocab_size(self) -> int:
        return self.impl.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        prepend = "<BOS>" if add_bos else None
        append = "<EOS>" if add_eos else None
        return self.impl.encode(text, prepend=prepend, append=append)

    def special_id(self, token: str) -> int:
        return self.impl.encode_special(token)

    def decode(self, ids: Sequence[int]) -> str:
        return self.impl.decode(ids)

    def save(self, output_path: str) -> str:
        if output_path.endswith(".json") or output_path.endswith(".pkl"):
            out_dir = os.path.dirname(output_path) or "."
        else:
            out_dir = output_path
        os.makedirs(out_dir, exist_ok=True)
        saved_path = self.impl.save(out_dir)
        meta = {
            "backend": self.backend,
            "special_tokens": SPECIAL_TOKENS,
            "split_pattern": SPLIT_PATTERN,
        }
        with open(os.path.join(out_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return saved_path

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterable[str],
        vocab_size: int,
        backend: str = "rustbpe",
        min_freq: int = 0,
    ) -> "Tokenizer":
        backend = backend.lower()
        if backend in {"rustbpe", "tiktoken"}:
            impl = RustBPETokenizer.train_from_iterator(text_iterator, vocab_size)
            return cls(impl, "rustbpe")
        if backend in {"hf", "tokenizers"}:
            impl = HuggingFaceTokenizer.train_from_iterator(
                text_iterator, vocab_size, min_freq=min_freq
            )
            return cls(impl, "hf")
        raise ValueError(f"Unknown tokenizer backend: {backend}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        if not os.path.exists(path):
            parent = os.path.dirname(path)
            if parent and os.path.isdir(parent):
                return cls.load(parent)
        if os.path.isdir(path):
            meta_path = os.path.join(path, "tokenizer_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                backend = meta.get("backend")
                if backend == "rustbpe":
                    return cls(RustBPETokenizer.from_directory(path), "rustbpe")
                if backend == "hf":
                    return cls(HuggingFaceTokenizer.from_directory(path), "hf")
                if backend == "legacy":
                    json_path = os.path.join(path, "tokenizer.json")
                    return cls(LegacyByteBPETokenizer.from_file(json_path), "legacy")
            pkl_path = os.path.join(path, "tokenizer.pkl")
            json_path = os.path.join(path, "tokenizer.json")
            if os.path.exists(pkl_path):
                return cls(RustBPETokenizer.from_directory(path), "rustbpe")
            if os.path.exists(json_path):
                if _is_legacy_tokenizer_json(json_path):
                    return cls(LegacyByteBPETokenizer.from_file(json_path), "legacy")
                return cls(HuggingFaceTokenizer.from_directory(path), "hf")
            raise SystemExit(f"No tokenizer files found in {path}")

        if path.endswith(".pkl"):
            return cls(RustBPETokenizer.from_file(path), "rustbpe")
        if path.endswith(".json"):
            if _is_legacy_tokenizer_json(path):
                return cls(LegacyByteBPETokenizer.from_file(path), "legacy")
            return cls(HuggingFaceTokenizer.from_file(path), "hf")
        raise SystemExit(f"Unsupported tokenizer path: {path}")


def _is_legacy_tokenizer_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return isinstance(payload, dict) and "id_to_token" in payload and "merges" in payload
