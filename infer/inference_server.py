from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import hashlib

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

from eval.compile_render_cli import render_shader
from eval.metrics import compute_variance, nan_inf_detected, temporal_delta
from infshader.model import GPT, ModelConfig
from infshader.tokenizer import Tokenizer
from infshader.utils import ensure_dir, normalize_for_dedup

try:
    import websockets
except Exception as exc:  # pragma: no cover
    raise SystemExit("websockets is required. Install with: pip install websockets") from exc

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass
class ServerConfig:
    host: str
    port: int
    ckpt: str
    tokenizer: str
    prompt: str
    add_bos: bool
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    num_candidates: int
    device: str
    template_dir: str
    textures_dir: str
    render_size: int
    time_value: float
    time_delta: float
    timeout_ms: float
    black_var_th: float
    black_mean_th: float
    static_th: float
    raw_mode: bool
    copy_index: str
    copy_meta: str
    copy_threshold_chunks: int
    backend: str


class CopyDetector:
    def __init__(self, index_path: str, meta_path: str, threshold_chunks: int) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.threshold_chunks = threshold_chunks
        self.enabled = False
        self.chunk_size = 512
        self.stride = 128
        self._index: set[int] = set()
        self._load()

    @staticmethod
    def _hash64(data: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")

    def _load(self) -> None:
        if not self.index_path or not os.path.exists(self.index_path):
            return
        try:
            from array import array

            arr = array("Q")
            with open(self.index_path, "rb") as f:
                arr.fromfile(f, os.path.getsize(self.index_path) // 8)
            self._index = set(arr)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.chunk_size = int(meta.get("chunk_size", self.chunk_size))
                self.stride = int(meta.get("stride", self.stride))
            self.enabled = True
        except Exception:
            self.enabled = False

    def check(self, text: str) -> Tuple[bool, int]:
        if not self.enabled:
            return False, 0
        data = normalize_for_dedup(text).encode("utf-8", errors="ignore")
        hits = 0
        max_run = 0
        for i in range(0, max(1, len(data) - self.chunk_size + 1), self.stride):
            chunk = data[i : i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                break
            h = self._hash64(chunk)
            if h in self._index:
                hits += 1
                max_run = max(max_run, hits)
            else:
                hits = 0
        return max_run >= self.threshold_chunks, max_run


class InferenceServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self.inflight = False
        self.paused = False
        self.last_error: str = ""
        self.generation_lock = asyncio.Lock()

        self.tokenizer = Tokenizer.load(config.tokenizer)
        self.model = self._load_model(config.ckpt, config.device)
        self.prompt_ids = self.tokenizer.encode(
            config.prompt, add_bos=config.add_bos, add_eos=False
        )
        self.eos_id = self._get_eos_id() 
        self.strip_ids = set(self._get_special_ids())
        self.copy_detector = CopyDetector(
            config.copy_index, config.copy_meta, config.copy_threshold_chunks
        )

        self.run_root = Path("runs") / datetime.now().strftime("%Y%m%d")
        self.shader_dir = self.run_root / "shaders"
        self.thumb_dir = self.run_root / "thumbs"
        self.metrics_dir = self.run_root / "metrics"
        ensure_dir(str(self.shader_dir))
        ensure_dir(str(self.thumb_dir))
        ensure_dir(str(self.metrics_dir))
        ensure_dir(str(self.run_root))

    def _load_model(self, ckpt_path: str, device: str) -> GPT:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_dict = ckpt.get("config") or {}
        if "vocab_size" not in cfg_dict:
            raise SystemExit("Checkpoint missing config.vocab_size")
        config = ModelConfig(**cfg_dict)
        model = GPT(config)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        model.eval()
        return model

    def _get_eos_id(self) -> Optional[int]:
        ids = self.tokenizer.encode("", add_bos=False, add_eos=True)
        if not ids:
            return None
        return ids[-1]

    def _get_special_ids(self) -> list[int]:
        ids: list[int] = []
        for tok in ("<BOS>", "<EOS>", "<PAD>", "<CONT>"):
            try:
                ids.append(self.tokenizer.special_id(tok))
            except Exception:
                continue
        return ids

    def _trim_after_eos(self, ids: list[int]) -> list[int]:
        if self.eos_id is None:
            return ids
        try:
            idx = ids.index(self.eos_id)
        except ValueError:
            return ids
        return ids[:idx]

    def _strip_special(self, ids: list[int]) -> list[int]:
        if not self.strip_ids:
            return ids
        return [tok for tok in ids if tok not in self.strip_ids]

    def _generate_body(self) -> str:
        idx = torch.tensor([self.prompt_ids], dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k if self.config.top_k > 0 else None,
                top_p=self.config.top_p if self.config.top_p > 0 else None,
                eos_token_id=self.eos_id,
            )
        out_ids = out[0].tolist()[len(self.prompt_ids) :]
        out_ids = self._trim_after_eos(out_ids)
        out_ids = self._strip_special(out_ids)
        return self.tokenizer.decode(out_ids)

    def _texture_paths(self) -> list[str]:
        base = Path(self.config.textures_dir)
        return [
            str(base / "bluenoise_256.png"),
            str(base / "whitenoise_256.png"),
            str(base / "grad.png"),
            str(base / "lowfreq.png"),
        ]

    def _evaluate(self, body: str) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        tex_paths = self._texture_paths()
        result1 = render_shader(
            body,
            self.config.render_size,
            self.config.render_size,
            self.config.time_value,
            self.config.template_dir,
            tex_paths,
            backend=self.config.backend,
        )
        if not result1.get("compile_ok"):
            return {"compile_ok": False, "error": result1.get("error", "")}, None

        result2 = render_shader(
            body,
            self.config.render_size,
            self.config.render_size,
            self.config.time_value + self.config.time_delta,
            self.config.template_dir,
            tex_paths,
            backend=self.config.backend,
        )
        if not result2.get("compile_ok"):
            return {"compile_ok": False, "error": result2.get("error", "")}, None

        img1 = result1["image"]
        img2 = result2["image"]
        render_ms_mean = (result1["render_ms"] + result2["render_ms"]) / 2.0
        variance = compute_variance(img1[..., :3])
        delta = temporal_delta(img1[..., :3], img2[..., :3])
        nan_inf = nan_inf_detected(img1)
        mean_luma = float(np.mean(img1[..., :3]))

        metrics: Dict[str, Any] = {
            "compile_ok": True,
            "render_ms_mean": render_ms_mean,
            "variance": variance,
            "temporal_delta": delta,
            "nan_inf": nan_inf,
            "mean_luma": mean_luma,
        }
        return metrics, img1

    def _check_accept(self, metrics: Dict[str, Any], copy_suspect: bool) -> Tuple[bool, Optional[str]]:
        if not metrics.get("compile_ok"):
            return False, "compile_error"
        if metrics.get("nan_inf"):
            return False, "nan_inf"
        if metrics.get("render_ms_mean", 0.0) > self.config.timeout_ms:
            return False, "timeout"
        if metrics.get("variance", 0.0) < self.config.black_var_th or metrics.get(
            "mean_luma", 0.0
        ) < self.config.black_mean_th:
            return False, "black"
        if metrics.get("temporal_delta", 0.0) < self.config.static_th:
            return False, "static"
        if copy_suspect:
            return False, "copy"
        return True, None

    def _save_preview(self, shader_id: str, img: Optional[np.ndarray]) -> str:
        if img is None or Image is None:
            return ""
        preview = np.clip(img[..., :3] * 255.0, 0, 255).astype(np.uint8)
        path = self.thumb_dir / f"{shader_id}.png"
        Image.fromarray(preview).save(path)
        return f"/runs/{self.run_root.name}/thumbs/{shader_id}.png"

    def _save_artifacts(self, shader_id: str, body: str, metrics: Dict[str, Any], img: Optional[np.ndarray]) -> Dict[str, Any]:
        body_path = self.shader_dir / f"{shader_id}.glsl"
        with open(body_path, "w", encoding="utf-8") as f:
            f.write(body)

        metrics_path = self.metrics_dir / f"{shader_id}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        thumb_path = self._save_preview(shader_id, img)
        history_path = self.run_root / "history.jsonl"
        history_item = {
            "shader_id": shader_id,
            "body_path": str(body_path),
            "thumb_path": thumb_path,
            "metrics": metrics,
            "ts": time.time(),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_item, ensure_ascii=False) + "\n")
        return history_item

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.clients:
            return
        data = json.dumps(payload, ensure_ascii=False)
        await asyncio.gather(
            *(client.send(data) for client in list(self.clients)),
            return_exceptions=True,
        )

    async def _send_status(self) -> None:
        await self._broadcast(
            {
                "type": "status",
                "queue_depth": 1 if self.inflight else 0,
                "paused": self.paused,
                "last_error": self.last_error,
            }
        )

    async def _handle_request_next(self, payload: Dict[str, Any]) -> None:
        mode = payload.get("mode", "auto")
        if self.paused and mode != "manual":
            await self._send_status()
            return
        if self.inflight:
            return
        self.inflight = True
        asyncio.create_task(self._generate_and_send())

    async def _generate_and_send(self) -> None:
        async with self.generation_lock:
            accepted = False
            last_reason: Optional[str] = None
            try:
                for _ in range(max(1, self.config.num_candidates)):
                    body = self._generate_body()
                    metrics, img = self._evaluate(body)

                    copy_suspect = False
                    max_run = 0
                    if self.copy_detector.enabled:
                        copy_suspect, max_run = self.copy_detector.check(body)
                    metrics["copy_suspect"] = copy_suspect
                    metrics["copy_max_run"] = max_run

                    ok, reason = self._check_accept(metrics, copy_suspect)
                    if self.config.raw_mode:
                        last_reason = reason
                        shader_id = uuid.uuid4().hex[:12]
                        history_item = self._save_artifacts(shader_id, body, metrics, img)
                        payload = {
                            "type": "shader_candidate",
                            "shader_id": shader_id,
                            "body": body,
                            "template_id": "shadertoy_webgl2_v1",
                            "channels": {
                                "iChannel0": "bluenoise_256.png",
                                "iChannel1": "whitenoise_256.png",
                                "iChannel2": "grad.png",
                                "iChannel3": "lowfreq.png",
                            },
                            "metrics": metrics,
                            "rejected_reason": reason,
                        }
                        await self._broadcast(payload)
                        await self._broadcast(
                            {
                                "type": "history_item",
                                "shader_id": shader_id,
                                "thumb_path": history_item.get("thumb_path", ""),
                                "metrics": metrics,
                            }
                        )
                        accepted = True
                        break

                    if ok:
                        shader_id = uuid.uuid4().hex[:12]
                        history_item = self._save_artifacts(shader_id, body, metrics, img)
                        payload = {
                            "type": "shader_candidate",
                            "shader_id": shader_id,
                            "body": body,
                            "template_id": "shadertoy_webgl2_v1",
                            "channels": {
                                "iChannel0": "bluenoise_256.png",
                                "iChannel1": "whitenoise_256.png",
                                "iChannel2": "grad.png",
                                "iChannel3": "lowfreq.png",
                            },
                            "metrics": metrics,
                        }
                        await self._broadcast(payload)
                        await self._broadcast(
                            {
                                "type": "history_item",
                                "shader_id": shader_id,
                                "thumb_path": history_item.get("thumb_path", ""),
                                "metrics": metrics,
                            }
                        )
                        accepted = True
                        break

                    last_reason = reason
                    await self._broadcast(
                        {
                            "type": "shader_rejected",
                            "reason": reason,
                            "detail": metrics.get("error", ""),
                            "metrics": metrics,
                        }
                    )

                if not accepted:
                    await self._broadcast(
                        {
                            "type": "shader_rejected",
                            "reason": last_reason or "unknown",
                            "detail": "all_candidates_failed",
                            "metrics": {},
                            "final": True,
                        }
                    )
            except Exception as exc:
                self.last_error = str(exc)
                await self._send_status()
            finally:
                self.inflight = False

    async def _handle_message(self, payload: Dict[str, Any]) -> None:
        msg_type = payload.get("type")
        if msg_type == "request_next":
            await self._handle_request_next(payload)
        elif msg_type == "pause":
            self.paused = bool(payload.get("value", True))
            await self._send_status()
        elif msg_type == "set_params":
            self._apply_params(payload)
            await self._send_status()
        elif msg_type == "save_favorite":
            await self._save_favorite(payload)
        elif msg_type == "toggle_overlay":
            await self._send_status()

    def _apply_params(self, payload: Dict[str, Any]) -> None:
        for key in ("temperature", "top_p", "top_k", "max_tokens", "num_candidates"):
            if key in payload:
                setattr(self.config, key, payload[key])
        if "raw_mode" in payload:
            self.config.raw_mode = bool(payload["raw_mode"])
        if "render_size" in payload:
            self.config.render_size = int(payload["render_size"])

    async def _save_favorite(self, payload: Dict[str, Any]) -> None:
        shader_id = payload.get("shader_id", "")
        if not shader_id:
            return
        fav_path = self.run_root / "favorites.jsonl"
        with open(fav_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"shader_id": shader_id, "ts": time.time()}, ensure_ascii=False
                )
                + "\n"
            )
        await self._send_status()

    async def handler(
        self, websocket: websockets.WebSocketServerProtocol, path: str | None = None
    ) -> None:
        self.clients.add(websocket)
        await self._send_status()
        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                await self._handle_message(payload)
        finally:
            self.clients.discard(websocket)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_hf() -> None:
    if snapshot_download is None:
        raise SystemExit(
            "huggingface_hub is required for --hf-repo. Install with: pip install huggingface_hub"
        )


def _normalize_repo_dir(repo_id: str, local_dir: str) -> Path:
    safe = repo_id.replace("/", "__")
    return Path(local_dir) / safe


def _download_hf_repo(repo_id: str, revision: str, local_dir: str) -> Path:
    _require_hf()
    target = _normalize_repo_dir(repo_id, local_dir)
    ensure_dir(str(target))
    repo_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return Path(repo_path)


def _score_ckpt(path: Path) -> float:
    name = path.name.lower()
    score = 0.0
    if "ckpt" in name:
        score += 10.0
    if "checkpoint" in name:
        score += 8.0
    if "final" in name:
        score += 6.0
    if "best" in name:
        score += 4.0
    if path.suffix in {".pt", ".pth", ".ckpt"}:
        score += 2.0
    try:
        score += path.stat().st_size / 1e9
    except Exception:
        pass
    return score


def _find_ckpt(repo_dir: Path, preferred: str) -> str:
    if preferred:
        cand = Path(preferred)
        if cand.exists():
            return str(cand)
        rel = repo_dir / preferred
        if rel.exists():
            return str(rel)
    candidates = []
    for ext in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(repo_dir.rglob(ext))
    if not candidates:
        return ""
    best = max(candidates, key=_score_ckpt)
    return str(best)


def _find_tokenizer(repo_dir: Path, preferred: str) -> str:
    if preferred:
        cand = Path(preferred)
        if cand.exists():
            return str(cand)
        rel = repo_dir / preferred
        if rel.exists():
            return str(rel)
    meta = list(repo_dir.rglob("tokenizer_meta.json"))
    if meta:
        return str(meta[0].parent)
    pkl = list(repo_dir.rglob("tokenizer.pkl"))
    if pkl:
        return str(pkl[0].parent)
    json_tok = list(repo_dir.rglob("tokenizer.json"))
    if json_tok:
        return str(json_tok[0].parent)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-local-dir", default="models")
    parser.add_argument("--hf-ckpt", default="")
    parser.add_argument("--hf-tokenizer", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prompt", default="// write complete shader code!")
    parser.add_argument("--add-bos", action="store_true", default=True)
    parser.add_argument("--no-bos", dest="add_bos", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--template-dir", default="eval/shadertoy_template")
    parser.add_argument("--textures-dir", default="ui_optional/web/textures")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--time-delta", type=float, default=1.0)
    parser.add_argument("--timeout-ms", type=float, default=200.0)
    parser.add_argument("--black-var-th", type=float, default=1e-4)
    parser.add_argument("--black-mean-th", type=float, default=0.02)
    parser.add_argument("--static-th", type=float, default=1e-3)
    parser.add_argument("--raw-mode", action="store_true", default=False)
    parser.add_argument("--copy-index", default="data/processed/dedup/copy_index.bin")
    parser.add_argument("--copy-meta", default="data/processed/dedup/copy_index.bin.json")
    parser.add_argument("--copy-threshold-chunks", type=int, default=2)
    parser.add_argument("--backend", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.ckpt if args.ckpt and os.path.exists(args.ckpt) else ""
    tokenizer_path = (
        args.tokenizer if args.tokenizer and os.path.exists(args.tokenizer) else ""
    )

    if args.hf_repo:
        repo_dir = _download_hf_repo(args.hf_repo, args.hf_revision, args.hf_local_dir)
        ckpt_path = _find_ckpt(repo_dir, args.hf_ckpt or args.ckpt)
        tokenizer_path = _find_tokenizer(repo_dir, args.hf_tokenizer or args.tokenizer)

    if not ckpt_path:
        raise SystemExit("Checkpoint not found. Provide --ckpt or --hf-repo.")
    if not tokenizer_path:
        raise SystemExit("Tokenizer not found. Provide --tokenizer or --hf-repo.")

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using tokenizer: {tokenizer_path}")

    config = ServerConfig(
        host=args.host,
        port=args.port,
        ckpt=ckpt_path,
        tokenizer=tokenizer_path,
        prompt=args.prompt,
        add_bos=args.add_bos,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_candidates=args.num_candidates,
        device=args.device,
        template_dir=args.template_dir,
        textures_dir=args.textures_dir,
        render_size=args.render_size,
        time_value=args.time,
        time_delta=args.time_delta,
        timeout_ms=args.timeout_ms,
        black_var_th=args.black_var_th,
        black_mean_th=args.black_mean_th,
        static_th=args.static_th,
        raw_mode=args.raw_mode,
        copy_index=args.copy_index,
        copy_meta=args.copy_meta,
        copy_threshold_chunks=args.copy_threshold_chunks,
        backend=args.backend,
    )
    server = InferenceServer(config)

    async def run() -> None:
        async with websockets.serve(server.handler, config.host, config.port):
            print(f"WS server running on ws://{config.host}:{config.port}/ws")
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
