#!/usr/bin/env python
"""Compile and render a Shadertoy-style shader body."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from eval.metrics import compute_variance, nan_inf_detected, temporal_delta

try:
    import moderngl
except Exception:  # pragma: no cover
    moderngl = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


_CTX_CACHE: Dict[str, "moderngl.Context"] = {}
_FBO_CACHE: Dict[Tuple[int, int, int], Tuple["moderngl.Framebuffer", "moderngl.Texture"]] = {}
_TEMPLATE_CACHE: Dict[str, Dict[str, str]] = {}


def load_template(template_dir: str) -> Dict[str, str]:
    cached = _TEMPLATE_CACHE.get(template_dir)
    if cached is not None:
        return cached
    with open(os.path.join(template_dir, "template_frag.glsl"), "r", encoding="utf-8") as f:
        frag = f.read()
    with open(os.path.join(template_dir, "template_vert.glsl"), "r", encoding="utf-8") as f:
        vert = f.read()
    payload = {"frag": frag, "vert": vert}
    _TEMPLATE_CACHE[template_dir] = payload
    return payload


def build_fragment_shader(template: str, body: str) -> str:
    return template.replace("/*__BODY__*/", body)


def make_noise_texture(width: int, height: int) -> np.ndarray:
    return (np.random.rand(height, width, 3) * 255).astype(np.uint8)


def make_gradient_texture(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    gx, gy = np.meshgrid(x, y)
    return np.stack([gx, gy, np.full_like(gx, 128)], axis=-1)


def load_texture(path: Optional[str], width: int, height: int) -> np.ndarray:
    if path and Image is not None and os.path.exists(path):
        img = Image.open(path).convert("RGB").resize((width, height))
        return np.array(img)
    return make_noise_texture(width, height)


def _create_context(backend: Optional[str]):
    if backend:
        return moderngl.create_standalone_context(backend=backend)
    return moderngl.create_standalone_context()


def get_context(backend: str) -> "moderngl.Context":
    cached = _CTX_CACHE.get(backend)
    if cached is not None:
        return cached
    if backend == "auto":
        candidates = []
        env_backend = os.environ.get("MGL_BACKEND")
        if env_backend:
            candidates.append(env_backend)
        candidates.extend(["egl", "osmesa", "x11", None])
        last_exc: Exception | None = None
        for cand in candidates:
            try:
                ctx = _create_context(cand)
                _CTX_CACHE[backend] = ctx
                return ctx
            except Exception as exc:  # pragma: no cover - depends on system
                last_exc = exc
        raise RuntimeError(f"Failed to create moderngl context: {last_exc}")
    ctx = _create_context(backend)
    _CTX_CACHE[backend] = ctx
    return ctx


def get_framebuffer(ctx: "moderngl.Context", width: int, height: int):
    key = (id(ctx), width, height)
    cached = _FBO_CACHE.get(key)
    if cached is not None:
        return cached
    fbo_tex = ctx.texture((width, height), 4, dtype="f4")
    fbo = ctx.framebuffer([fbo_tex])
    _FBO_CACHE[key] = (fbo, fbo_tex)
    return fbo, fbo_tex


def compile_shader(
    body: str,
    template_dir: str,
    backend: str = "auto",
) -> Dict:
    if moderngl is None:
        return {"compile_ok": False, "error": "moderngl not installed"}

    template = load_template(template_dir)
    frag_src = build_fragment_shader(template["frag"], body)
    vert_src = template["vert"]

    try:
        ctx = get_context(backend)
    except Exception as exc:
        return {"compile_ok": False, "error": str(exc)}
    try:
        ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    except Exception as exc:
        return {"compile_ok": False, "error": str(exc)}
    return {"compile_ok": True}


def render_shader(
    body: str,
    width: int,
    height: int,
    time_value: float,
    template_dir: str,
    tex_paths: list[str],
    backend: str = "auto",
) -> Dict:
    if moderngl is None:
        return {"compile_ok": False, "error": "moderngl not installed"}

    template = load_template(template_dir)
    frag_src = build_fragment_shader(template["frag"], body)
    vert_src = template["vert"]

    try:
        ctx = get_context(backend)
    except Exception as exc:
        return {"compile_ok": False, "error": str(exc)}
    try:
        prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    except Exception as exc:
        return {"compile_ok": False, "error": str(exc)}

    vbo = ctx.buffer(np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype="f4").tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

    texture_size = 256
    textures = []
    for i in range(4):
        img = load_texture(tex_paths[i] if i < len(tex_paths) else None, texture_size, texture_size)
        tex = ctx.texture((texture_size, texture_size), 3, img.tobytes())
        tex.build_mipmaps()
        textures.append(tex)

    if "iResolution" in prog:
        prog["iResolution"].value = (float(width), float(height), 1.0)
    if "iTime" in prog:
        prog["iTime"].value = float(time_value)
    if "iTimeDelta" in prog:
        prog["iTimeDelta"].value = 0.016
    if "iFrame" in prog:
        prog["iFrame"].value = int(time_value / 0.016)
    if "iChannelResolution" in prog:
        prog["iChannelResolution"].value = [(texture_size, texture_size, 1.0)] * 4

    for i, tex in enumerate(textures):
        name = f"iChannel{i}"
        if name in prog:
            tex.use(location=i)
            prog[name].value = i

    fbo, fbo_tex = get_framebuffer(ctx, width, height)
    fbo.use()

    start = time.perf_counter()
    vao.render(moderngl.TRIANGLES)
    ctx.finish()
    render_ms = (time.perf_counter() - start) * 1000.0

    data = np.frombuffer(fbo_tex.read(), dtype=np.float32).reshape((height, width, 4))
    return {
        "compile_ok": True,
        "render_ms": render_ms,
        "image": data,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--template-dir", default="eval/shadertoy_template")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--time-delta", type=float, default=0.5)
    parser.add_argument("--preview", default="")
    parser.add_argument("--tex0", default="")
    parser.add_argument("--tex1", default="")
    parser.add_argument("--tex2", default="")
    parser.add_argument("--tex3", default="")
    parser.add_argument(
        "--backend",
        default="auto",
        help="moderngl backend (auto/egl/osmesa/x11). auto tries MGL_BACKEND then egl/osmesa/x11",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            body = f.read()
    else:
        body = args.body

    tex_paths = [args.tex0, args.tex1, args.tex2, args.tex3]

    result1 = render_shader(
        body,
        args.width,
        args.height,
        args.time,
        args.template_dir,
        tex_paths,
        backend=args.backend,
    )
    if not result1.get("compile_ok"):
        print(json.dumps(result1, ensure_ascii=False))
        return

    result2 = render_shader(
        body,
        args.width,
        args.height,
        args.time + args.time_delta,
        args.template_dir,
        tex_paths,
        backend=args.backend,
    )
    if not result2.get("compile_ok"):
        print(json.dumps(result2, ensure_ascii=False))
        return

    img1 = result1["image"]
    img2 = result2["image"]
    metrics = {
        "compile_ok": True,
        "render_ms_mean": (result1["render_ms"] + result2["render_ms"]) / 2.0,
        "variance": compute_variance(img1[..., :3]),
        "temporal_delta": temporal_delta(img1[..., :3], img2[..., :3]),
        "nan_inf": nan_inf_detected(img1),
    }

    if args.preview and Image is not None:
        preview = np.clip(img1[..., :3] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(preview).save(args.preview)
        metrics["preview"] = args.preview

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
