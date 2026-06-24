#!/usr/bin/env python3
"""deepdive_kai0 showcase — public-facing demo/progress page.

Lightweight FastAPI single-process server. No ROS / CUDA / model imports —
runs anywhere with just `pip install fastapi uvicorn`.

Usage:
    python web/showcase/server.py [--port 8765] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


# ── Paths ──
_web_dir = Path(__file__).resolve().parent
_project_root = _web_dir.parent.parent
_docs_root = _project_root / "docs"
_content_dir = _web_dir / "content"


# ── App ──
app = FastAPI(
    title="deepdive_kai0 Showcase",
    version="0.1.0",
    description="Public-facing project showcase for deepdive_kai0 (kai0/π0.5 deployment).",
)


# ── Static & templates ──
app.mount("/static", StaticFiles(directory=str(_web_dir / "static")), name="static")

# ── Standalone reports (self-contained HTML + figures/videos); /reports/<name>/ serves index.html ──
_reports_dir = _web_dir / "reports"
if _reports_dir.is_dir():
    app.mount("/reports", StaticFiles(directory=str(_reports_dir), html=True), name="reports")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(_web_dir / "templates" / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "content": {
            "features": (_content_dir / "features.json").is_file(),
            "milestones": (_content_dir / "milestones.json").is_file(),
            "docs_index": (_content_dir / "docs_index.json").is_file(),
        },
        "docs_root": str(_docs_root),
    }


# ── Content endpoints (read JSON from content/) ──
def _read_json(name: str) -> dict:
    path = _content_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"{name} parse error: {e}")


@app.get("/api/features")
async def get_features():
    return _read_json("features.json")


@app.get("/api/milestones")
async def get_milestones():
    return _read_json("milestones.json")


@app.get("/api/docs/index")
async def get_docs_index():
    return _read_json("docs_index.json")


# ── Doc serving (markdown raw text) ──
# Resolves arbitrary relative paths under docs/ with strict per-segment sanitization
# (no path traversal possible). Falls back to recursive search for bare filenames so
# legacy `/api/doc/foo.md` calls keep working after docs were moved into subdirs.
_SEG_DIR_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_SEG_FILE_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+\.md$")
_MAX_DEPTH = 5


def _resolve_doc_path(rel: str) -> Optional[Path]:
    rel = rel.strip().strip("/")
    if not rel:
        return None
    parts = rel.split("/")
    if len(parts) > _MAX_DEPTH or any(p in ("", ".", "..") for p in parts):
        return None
    if not all(_SEG_DIR_RE.match(p) for p in parts[:-1]):
        return None
    if not _SEG_FILE_RE.match(parts[-1]):
        return None
    docs_root = _docs_root.resolve()
    candidate = (docs_root / rel).resolve()
    try:
        candidate.relative_to(docs_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _resolve_doc(name: str) -> Optional[Path]:
    # 1) Treat `name` as a relative path under docs/ (current scheme).
    p = _resolve_doc_path(name)
    if p is not None:
        return p
    # 2) Backward compat: bare filename → recursive search (deterministic via sort).
    if "/" not in name and _SEG_FILE_RE.match(name):
        for hit in sorted(_docs_root.rglob(name)):
            if hit.is_file():
                return hit
    return None


@app.get("/api/doc/{name:path}", response_class=PlainTextResponse)
async def get_doc(name: str):
    path = _resolve_doc(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"doc {name!r} not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/api/readme", response_class=PlainTextResponse)
async def get_readme(lang: str = "zh"):
    candidates = []
    if lang == "en":
        candidates = ["README_en.md", "README.md"]
    else:
        candidates = ["README.md", "README_zh.md"]
    for name in candidates:
        path = _project_root / name
        if path.is_file():
            return PlainTextResponse(path.read_text(encoding="utf-8"))
    return PlainTextResponse(
        f"README ({lang}) not found at {_project_root}", status_code=404
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHOWCASE_PORT", 8765)))
    parser.add_argument("--host", default=os.environ.get("SHOWCASE_HOST", "0.0.0.0"))
    parser.add_argument("--reload", action="store_true", help="dev: auto-reload on edit")
    args = parser.parse_args()

    print("=" * 60)
    print("  deepdive_kai0 Showcase")
    print(f"  http://{args.host}:{args.port}/")
    print(f"  docs_root  = {_docs_root}")
    print(f"  content    = {_content_dir}")
    print("=" * 60)

    uvicorn.run(
        "web.showcase.server:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
