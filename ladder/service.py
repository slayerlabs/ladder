"""Small authenticated-by-network benchmark scoring service.

The service keeps one model resident on one GPU and serializes requests so a
second caller cannot interleave batches or exhaust VRAM. It is intended for a
private SSH/tailscale network, not direct public exposure.
"""
import os
import threading
import time
import gc
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .scoring import Scorer, score_pairs


class ScoreRequest(BaseModel):
    pairs: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


class MicroRequest(BaseModel):
    model: str = Field(min_length=3, max_length=200)
    revision: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=2000, ge=100, le=2000)


def normalize_hf_model(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("model URL must point to huggingface.co")
        value = parsed.path.strip("/")
    if value.startswith("models/"):
        value = value[7:]
    if value.count("/") != 1 or any(part in value for part in ("..", "\\")):
        raise ValueError("use a Hugging Face model id such as org/model")
    return value


def create_app(model: str, revision: str | None, adapter: str | None,
               device: str, batch_size: int, context: int) -> FastAPI:
    scorer = Scorer(model, revision, device, batch_size, context, adapter, "sdpa_math" if adapter == "koliber" else "eager")
    lock = threading.Lock()
    pairs_path = Path(os.environ.get("LADDER_PAIRS", "data/multiblimp-pl-v0/candidates.jsonl"))
    app = FastAPI(title="tiny-LLM benchmark scorer", version="0.2.0")

    @app.get("/", include_in_schema=False)
    def landing_page():
        page = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        return FileResponse(page, media_type="text/html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": scorer.metadata}

    @app.post("/v1/score/pairs")
    def pairs(request: ScoreRequest):
        started = time.perf_counter()
        try:
            with lock:
                rows = score_pairs(scorer, request.pairs)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"count": len(rows), "elapsed_seconds": time.perf_counter() - started,
                "model": scorer.metadata, "results": rows,
                "decision_eligible": False}

    @app.post("/v1/benchmark/micro")
    def micro(request: MicroRequest):
        """Run the provisional Polish agreement micro battery on an HF model."""
        try:
            requested_model = normalize_hf_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not pairs_path.exists():
            raise HTTPException(status_code=503, detail="micro pair data is not installed")
        with pairs_path.open(encoding="utf-8") as handle:
            pairs = [json.loads(line) for line in handle if line.strip()][:request.limit]
        started = time.perf_counter()
        transient = None
        try:
            with lock:
                transient_adapter = "koliber" if requested_model == model else None
                transient_revision = request.revision or (revision if transient_adapter == "koliber" else None)
                transient = Scorer(requested_model, transient_revision, device, batch_size, context,
                                   transient_adapter, "sdpa_math" if transient_adapter == "koliber" else "eager")
                rows = score_pairs(transient, pairs)
            mean_sentence = sum(row["sentence_prob"] for row in rows) / len(rows)
            region_rows = [row["region_prob"] for row in rows if row["region_prob"] is not None]
            result = {"model": transient.metadata, "count": len(rows),
                      "mean_sentence_probability": mean_sentence,
                      "mean_region_probability": (sum(region_rows) / len(region_rows) if region_rows else None),
                      "elapsed_seconds": time.perf_counter() - started,
                      "provisional": True, "native_review": "pending",
                      "decision_eligible": False}
            return result
        except (KeyError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if transient is not None:
                del transient
                gc.collect()
                if device.startswith("cuda"):
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    return app


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LADDER_MODEL", "OrisTeam/Koliber-v1.1-Base-Preview"))
    parser.add_argument("--revision", default=os.environ.get("LADDER_REVISION"))
    parser.add_argument("--adapter", choices=["koliber"], default=os.environ.get("LADDER_ADAPTER"))
    parser.add_argument("--device", default=os.environ.get("LADDER_DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("LADDER_BATCH_SIZE", "8")))
    parser.add_argument("--context", type=int, default=int(os.environ.get("LADDER_CONTEXT", "512")))
    parser.add_argument("--host", default=os.environ.get("LADDER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LADDER_PORT", "18150")))
    args = parser.parse_args()
    app = create_app(args.model, args.revision, args.adapter, args.device, args.batch_size, args.context)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
