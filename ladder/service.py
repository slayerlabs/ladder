"""Small authenticated-by-network benchmark scoring service.

The service keeps one model resident on one GPU and serializes requests so a
second caller cannot interleave batches or exhaust VRAM. It is intended for a
private SSH/tailscale network, not direct public exposure.
"""
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .scoring import Scorer, score_pairs


class ScoreRequest(BaseModel):
    pairs: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


def create_app(model: str, revision: str | None, adapter: str | None,
               device: str, batch_size: int, context: int) -> FastAPI:
    scorer = Scorer(model, revision, device, batch_size, context, adapter, "sdpa_math" if adapter == "koliber" else "eager")
    lock = threading.Lock()
    app = FastAPI(title="tiny-LLM benchmark scorer", version="0.2.0")

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
