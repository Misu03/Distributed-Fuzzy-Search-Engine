"""
FastAPI REST server — accepts search requests, returns results + analytics.
"""

import os
import time
import logging
import uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from coordinator import FuzzySearchCoordinator, SearchConfig, SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Distributed Fuzzy Search API",
    description="Damerau-Levenshtein fuzzy search on large texts using Celery workers",
    version="1.0.0",
)

# In-memory job store (replace with Redis for production)
_jobs: dict = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Word or phrase to search for")
    text: Optional[str] = Field(None, description="Inline text to search (≤ a few MB)")
    file_path: Optional[str] = Field(None, description="Server-side file path for large files")
    max_distance: int = Field(2, ge=0, le=10, description="Max Damerau-Levenshtein distance")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    chunk_size: int = Field(50_000, ge=1_000, le=5_000_000)
    overlap_size: int = Field(500, ge=0)
    context_size: int = Field(80, ge=0, le=500)


class BenchmarkRequest(BaseModel):
    query: str
    text: str
    chunk_sizes: List[int] = Field(default=[10_000, 50_000, 100_000])
    overlap_sizes: List[int] = Field(default=[0, 200, 500])
    max_distance: int = Field(2, ge=0, le=10)


class MatchOut(BaseModel):
    position: int
    matched_text: str
    distance: int
    similarity: float
    context: str


class SearchResponse(BaseModel):
    job_id: str
    query: str
    total_matches: int
    matches: List[MatchOut]
    chunks_processed: int
    wall_time_s: float
    total_worker_time_s: float
    speedup: float
    throughput_chunks_per_s: float
    throughput_mb_per_s: float
    text_size_mb: float
    errors: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Synchronous search endpoint.
    Blocks until all distributed tasks complete.
    """
    if not req.text and not req.file_path:
        raise HTTPException(400, "Provide either 'text' or 'file_path'")

    if req.overlap_size >= req.chunk_size:
        raise HTTPException(400, "overlap_size must be less than chunk_size")

    config = SearchConfig(
        chunk_size=req.chunk_size,
        overlap_size=req.overlap_size,
        max_distance=req.max_distance,
        similarity_threshold=req.similarity_threshold,
        context_size=req.context_size,
    )
    coord = FuzzySearchCoordinator(config)

    try:
        if req.file_path:
            if not os.path.isfile(req.file_path):
                raise HTTPException(404, f"File not found: {req.file_path}")
            result: SearchResult = coord.search_file(req.file_path, req.query)
        else:
            result: SearchResult = coord.search_text(req.text, req.query)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(500, f"Search failed: {e}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = result

    return SearchResponse(
        job_id=job_id,
        query=result.query,
        total_matches=result.total_matches,
        matches=[MatchOut(**m) for m in result.matches],
        chunks_processed=result.chunks_processed,
        wall_time_s=result.wall_time_s,
        total_worker_time_s=result.total_worker_time_s,
        speedup=result.speedup,
        throughput_chunks_per_s=result.throughput_chunks_per_s,
        throughput_mb_per_s=result.throughput_mb_per_s,
        text_size_mb=round(result.text_size_bytes / 1e6, 4),
        errors=result.errors,
    )


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    """
    Grid-search over chunk_size × overlap_size to measure speedup & throughput.
    Returns comparison table.
    """
    config = SearchConfig(max_distance=req.max_distance)
    coord = FuzzySearchCoordinator(config)
    try:
        table = coord.benchmark(
            req.text, req.query,
            req.chunk_sizes, req.overlap_sizes,
        )
    except Exception as e:
        logger.exception("Benchmark failed")
        raise HTTPException(500, str(e))

    return {"benchmark_results": table}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    r = _jobs[job_id]
    return {
        "job_id": job_id,
        "query": r.query,
        "total_matches": r.total_matches,
        "matches": r.matches,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
