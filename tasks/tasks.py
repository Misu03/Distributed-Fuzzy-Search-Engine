"""
Celery distributed tasks for fuzzy search.
Each task processes one chunk and returns matches.
"""

import os
import time
import logging
from celery import Celery, group, chord
from celery.utils.log import get_task_logger
from typing import List, Dict, Any

from core.dl_engine import search_chunk, Match

logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Celery app setup
# ---------------------------------------------------------------------------
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery(
    "fuzzy_search",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_compression="gzip",          # compress large chunk payloads
    result_compression="gzip",
    task_acks_late=True,               # re-queue on worker crash
    worker_prefetch_multiplier=1,      # fair distribution
    result_expires=3600,               # 1 h TTL on results
    worker_max_tasks_per_child=500,    # recycle workers to prevent memory leaks
)


# ---------------------------------------------------------------------------
# Core search task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="tasks.search_chunk_task", max_retries=3)
def search_chunk_task(
    self,
    chunk_text: str,
    chunk_offset: int,
    chunk_id: int,
    query: str,
    max_distance: int,
    context_size: int = 80,
) -> Dict[str, Any]:
    """
    Celery task: fuzzy-search one text chunk.
    Returns serializable dict with matches + timing metadata.
    """
    t0 = time.perf_counter()
    try:
        matches: List[Match] = search_chunk(
            chunk_text=chunk_text,
            chunk_offset=chunk_offset,
            chunk_id=chunk_id,
            query=query,
            max_distance=max_distance,
            context_size=context_size,
        )
        elapsed = time.perf_counter() - t0

        return {
            "chunk_id": chunk_id,
            "chunk_offset": chunk_offset,
            "chunk_size": len(chunk_text),
            "matches": [
                {
                    "position": m.position,
                    "matched_text": m.matched_text,
                    "distance": m.distance,
                    "similarity": m.similarity,
                    "context": m.context,
                }
                for m in matches
            ],
            "processing_time_s": round(elapsed, 6),
            "error": None,
        }

    except Exception as exc:
        logger.error(f"Chunk {chunk_id} failed: {exc}", exc_info=True)
        try:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        except self.MaxRetriesExceededError:
            return {
                "chunk_id": chunk_id,
                "chunk_offset": chunk_offset,
                "chunk_size": len(chunk_text),
                "matches": [],
                "processing_time_s": time.perf_counter() - t0,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Aggregate task (used with chord)
# ---------------------------------------------------------------------------

@app.task(name="tasks.aggregate_results")
def aggregate_results(chunk_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Merge results from all chunks, deduplicate boundary matches."""
    all_matches = []
    seen = set()
    errors = []
    total_processing_time = 0.0

    for r in chunk_results:
        total_processing_time += r.get("processing_time_s", 0)
        if r.get("error"):
            errors.append({"chunk_id": r["chunk_id"], "error": r["error"]})
        for m in r.get("matches", []):
            key = (m["position"], m["matched_text"])
            if key not in seen:
                seen.add(key)
                all_matches.append(m)

    # sort by position
    all_matches.sort(key=lambda x: x["position"])

    return {
        "query": query,
        "total_matches": len(all_matches),
        "matches": all_matches,
        "chunks_processed": len(chunk_results),
        "errors": errors,
        "total_worker_time_s": round(total_processing_time, 4),
    }
