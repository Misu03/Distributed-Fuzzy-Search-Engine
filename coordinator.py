"""
Search coordinator: orchestrates chunking, task dispatch, result collection.
Provides both synchronous and async interfaces.
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from celery import group, chord
from celery.result import GroupResult

from core.chunker import chunk_text_streaming, chunk_text_by_words, estimate_chunk_count
from tasks.tasks import app, search_chunk_task, aggregate_results

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    chunk_size: int = 50_000        # characters per chunk
    overlap_size: int = 500         # overlap on each side
    max_distance: int = 2           # max DL edit distance
    similarity_threshold: float = 0.7  # derived filter (post-hoc)
    context_size: int = 80          # chars of surrounding context
    timeout: int = 300              # seconds to wait for all tasks
    batch_size: int = 50            # tasks per chord batch


@dataclass
class SearchResult:
    query: str
    config: SearchConfig
    matches: List[Dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    chunks_processed: int = 0
    wall_time_s: float = 0.0
    total_worker_time_s: float = 0.0
    speedup: float = 0.0
    throughput_chunks_per_s: float = 0.0
    throughput_mb_per_s: float = 0.0
    errors: List[Dict] = field(default_factory=list)
    text_size_bytes: int = 0


class FuzzySearchCoordinator:
    """
    High-level API for distributed fuzzy search.
    
    Usage:
        coord = FuzzySearchCoordinator(config)
        result = coord.search_file("/path/to/bigfile.txt", "search term")
        result = coord.search_text(text_string, "search term")
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search_file(self, file_path: str, query: str) -> SearchResult:
        """Search a large file using distributed workers."""
        t0 = time.perf_counter()
        file_size = os.path.getsize(file_path)

        logger.info(
            f"Starting search: file={file_path} ({file_size/1e6:.1f} MB), "
            f"query='{query}', config={self.config}"
        )

        tasks = []
        chunk_count = 0
        for chunk_id, offset, chunk_text in chunk_text_streaming(
            file_path,
            self.config.chunk_size,
            self.config.overlap_size,
        ):
            tasks.append(
                search_chunk_task.s(
                    chunk_text,
                    offset,
                    chunk_id,
                    query,
                    self.config.max_distance,
                    self.config.context_size,
                )
            )
            chunk_count += 1

        result = self._dispatch_and_collect(tasks, query, t0)
        result.text_size_bytes = file_size
        result.throughput_mb_per_s = (
            file_size / 1e6 / result.wall_time_s if result.wall_time_s > 0 else 0
        )
        return result

    def search_text(self, text: str, query: str) -> SearchResult:
        """Search an in-memory text string."""
        t0 = time.perf_counter()
        text_size = len(text.encode("utf-8"))

        chunks = list(chunk_text_by_words(
            text, self.config.chunk_size, self.config.overlap_size
        ))

        tasks = [
            search_chunk_task.s(
                chunk_text,
                offset,
                chunk_id,
                query,
                self.config.max_distance,
                self.config.context_size,
            )
            for chunk_id, (offset, chunk_text) in enumerate(chunks)
        ]

        result = self._dispatch_and_collect(tasks, query, t0)
        result.text_size_bytes = text_size
        result.throughput_mb_per_s = (
            text_size / 1e6 / result.wall_time_s if result.wall_time_s > 0 else 0
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_and_collect(
        self,
        tasks: list,
        query: str,
        t0: float,
    ) -> SearchResult:
        n = len(tasks)
        logger.info(f"Dispatching {n} tasks")

        if n == 0:
            return SearchResult(query=query, config=self.config)

        # Use chord: fan-out → aggregate callback
        job = chord(group(tasks))(aggregate_results.s(query))
        agg = job.get(timeout=self.config.timeout)   # blocks until done

        wall_time = time.perf_counter() - t0

        # Apply similarity threshold filter
        filtered_matches = [
            m for m in agg["matches"]
            if m["similarity"] >= self.config.similarity_threshold
        ]
        filtered_matches.sort(key=lambda x: x["position"])

        speedup = (
            agg["total_worker_time_s"] / wall_time
            if wall_time > 0 else 0
        )

        return SearchResult(
            query=query,
            config=self.config,
            matches=filtered_matches,
            total_matches=len(filtered_matches),
            chunks_processed=n,
            wall_time_s=round(wall_time, 4),
            total_worker_time_s=round(agg["total_worker_time_s"], 4),
            speedup=round(speedup, 2),
            throughput_chunks_per_s=round(n / wall_time, 2) if wall_time > 0 else 0,
            errors=agg.get("errors", []),
        )

    # ------------------------------------------------------------------
    # Benchmark helper
    # ------------------------------------------------------------------

    def benchmark(
        self,
        text: str,
        query: str,
        chunk_sizes: List[int],
        overlap_sizes: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Run a grid search over (chunk_size × overlap_size) configurations
        and return a comparison table.
        """
        results = []
        for cs in chunk_sizes:
            for ov in overlap_sizes:
                self.config.chunk_size = cs
                self.config.overlap_size = ov
                r = self.search_text(text, query)
                results.append({
                    "chunk_size": cs,
                    "overlap_size": ov,
                    "wall_time_s": r.wall_time_s,
                    "chunks": r.chunks_processed,
                    "matches": r.total_matches,
                    "speedup": r.speedup,
                    "throughput_chunks_per_s": r.throughput_chunks_per_s,
                    "throughput_mb_per_s": r.throughput_mb_per_s,
                })
        return results
