#!/usr/bin/env python3
"""
CLI interface for fuzzy search.

Examples:
    # Search in a file
    python cli.py search --file big_text.txt --query "informatica" --distance 2

    # Search inline text
    python cli.py search --text "Ana are mere si pere" --query "mere" --distance 1

    # Benchmark different chunk sizes
    python cli.py benchmark --file big_text.txt --query "cautare" \
        --chunk-sizes 10000,50000,100000 --overlap-sizes 0,200,500

    # Generate a test file
    python cli.py generate --size 10 --output test_10mb.txt
"""

import argparse
import json
import os
import sys
import time
import random
import string
import textwrap
from pathlib import Path

# ── make sure project root is on the path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from coordinator import FuzzySearchCoordinator, SearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_WORDS = (
    "informatica distribuita cautare fuzzy algoritm text document "
    "prelucrare date nod cluster worker celery redis python "
    "performanta viteza throughput latenta paralelism "
    "Damerau Levenshtein distanta editare inserare stergere "
    "substitutie transpozitie similaritate aproximativa "
    "fragment suprapunere dimensiune analiza benchmark "
    "rezultat pozitie context fereastra alunecare token"
).split()


def generate_test_file(size_mb: int, output_path: str, inject_query: str = "algoritm fuzzy"):
    """Generate a synthetic text file of given size with injected query occurrences."""
    target_bytes = size_mb * 1_000_000
    written = 0
    injections = 0
    inject_every = max(1, target_bytes // (len(inject_query) * 50))

    with open(output_path, "w", encoding="utf-8") as f:
        while written < target_bytes:
            line_words = random.choices(SAMPLE_WORDS, k=random.randint(10, 30))

            # periodically inject the exact query
            if written % inject_every < 500 and injections < 100:
                pos = random.randint(0, len(line_words))
                line_words.insert(pos, inject_query)
                injections += 1

            line = " ".join(line_words) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))

    size_actual = os.path.getsize(output_path)
    print(f"✓ Generated {output_path}  ({size_actual/1e6:.2f} MB, ~{injections} injections of '{inject_query}')")


def print_result(result, verbose: bool = False):
    print("\n" + "═" * 60)
    print(f"  Query           : '{result.query}'")
    print(f"  Total matches   : {result.total_matches}")
    print(f"  Chunks processed: {result.chunks_processed}")
    print(f"  Wall time       : {result.wall_time_s:.3f} s")
    print(f"  Worker CPU time : {result.total_worker_time_s:.3f} s")
    print(f"  Speedup         : {result.speedup:.2f}×")
    print(f"  Throughput      : {result.throughput_chunks_per_s:.1f} chunks/s  |  {result.throughput_mb_per_s:.2f} MB/s")
    print(f"  Text size       : {result.text_size_bytes/1e6:.2f} MB")
    if result.errors:
        print(f"  ⚠ Errors        : {len(result.errors)}")
    print("═" * 60)

    if result.matches:
        limit = len(result.matches) if verbose else min(20, len(result.matches))
        print(f"\nTop {limit} matches:\n")
        for i, m in enumerate(result.matches[:limit], 1):
            print(f"  [{i:3d}] pos={m['position']:>10}  dist={m['distance']}  sim={m['similarity']:.3f}")
            print(f"        matched : {m['matched_text']!r}")
            ctx = textwrap.shorten(m['context'], width=90, placeholder="…")
            print(f"        context : …{ctx}…")
        if not verbose and result.total_matches > 20:
            print(f"\n  … and {result.total_matches - 20} more (use --verbose to see all)")
    else:
        print("\n  No matches found.")


def print_benchmark_table(rows):
    print("\n" + "─" * 80)
    hdr = f"{'chunk_size':>12}  {'overlap':>8}  {'chunks':>7}  {'wall(s)':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}"
    print(hdr)
    print("─" * 80)
    for r in rows:
        print(
            f"{r['chunk_size']:>12,}  {r['overlap_size']:>8,}  {r['chunks']:>7}  "
            f"{r['wall_time_s']:>8.3f}  {r['speedup']:>8.2f}  "
            f"{r['throughput_mb_per_s']:>7.2f}  {r['matches']:>8}"
        )
    print("─" * 80)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_search(args):
    config = SearchConfig(
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        max_distance=args.distance,
        similarity_threshold=args.similarity,
        context_size=args.context,
    )
    coord = FuzzySearchCoordinator(config)

    if args.file:
        result = coord.search_file(args.file, args.query)
    else:
        result = coord.search_text(args.text, args.query)

    print_result(result, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"query": result.query, "matches": result.matches,
                 "stats": {
                     "wall_time_s": result.wall_time_s,
                     "speedup": result.speedup,
                     "throughput_mb_per_s": result.throughput_mb_per_s,
                 }},
                f, indent=2, ensure_ascii=False
            )
        print(f"\nResults saved to {args.output}")


def cmd_benchmark(args):
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    overlap_sizes = [int(x) for x in args.overlap_sizes.split(",")]

    config = SearchConfig(max_distance=args.distance)
    coord = FuzzySearchCoordinator(config)

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(10_000_000)   # cap at 10 MB for benchmark
    else:
        text = args.text

    print(f"\nBenchmarking '{args.query}' over {len(text)/1e6:.2f} MB of text …")
    rows = coord.benchmark(text, args.query, chunk_sizes, overlap_sizes)
    print_benchmark_table(rows)


def cmd_generate(args):
    generate_test_file(args.size, args.output, inject_query=args.inject or "algoritm fuzzy")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Distributed Fuzzy Search CLI (Damerau-Levenshtein + Celery)"
    )
    sub = p.add_subparsers(dest="command")

    # ── search ──
    s = sub.add_parser("search", help="Search for a query in text or file")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", metavar="PATH")
    src.add_argument("--text", metavar="TEXT")
    s.add_argument("--query", required=True)
    s.add_argument("--distance", type=int, default=2, help="Max edit distance (default 2)")
    s.add_argument("--similarity", type=float, default=0.7)
    s.add_argument("--chunk-size", type=int, default=50_000, dest="chunk_size")
    s.add_argument("--overlap-size", type=int, default=500, dest="overlap_size")
    s.add_argument("--context", type=int, default=80)
    s.add_argument("--output", metavar="JSON_FILE")
    s.add_argument("--verbose", action="store_true")

    # ── benchmark ──
    b = sub.add_parser("benchmark", help="Grid-search chunk/overlap configurations")
    bsrc = b.add_mutually_exclusive_group(required=True)
    bsrc.add_argument("--file", metavar="PATH")
    bsrc.add_argument("--text", metavar="TEXT")
    b.add_argument("--query", required=True)
    b.add_argument("--distance", type=int, default=2)
    b.add_argument("--chunk-sizes", default="10000,50000,100000")
    b.add_argument("--overlap-sizes", default="0,200,500")

    # ── generate ──
    g = sub.add_parser("generate", help="Generate a synthetic test file")
    g.add_argument("--size", type=int, default=10, help="Size in MB")
    g.add_argument("--output", default="test_data.txt")
    g.add_argument("--inject", metavar="QUERY", help="Query to inject into text")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "search": cmd_search,
        "benchmark": cmd_benchmark,
        "generate": cmd_generate,
    }
    dispatch[args.command](args)
