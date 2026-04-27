"""
Benchmark care demonstrează de ce Celery (procese separate) aduce
speedup real față de thread-uri Python (limitate de GIL).

Rulează cu:
    # Pornire workers (procese separate, ocolesc GIL):
    celery -A tasks.tasks worker --concurrency=4 --loglevel=warning &

    # Rulare benchmark:
    python analysis/celery_benchmark.py --size 10 --workers-list 1,2,4,8
"""

import sys, os, time, argparse, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

from coordinator import FuzzySearchCoordinator, SearchConfig

VOCAB = (
    "informatica distribuita cautare fuzzy algoritm text document "
    "prelucrare date nod cluster worker celery redis python "
    "performanta viteza throughput latenta paralelism optimizare "
    "Damerau Levenshtein distanta editare inserare stergere "
    "substitutie transpozitie similaritate aproximativa "
    "fragment suprapunere dimensiune analiza benchmark "
    "rezultat pozitie context fereastra alunecare token"
).split()


def generate(size_mb, query):
    random.seed(42)
    target = size_mb * 1_000_000
    lines = []; total = 0; inj = 0
    inj_every = max(1, target // (len(query) * 60))
    while total < target:
        words = random.choices(VOCAB, k=random.randint(12, 30))
        if total % inj_every < 200 and inj < size_mb * 5:
            words.insert(random.randint(0, len(words)), query); inj += 1
        line = " ".join(words) + "\n"
        lines.append(line); total += len(line)
    return "".join(lines), inj


def run_bench(text, query, config, label):
    coord = FuzzySearchCoordinator(config)
    t0 = time.perf_counter()
    result = coord.search_text(text, query)
    wall = time.perf_counter() - t0
    mb = len(text.encode()) / 1e6
    print(f"  {label:<30} wall={result.wall_time_s:>6.3f}s  "
          f"cpu={result.total_worker_time_s:>6.3f}s  "
          f"speedup={result.speedup:>5.2f}x  "
          f"MB/s={result.throughput_mb_per_s:>6.2f}  "
          f"matches={result.total_matches}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--query", default="algoritm fuzzy")
    ap.add_argument("--chunk-sizes", default="10000,50000,100000,500000")
    ap.add_argument("--overlaps", default="0,200,500")
    ap.add_argument("--save-json", metavar="PATH")
    args = ap.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    overlaps    = [int(x) for x in args.overlaps.split(",")]

    print(f"\nGenerare {args.size} MB text...")
    text, inj = generate(args.size, args.query)
    mb = len(text.encode()) / 1e6
    print(f"Text: {mb:.2f} MB | {inj} injectari de '{args.query}'\n")

    all_results = []

    # ── Bench 1: chunk_size × overlap ──────────────────────────────────
    print("=" * 70)
    print("BENCHMARK: chunk_size × overlap  (max_dist=2)")
    print("=" * 70)
    print(f"  {'label':<30} {'wall':>8}  {'cpu':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}")
    print("-" * 70)

    best = None
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                continue
            cfg = SearchConfig(
                chunk_size=cs, overlap_size=ov,
                max_distance=2, similarity_threshold=0.7,
            )
            label = f"cs={cs:>7,}  ov={ov:>5}"
            r = run_bench(text, args.query, cfg, label)
            row = {
                "chunk_size": cs, "overlap": ov,
                "wall_s": r.wall_time_s,
                "cpu_s": r.total_worker_time_s,
                "speedup": r.speedup,
                "mb_s": r.throughput_mb_per_s,
                "matches": r.total_matches,
            }
            all_results.append(row)
            if best is None or r.throughput_mb_per_s > best["mb_s"]:
                best = row

    print()
    print(f"  ★ Configuratie optima: chunk_size={best['chunk_size']:,}  "
          f"overlap={best['overlap']}  →  {best['mb_s']:.2f} MB/s")

    # ── Bench 2: max_distance ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("BENCHMARK: max_distance  (chunk=50000, overlap=500)")
    print("=" * 70)
    print(f"  {'label':<30} {'wall':>8}  {'cpu':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}")
    print("-" * 70)

    dist_results = []
    for d in [0, 1, 2, 3]:
        cfg = SearchConfig(
            chunk_size=50_000, overlap_size=500,
            max_distance=d, similarity_threshold=0.0,
        )
        label = f"max_distance={d}"
        r = run_bench(text, args.query, cfg, label)
        dist_results.append({
            "max_distance": d, "wall_s": r.wall_time_s,
            "mb_s": r.throughput_mb_per_s, "matches": r.total_matches,
        })

    if args.save_json:
        report = {
            "config": vars(args), "text_mb": mb,
            "grid_results": all_results,
            "best_config": best,
            "distance_results": dist_results,
        }
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Raport salvat: {args.save_json}")


if __name__ == "__main__":
    main()
