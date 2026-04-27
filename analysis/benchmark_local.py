"""
Generare date sintetice + analiză speedup/throughput locală.
Rulează fără Celery — folosește ThreadPoolExecutor pentru a simula N workers.

Usage:
    python analysis/benchmark_local.py --size 5 --workers 1,2,4,8
    python analysis/benchmark_local.py --size 20 --chunk-sizes 5000,20000,50000,100000
    python analysis/benchmark_local.py --generate-only --size 50
"""

import sys, os, time, argparse, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from concurrent.futures import ThreadPoolExecutor, as_completed
from core.dl_engine import search_chunk
from core.chunker import chunk_text_by_words


# ── Vocabulary românesc extins ───────────────────────────────────────────────
VOCAB = (
    "informatica distribuita cautare fuzzy algoritm text document "
    "prelucrare date nod cluster worker celery redis python "
    "performanta viteza throughput latenta paralelism optimizare "
    "Damerau Levenshtein distanta editare inserare stergere "
    "substitutie transpozitie similaritate aproximativa "
    "fragment suprapunere dimensiune analiza benchmark "
    "rezultat pozitie context fereastra alunecare token "
    "retea server client cerere raspuns protocol comunicare "
    "memorie procesor fir executie sincron asincron distribuire "
    "compresie serializare deserializare codificare decodificare "
    "eficienta scalabilitate toleranta eroare replicare partitionare "
    "index structura arbore tabla hash sortare interogare filtrare"
).split()

TARGET_PHRASES = [
    "algoritm fuzzy",
    "cautare distribuita",
    "Damerau Levenshtein",
    "performanta throughput",
    "prelucrare date",
]


def generate_text(size_mb: int, inject_query: str, inject_count: int = 100) -> str:
    """Generează text sintetic de dimensiunea dată cu query-ul injectat."""
    target_chars = size_mb * 1_000_000
    random.seed(42)
    lines = []
    total = 0
    inject_every = max(1, target_chars // (len(inject_query) * inject_count))
    injections = 0

    while total < target_chars:
        line_words = random.choices(VOCAB, k=random.randint(15, 40))
        if total % inject_every < 300 and injections < inject_count:
            pos = random.randint(0, len(line_words))
            line_words.insert(pos, inject_query)
            injections += 1
        line = " ".join(line_words) + "\n"
        lines.append(line)
        total += len(line)

    text = "".join(lines)
    print(f"  Text generat: {len(text)/1e6:.2f} MB, ~{injections} injectari de '{inject_query}'")
    return text


def save_text(text: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Salvat: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


def search_parallel(text, query, chunk_size, overlap, max_dist, n_workers):
    """Cauta in text folosind N thread-uri (simulare workers Celery)."""
    chunks = list(chunk_text_by_words(text, chunk_size, overlap))
    all_matches, seen = [], set()
    t0 = time.perf_counter()

    def task(cid, offset, chunk_text):
        t = time.perf_counter()
        matches = search_chunk(chunk_text, offset, cid, query, max_dist)
        return matches, time.perf_counter() - t

    total_cpu = 0.0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(task, cid, off, ct): cid for cid, (off, ct) in enumerate(chunks)}
        for fut in as_completed(futures):
            matches, cpu_t = fut.result()
            total_cpu += cpu_t
            for m in matches:
                key = (m.position, m.matched_text)
                if key not in seen:
                    seen.add(key)
                    all_matches.append(m)

    wall = time.perf_counter() - t0
    all_matches.sort(key=lambda x: x.position)
    return {
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "n_chunks": len(chunks),
        "n_matches": len(all_matches),
        "wall_s": round(wall, 4),
        "total_cpu_s": round(total_cpu, 4),
        "speedup": round(total_cpu / wall, 2) if wall > 0 else 0,
        "throughput_chunks_s": round(len(chunks) / wall, 1),
        "throughput_mb_s": round(len(text.encode()) / 1e6 / wall, 2),
    }


def search_serial(text, query, chunk_size, overlap, max_dist):
    """Versiunea serială (1 worker) pentru baseline."""
    return search_parallel(text, query, chunk_size, overlap, max_dist, n_workers=1)


# ── Benchmark 1: Speedup vs N workers ───────────────────────────────────────

def bench_workers(text: str, query: str, workers_list: list,
                  chunk_size=50_000, overlap=500, max_dist=2):
    print(f"\n{'━'*65}")
    print(f"  BENCHMARK 1: Speedup vs N workers")
    print(f"  Text: {len(text)/1e6:.1f} MB | chunk_size={chunk_size:,} | overlap={overlap}")
    print(f"  Query: '{query}' | max_dist={max_dist}")
    print(f"{'━'*65}")
    print(f"  {'workers':>8}  {'chunks':>8}  {'wall(s)':>8}  {'cpu(s)':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}")
    print(f"  {'-'*65}")

    results = []
    for nw in workers_list:
        r = search_parallel(text, query, chunk_size, overlap, max_dist, n_workers=nw)
        results.append(r)
        print(f"  {nw:>8}  {r['n_chunks']:>8}  {r['wall_s']:>8.3f}  "
              f"{r['total_cpu_s']:>8.3f}  {r['speedup']:>8.2f}×  "
              f"{r['throughput_mb_s']:>7.2f}  {r['n_matches']:>8}")
    return results


# ── Benchmark 2: Speedup vs chunk_size ──────────────────────────────────────

def bench_chunk_sizes(text: str, query: str, chunk_sizes: list,
                      n_workers=4, overlap=500, max_dist=2):
    print(f"\n{'━'*65}")
    print(f"  BENCHMARK 2: Throughput vs chunk_size")
    print(f"  Text: {len(text)/1e6:.1f} MB | workers={n_workers} | overlap={overlap}")
    print(f"  Query: '{query}' | max_dist={max_dist}")
    print(f"{'━'*65}")
    print(f"  {'chunk_size':>12}  {'chunks':>8}  {'wall(s)':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}")
    print(f"  {'-'*65}")

    results = []
    for cs in chunk_sizes:
        ov = min(overlap, cs // 4)
        r = search_parallel(text, query, cs, ov, max_dist, n_workers=n_workers)
        results.append(r)
        marker = " ◄ optim" if cs == 50_000 or cs == 100_000 else ""
        print(f"  {cs:>12,}  {r['n_chunks']:>8}  {r['wall_s']:>8.3f}  "
              f"{r['speedup']:>8.2f}×  {r['throughput_mb_s']:>7.2f}{marker}")
    return results


# ── Benchmark 3: Grid chunk_size x overlap ───────────────────────────────────

def bench_grid(text: str, query: str, chunk_sizes: list, overlaps: list,
               n_workers=4, max_dist=2):
    print(f"\n{'━'*75}")
    print(f"  BENCHMARK 3: Grid chunk_size × overlap (workers={n_workers})")
    print(f"{'━'*75}")
    print(f"  {'chunk_size':>12}  {'overlap':>8}  {'wall(s)':>8}  {'speedup':>8}  {'MB/s':>7}  {'matches':>8}")
    print(f"  {'-'*75}")

    best = None
    results = []
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                continue
            r = search_parallel(text, query, cs, ov, max_dist, n_workers=n_workers)
            results.append(r)
            if best is None or r['throughput_mb_s'] > best['throughput_mb_s']:
                best = r
            print(f"  {cs:>12,}  {ov:>8,}  {r['wall_s']:>8.3f}  "
                  f"{r['speedup']:>8.2f}×  {r['throughput_mb_s']:>7.2f}  {r['n_matches']:>8}")

    print(f"\n  ★ Configuratie optima: chunk_size={best['chunk_size']:,}  "
          f"overlap={best['overlap']}  →  {best['throughput_mb_s']} MB/s")
    return results, best


# ── Benchmark 4: max_distance impact ─────────────────────────────────────────

def bench_distance(text: str, query: str, distances: list,
                   chunk_size=50_000, overlap=500, n_workers=4):
    print(f"\n{'━'*65}")
    print(f"  BENCHMARK 4: Impactul max_distance asupra timpului")
    print(f"  workers={n_workers} | chunk_size={chunk_size:,}")
    print(f"{'━'*65}")
    print(f"  {'max_dist':>10}  {'wall(s)':>8}  {'MB/s':>7}  {'matches':>10}  {'note':}")
    print(f"  {'-'*55}")

    results = []
    for d in distances:
        r = search_parallel(text, query, chunk_size, overlap, d, n_workers=n_workers)
        results.append(r)
        note = {0: "exact", 1: "1 greseala", 2: "recomandat", 3: "permisiv", 4: "lent"}.get(d, "")
        print(f"  {d:>10}  {r['wall_s']:>8.3f}  {r['throughput_mb_s']:>7.2f}  "
              f"{r['n_matches']:>10}  {note}")
    return results


# ── Raport final ──────────────────────────────────────────────────────────────

def print_summary(b1, b2, b4, text_mb):
    print(f"\n{'━'*65}")
    print(f"  SUMAR ANALIZA PERFORMANTA")
    print(f"{'━'*65}")

    # Speedup efficiency
    if b1:
        baseline = b1[0]["wall_s"]
        print(f"\n  Eficienta paralela (chunk_size=50K, text={text_mb:.0f} MB):")
        for r in b1:
            eff = (baseline / r["wall_s"]) / r["n_workers"] * 100
            bar = "█" * int(eff / 5)
            print(f"    {r['n_workers']:2d} workers: {r['speedup']:5.2f}× speedup  {eff:5.1f}% eficienta  {bar}")

    # Best chunk size
    if b2:
        best_cs = max(b2, key=lambda x: x["throughput_mb_s"])
        print(f"\n  Chunk size optim: {best_cs['chunk_size']:,} chars → {best_cs['throughput_mb_s']} MB/s")

    # Distance impact
    if b4:
        base_t = b4[0]["wall_s"]
        print(f"\n  Cost max_distance (relativ la dist=0):")
        for r in b4:
            overhead = (r["wall_s"] / base_t - 1) * 100
            print(f"    dist={r['max_distance'] if 'max_distance' in r else '?'}: "
                  f"+{overhead:.0f}% timp  {r['n_matches']} matches")

    print(f"\n  Recomandari:")
    print(f"    • chunk_size : 50 000 – 100 000 chars")
    print(f"    • overlap    : lungime_query × 3 (minim 200)")
    print(f"    • max_dist   : 2 (echilibru precizie/recall)")
    print(f"    • workers    : egal cu nr. CPU cores disponibile")
    print(f"{'━'*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark local fuzzy search")
    p.add_argument("--size", type=int, default=5, help="Dimensiune text test (MB)")
    p.add_argument("--query", default="algoritm fuzzy", help="Query de cautat")
    p.add_argument("--workers", default="1,2,4,8", help="Lista nr workers (virgula)")
    p.add_argument("--chunk-sizes", default="5000,20000,50000,100000,200000",
                   help="Lista chunk sizes (virgula)")
    p.add_argument("--overlaps", default="0,200,500", help="Lista overlap sizes (virgula)")
    p.add_argument("--distances", default="0,1,2,3,4", help="Lista max_distance (virgula)")
    p.add_argument("--generate-only", action="store_true", help="Doar genereaza fisierul")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--save-json", metavar="PATH", help="Salveaza rezultate JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workers_list   = [int(x) for x in args.workers.split(",")]
    chunk_sizes    = [int(x) for x in args.chunk_sizes.split(",")]
    overlaps       = [int(x) for x in args.overlaps.split(",")]
    distances      = [int(x) for x in args.distances.split(",")]

    print(f"\n{'━'*65}")
    print(f"  DISTRIBUTED FUZZY SEARCH — Analiza Performanta")
    print(f"{'━'*65}")
    print(f"  Generare text {args.size} MB ...")

    text = generate_text(args.size, args.query, inject_count=max(20, args.size * 2))

    out_file = os.path.join(args.output_dir, f"test_{args.size}mb.txt")
    save_text(text, out_file)

    if args.generate_only:
        print("\n  --generate-only: stop.")
        sys.exit(0)

    text_mb = len(text.encode()) / 1e6

    b1 = bench_workers(text, args.query, workers_list,
                       chunk_size=50_000, overlap=500)

    b2 = bench_chunk_sizes(text, args.query, chunk_sizes,
                            n_workers=min(4, max(workers_list)))

    b3_results, b3_best = bench_grid(text, args.query,
                                      [10_000, 50_000, 100_000],
                                      [0, 200, 500],
                                      n_workers=min(4, max(workers_list)))

    b4 = bench_distance(text, args.query, distances,
                        chunk_size=50_000, n_workers=min(4, max(workers_list)))
    # patch max_distance field
    for r, d in zip(b4, distances):
        r["max_distance"] = d

    print_summary(b1, b2, b4, text_mb)

    if args.save_json:
        report = {
            "config": vars(args),
            "text_mb": text_mb,
            "bench_workers": b1,
            "bench_chunk_sizes": b2,
            "bench_grid": b3_results,
            "bench_distance": b4,
        }
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON salvat: {args.save_json}")
