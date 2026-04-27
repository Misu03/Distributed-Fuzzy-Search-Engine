"""
Integration tests — rulează fără Celery/Redis.
Testează coordonatorul folosind thread pool local.
Run: python tests/test_integration.py
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from concurrent.futures import ThreadPoolExecutor, as_completed
from core.dl_engine import search_chunk
from core.chunker import chunk_text_by_words

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  \u2713  {name}")
        passed += 1
    else:
        print(f"  \u2717  {name}  \u2192  {detail}")
        failed += 1

def local_search(text, query, chunk_size=2000, overlap=100,
                 max_dist=2, sim_thresh=0.0, n_workers=4, context_size=80):
    chunks = list(chunk_text_by_words(text, chunk_size, overlap))
    all_matches = []
    seen = set()
    t0 = time.perf_counter()
    def worker(args):
        cid, (offset, ct) = args
        return search_chunk(ct, offset, cid, query, max_dist, context_size)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(worker, (cid, c)): cid for cid, c in enumerate(chunks)}
        for fut in as_completed(futures):
            for m in fut.result():
                key = (m.position, m.matched_text)
                if key not in seen:
                    seen.add(key)
                    if m.similarity >= sim_thresh:
                        all_matches.append(m)
    all_matches.sort(key=lambda x: x.position)
    return all_matches, len(chunks), time.perf_counter() - t0

# 1
print("\n-- 1. Potrivire exacta --")
text1 = "Ana are mere si pere, iar Ion are gutui si prune."
m1, _, _ = local_search(text1, "mere", max_dist=0, sim_thresh=1.0)
check("found 1 match", len(m1)==1, len(m1))
check("pozitie corecta", m1[0].position==text1.index("mere"))
check("similarity 1.0", m1[0].similarity==1.0)
check("distance 0", m1[0].distance==0)

# 2
print("\n-- 2. Fuzzy (substitutie dist=1) --")
m2, _, _ = local_search("Algoritmul de cautate fuzzy este eficient.", "cautare", max_dist=1, sim_thresh=0.5)
check("gasit match fuzzy", len(m2)>=1, len(m2))
check("distance==1", any(m.distance==1 for m in m2))

# 3
print("\n-- 3. Transpozitie DL --")
m3, _, _ = local_search("Implementarea algortim Damerau este rapida.", "algoritm", max_dist=1, sim_thresh=0.5)
check("transpozitie detectata", len(m3)>=1, len(m3))
check("cost=1", any(m.distance==1 for m in m3))

# 4
print("\n-- 4. Fraza multi-cuvant --")
text4 = "Sistemul de cautare distribuita foloseste Celery si Redis."
m4, _, _ = local_search(text4, "cautare distribuita", max_dist=0, sim_thresh=1.0)
check("fraza gasita", len(m4)>=1, len(m4))
check("pozitie fraza", m4[0].position==text4.index("cautare") if m4 else False)

# 5
print("\n-- 5. Zero rezultate --")
m5, _, _ = local_search("Ana are mere si pere.", "elefant", max_dist=1)
check("fara false pozitive", len(m5)==0, len(m5))

# 6
print("\n-- 6. Match la granita chunk --")
text6 = "x "*49 + "BOUNDARY " + "y "*49
m6, nc6, _ = local_search(text6, "BOUNDARY", chunk_size=100, overlap=20, max_dist=0)
check("match la granita", len(m6)>=1, len(m6))
check("mai mult de 1 chunk", nc6>1, nc6)

# 7
print("\n-- 7. Text mare cu injectari --")
random.seed(42)
vocab = ["informatica","distribuita","cautare","fuzzy","algoritm","text","nod","worker","chunk","redis"]
bw = [random.choice(vocab) for _ in range(5000)]
TARGET = "Damerau"
inject_at = [100,500,1000,1500,2000,2500,3000,3500,4000,4500]
for ip in inject_at: bw[ip] = TARGET
big_text = " ".join(bw)
m7, nc7, wt7 = local_search(big_text, TARGET, chunk_size=2000, overlap=100, max_dist=0, sim_thresh=1.0)
check(f"gasit >=9 injectari ({len(m7)})", len(m7)>=9)
check("sortate dupa pozitie", all(m7[i].position<=m7[i+1].position for i in range(len(m7)-1)))
check("multi-chunk", nc7>1, nc7)
check(f"wall time ok ({wt7:.2f}s)", wt7<30)

# 8
print("\n-- 8. Edge cases --")
m8a, _, _ = local_search("", "query", max_dist=2)
check("text gol: 0 matches", len(m8a)==0)
try:
    m8b, _, _ = local_search("un text", "", max_dist=2)
    check("query gol: fara crash", True)
except Exception as e:
    check("query gol: fara crash", False, str(e))

# 9
print("\n-- 9. Context snippet --")
m9, _, _ = local_search("prefix words here TARGET suffix words here", "TARGET", max_dist=0, context_size=20)
check("context prezent", len(m9[0].context)>0 if m9 else False)
check("context contine match", "TARGET" in m9[0].context if m9 else False)

# 10
print("\n-- 10. Deduplicare overlap --")
m10, _, _ = local_search("cuvant "*200, "cuvant", chunk_size=100, overlap=50, max_dist=0)
positions = [m.position for m in m10]
check("fara pozitii duplicate", len(positions)==len(set(positions)))

# 11
print("\n-- 11. Similarity threshold --")
m_strict, _, _ = local_search("cautate cautate cautate", "cautare", max_dist=1, sim_thresh=0.95)
m_lax,    _, _ = local_search("cautate cautate cautate", "cautare", max_dist=1, sim_thresh=0.0)
check("threshold strict <= lax", len(m_strict)<=len(m_lax))

# 12 — benchmark
print("\n-- 12. Benchmark local --")
bench_text = " ".join(["cautare fuzzy algoritm distribuita"]*500)
rows = []
for cs in [500, 2000, 5000]:
    m, nc, wt = local_search(bench_text, "cautare", chunk_size=cs, overlap=50, max_dist=1)
    rows.append({"chunk_size":cs,"chunks":nc,"wall":wt,"matches":len(m)})
check("3 randuri benchmark", len(rows)==3)
check("chunk mic → mai multe chunks", rows[0]["chunks"]>rows[2]["chunks"])
print(f"\n  {'chunk_size':>12}  {'chunks':>8}  {'wall(s)':>8}  {'matches':>8}")
for r in rows:
    print(f"  {r['chunk_size']:>12}  {r['chunks']:>8}  {r['wall']:>8.4f}  {r['matches']:>8}")

print(f"\n{'='*55}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*55}")
sys.exit(0 if failed==0 else 1)
