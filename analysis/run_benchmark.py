import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from core.dl_engine import search_chunk
from core.chunker import chunk_text_by_words

VOCAB = ('informatica distribuita cautare fuzzy algoritm text nod worker '
         'chunk redis celery python performanta viteza throughput latenta '
         'paralelism Damerau Levenshtein distanta editare inserare stergere '
         'substitutie transpozitie similaritate rezultat pozitie context').split()

def gen(size_mb, query):
    target = size_mb * 1_000_000
    random.seed(42)
    lines = []
    total = 0
    inject_every = max(1, target // (len(query) * 40))
    inj = 0
    while total < target:
        words = random.choices(VOCAB, k=random.randint(12, 30))
        if total % inject_every < 300 and inj < 80:
            words.insert(random.randint(0, len(words)), query)
            inj += 1
        line = ' '.join(words) + '\n'
        lines.append(line)
        total += len(line)
    return ''.join(lines), inj

def run(text, query, cs, ov, md, nw):
    chunks = list(chunk_text_by_words(text, cs, ov))
    seen = set()
    all_m = []
    t0 = time.perf_counter()
    cpu_total = 0.0
    def task(cid, off, ct):
        t = time.perf_counter()
        m = search_chunk(ct, off, cid, query, md)
        return m, time.perf_counter() - t
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(task, i, off, ct): i for i, (off, ct) in enumerate(chunks)}
        for f in as_completed(futs):
            ms, ct = f.result()
            cpu_total += ct
            for m in ms:
                k = (m.position, m.matched_text)
                if k not in seen:
                    seen.add(k)
                    all_m.append(m)
    wall = time.perf_counter() - t0
    mb = len(text.encode()) / 1e6
    return {
        'nc': len(chunks), 'nm': len(all_m),
        'wall': round(wall, 4), 'cpu': round(cpu_total, 4),
        'speedup': round(cpu_total / wall, 2) if wall > 0 else 0,
        'mbs': round(mb / wall, 2) if wall > 0 else 0,
    }

QUERY = 'algoritm fuzzy'
SIZE_MB = 2
print(f'\nGenerare {SIZE_MB} MB text sintetetic...')
text, inj = gen(SIZE_MB, QUERY)
mb = len(text.encode()) / 1e6
print(f'Text: {mb:.2f} MB | {inj} injectari de "{QUERY}"\n')

print('=' * 62)
print('BENCHMARK 1: Speedup vs numar workers (chunk=50K, dist=2)')
print('=' * 62)
print(f'{"workers":>8} {"chunks":>7} {"wall(s)":>8} {"cpu(s)":>8} {"speedup":>9} {"MB/s":>7} {"matches":>8}')
print('-' * 62)
b1 = []
for nw in [1, 2, 4]:
    r = run(text, QUERY, 50000, 500, 2, nw)
    b1.append((nw, r))
    print(f'{nw:>8} {r["nc"]:>7} {r["wall"]:>8.3f} {r["cpu"]:>8.3f} '
          f'{r["speedup"]:>8.2f}x {r["mbs"]:>7.2f} {r["nm"]:>8}')

print()
print('=' * 62)
print('BENCHMARK 2: Throughput vs chunk_size (4 workers, dist=2)')
print('=' * 62)
print(f'{"chunk_size":>12} {"overlap":>8} {"chunks":>7} {"wall(s)":>8} {"speedup":>9} {"MB/s":>7}')
print('-' * 62)
b2 = []
for cs in [5000, 20000, 50000, 100000, 200000]:
    ov = min(500, cs // 5)
    r = run(text, QUERY, cs, ov, 2, 4)
    b2.append((cs, r))
    star = ' *' if cs in (50000, 100000) else ''
    print(f'{cs:>12,} {ov:>8} {r["nc"]:>7} {r["wall"]:>8.3f} '
          f'{r["speedup"]:>8.2f}x {r["mbs"]:>7.2f}{star}')
print('  * = zona optima recomandata')

print()
print('=' * 62)
print('BENCHMARK 3: Impactul max_distance (4 workers, chunk=50K)')
print('=' * 62)
print(f'{"dist":>6} {"wall(s)":>8} {"MB/s":>7} {"matches":>8}  nota')
print('-' * 62)
b3 = []
base_wall = None
for d in [0, 1, 2, 3, 4]:
    r = run(text, QUERY, 50000, 500, d, 4)
    b3.append((d, r))
    if base_wall is None:
        base_wall = r['wall']
    overhead = (r['wall'] / base_wall - 1) * 100
    notes = {0:'exact',1:'1 greseala',2:'recomandat',3:'permisiv',4:'lent/zgomot'}
    print(f'{d:>6} {r["wall"]:>8.3f} {r["mbs"]:>7.2f} {r["nm"]:>8}  '
          f'+{overhead:4.0f}% overhead  {notes[d]}')

print()
print('=' * 62)
print('BENCHMARK 4: Grid chunk_size x overlap (4 workers, dist=2)')
print('=' * 62)
print(f'{"chunk":>10} {"overlap":>8} {"wall(s)":>8} {"speedup":>9} {"MB/s":>7} {"matches":>8}')
print('-' * 62)
best = None
for cs in [10000, 50000, 100000]:
    for ov in [0, 200, 500]:
        r = run(text, QUERY, cs, ov, 2, 4)
        if best is None or r['mbs'] > best[2]['mbs']:
            best = (cs, ov, r)
        print(f'{cs:>10,} {ov:>8} {r["wall"]:>8.3f} {r["speedup"]:>8.2f}x '
              f'{r["mbs"]:>7.2f} {r["nm"]:>8}')
print(f'\n  Configuratie optima: chunk={best[0]:,}  overlap={best[1]}  '
      f'-> {best[2]["mbs"]} MB/s  {best[2]["nm"]} matches')

print()
print('=' * 62)
print('SUMAR & RECOMANDARI')
print('=' * 62)
baseline_wall = b1[0][1]['wall']
print(f'\n  Eficienta paralela (text={mb:.1f} MB):')
for nw, r in b1:
    eff = (baseline_wall / r['wall']) / nw * 100 if nw > 0 else 0
    bar = '|' + '#' * int(eff / 5) + '.' * (20 - int(eff / 5)) + '|'
    print(f'    {nw:2d} workers: {r["speedup"]:4.2f}x speedup  {eff:5.1f}% eficienta  {bar}')

best_cs = max(b2, key=lambda x: x[1]['mbs'])
print(f'\n  Chunk size optim: {best_cs[0]:,} -> {best_cs[1]["mbs"]} MB/s')
print(f'\n  Cost max_distance relativ la dist=0:')
base = b3[0][1]['wall']
for d, r in b3:
    print(f'    dist={d}: {(r["wall"]/base-1)*100:+5.0f}% overhead  {r["nm"]:4d} matches')
print()
print('  Recomandari finale:')
print('    chunk_size  : 50 000 - 100 000 chars')
print('    overlap     : max(200, lungime_query_chars * 3)')
print('    max_distance: 2  (echilibru precizie/recall)')
print('    workers     : egal cu nr. CPU cores fizice')
print()
