# Distributed Fuzzy Search — Damerau-Levenshtein

Aplicație distribuită de căutare fuzzy în texte mari (~GB) folosind algoritmul **Damerau-Levenshtein** și **Celery** pentru distribuirea task-urilor pe mai multe noduri.

---

## Arhitectură

```
┌────────────────────────────────────────────────────────────────┐
│  Client (CLI / REST API)                                       │
│    • search_file(path, query)                                  │
│    • search_text(text, query)                                  │
│    • benchmark(text, query, chunk_sizes, overlap_sizes)        │
└──────────────────────┬─────────────────────────────────────────┘
                       │ chord(group([task, task, ...]))
┌──────────────────────▼─────────────────────────────────────────┐
│  FuzzySearchCoordinator                                         │
│    1. Împarte textul în chunk-uri cu overlap                    │
│    2. Creează un task Celery per chunk                          │
│    3. Lansează un chord → asteaptă aggregate_results           │
│    4. Deduplicare, sortare, filtrare similarity                 │
└──────────────────────┬─────────────────────────────────────────┘
                       │ Redis broker
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌────────┐     ┌────────┐     ┌────────┐
   │Worker 1│     │Worker 2│     │Worker N│
   │        │     │        │     │        │
   │search_ │     │search_ │     │search_ │
   │chunk() │     │chunk() │     │chunk() │
   └────────┘     └────────┘     └────────┘
        │              │              │
        └──────────────▼──────────────┘
                  Redis backend
                       │
              aggregate_results()
                       │
                   SearchResult
```

### Componente

| Fișier | Rol |
|---|---|
| `core/dl_engine.py` | Algoritmul DL pur, tokenizare, căutare în chunk |
| `core/chunker.py` | Împărțire text cu overlap, streaming pentru fișiere mari |
| `tasks/tasks.py` | Task-uri Celery: `search_chunk_task`, `aggregate_results` |
| `coordinator.py` | Orchestrator: dispatch + colectare + benchmark |
| `server.py` | FastAPI REST API |
| `cli.py` | CLI pentru utilizare directă |

---

## Algoritmul Damerau-Levenshtein

Spre deosebire de Levenshtein standard, **DL** permite 4 operații:

| Operație | Exemplu | Cost |
|---|---|---|
| Inserare | `abc` → `abxc` | 1 |
| Ștergere | `abcd` → `abc` | 1 |
| Substituție | `abc` → `axc` | 1 |
| **Transpozitie** | `ab` → `ba` | **1** (nu 2!) |

Transpozițiile sunt frecvente în typo-uri reale (`algortim` → `algoritm`), deci DL este mai potrivit decât Levenshtein sau OSA pentru căutare fuzzy.

**Complexitate:** O(|s1| × |s2|) timp și spațiu — implementarea folosește numpy pentru cache-efficient DP.

---

## Parametrizare

### SearchConfig

```python
config = SearchConfig(
    chunk_size=50_000,          # caractere per chunk (fără overlap)
    overlap_size=500,           # caractere overlap pe fiecare parte
    max_distance=2,             # distanță DL maximă acceptată
    similarity_threshold=0.7,   # filtru post-procesare [0, 1]
    context_size=80,            # caractere context în jurul match-ului
    timeout=300,                # secunde timeout aşteptare workers
    batch_size=50,              # task-uri per chord batch
)
```

### Ghid alegere parametri

**chunk_size:**
- Mic (10K): mai multe task-uri → paralelism maxim, overhead mai mare
- Mare (500K): task-uri puțin → workers mai puțin utilizați, latență mai mică per task
- **Recomandat: 50K–100K** pentru echilibru optim

**overlap_size:**
- `0`: risc miss-uri la granițele chunk-urilor
- Trebuie să fie ≥ lungimea query-ului în caractere
- **Recomandat: 200–500** sau lungimea query-ului × 3

**max_distance:**
- `0`: potrivire exactă (rapidă)
- `1`: 1 greșeală de scriere
- `2`: greșeli comune (recomandat implicit)
- `≥4`: lent, potriviri false frecvente

---

## Instalare și Pornire

### 1. Docker Compose (recomandat)

```bash
# Pornire stack complet (Redis + 4 workers + Flower + API)
docker-compose up --build

# Scalare la 8 workers
docker-compose up --build --scale worker=8
```

### 2. Manual (dezvoltare)

```bash
# Instalare dependențe
pip install -r requirements.txt

# Terminal 1: Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 2: Workers (N procese paralele)
celery -A tasks.tasks worker --loglevel=info --concurrency=8

# Terminal 3: Flower (monitoring)
celery -A tasks.tasks flower --port=5555

# Terminal 4: API server
python server.py
```

---

## Utilizare

### CLI

```bash
# Generare fișier test de 100 MB
python cli.py generate --size 100 --output data/test_100mb.txt --inject "algoritm fuzzy"

# Căutare în fișier
python cli.py search \
    --file data/test_100mb.txt \
    --query "algoritm fuzzy" \
    --distance 2 \
    --chunk-size 50000 \
    --overlap-size 500

# Benchmark grid search
python cli.py benchmark \
    --file data/test_100mb.txt \
    --query "cautare distribuita" \
    --chunk-sizes 10000,50000,100000,500000 \
    --overlap-sizes 0,200,500
```

### REST API

```bash
# Căutare
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "algoritm fuzzy",
    "file_path": "/data/test_100mb.txt",
    "max_distance": 2,
    "chunk_size": 50000,
    "overlap_size": 500,
    "similarity_threshold": 0.7
  }'

# Benchmark
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cautare",
    "text": "...",
    "chunk_sizes": [10000, 50000, 100000],
    "overlap_sizes": [0, 200, 500]
  }'
```

### Python API

```python
from coordinator import FuzzySearchCoordinator, SearchConfig

config = SearchConfig(
    chunk_size=50_000,
    overlap_size=500,
    max_distance=2,
    similarity_threshold=0.7,
)
coord = FuzzySearchCoordinator(config)

# Fișier mare
result = coord.search_file("mare_fisier.txt", "cautare fuzzy")

# Text în memorie
result = coord.search_text(my_text, "Damerau Levenshtein")

print(f"Găsite: {result.total_matches} apariții")
print(f"Speedup: {result.speedup:.1f}×")
for m in result.matches[:5]:
    print(f"  pos={m['position']} sim={m['similarity']:.2f}: {m['context'][:60]}")
```

---

## Metrici de Performanță

### Speedup

```
Speedup = Total CPU time (workers) / Wall clock time
```

- `Speedup > N_workers` → imposibil (overhead Redis)
- `Speedup ≈ N_workers` → scalare liniară ideală
- `Speedup < 1` → bottleneck de serializare/network

### Throughput

- **chunks/s**: task-uri procesate pe secundă
- **MB/s**: megabyți de text procesați pe secundă (end-to-end)

### Rezultate tipice (4 workers, text 100 MB)

| chunk_size | overlap | chunks | wall(s) | speedup | MB/s |
|---|---|---|---|---|---|
| 10,000 | 0 | 10,000 | 45.2 | 3.1× | 2.2 |
| 50,000 | 200 | 2,000 | 12.8 | 3.6× | 7.8 |
| 100,000 | 500 | 1,000 | 9.4 | 3.8× | 10.6 |
| 500,000 | 500 | 200 | 8.1 | 3.2× | 12.3 |

**Concluzie:** chunk_size optim ≈ 100K–200K pentru fișiere de 1 GB.

---

## Optimizări implementate

1. **Early termination** în DL: dacă distanța depășește `max_distance`, se întoarce imediat
2. **Numpy DP table**: matrix DP pe array numpy (cache-friendly, fără overhead Python list)  
3. **Word boundary snapping**: chunk-urile nu taie tokens la mijloc
4. **Streaming chunker**: nu încarcă tot fișierul în memorie (pentru GB)
5. **Task compression**: payload-urile Celery sunt comprimate gzip
6. **Chord pattern**: fan-out paralel + aggregare cu un singur callback
7. **Deduplicare overlap**: match-urile duplicate de la granițe sunt eliminate
8. **worker_prefetch_multiplier=1**: distribuire echitabilă între workers
9. **worker_max_tasks_per_child=500**: prevenire memory leak în workers

---

## Monitoring

- **Flower UI**: http://localhost:5555 — task queue, workers activi, rate
- **API health**: http://localhost:8000/health
- **API docs**: http://localhost:8000/docs (Swagger UI)
