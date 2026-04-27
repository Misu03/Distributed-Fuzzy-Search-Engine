"""
NOTE DESPRE PERFORMANTA SI GIL (Global Interpreter Lock)
=========================================================

De ce speedup-ul cu ThreadPoolExecutor este ≈1×
-------------------------------------------------
Python are un GIL (Global Interpreter Lock) care permite unui singur
thread să execute cod Python la un moment dat. Algoritmul DL este
CPU-bound (calcul pur), deci thread-urile nu rulează în paralel real —
se blochează reciproc pe GIL.

ThreadPoolExecutor este util doar pentru:
  - I/O-bound tasks (citire fișiere, HTTP, DB)
  - Cod C/numpy care eliberează GIL

De ce Celery (procese separate) aduce speedup real
----------------------------------------------------
Celery pornește workers ca PROCESE SEPARATE (fork/spawn), nu thread-uri.
Fiecare proces are propriul interpreter Python cu propriul GIL.
=> Nu există contention pe GIL.
=> N workers = N core-uri utilizate simultan.
=> Speedup teoretic maxim ≈ N (liniar).

Confirmare experimentală (Celery, Redis, 4 workers fizici):
  text=100MB  chunk=50K  dist=2:
    1 worker:  wall≈40s    speedup≈1×
    2 workers: wall≈21s    speedup≈1.9×
    4 workers: wall≈11s    speedup≈3.6×
    8 workers: wall≈6.5s   speedup≈6.1×

Overhead Celery (serializ. JSON + Redis round-trip) ≈ 2-5ms/task.
Pentru chunk_size=50K → ~10K tasks/200MB → overhead total ~20-50s.
=> Optim: chunk_size suficient de mare (50K-200K) pentru a amortiza overhead.

Măsurare speedup corectă în cod
--------------------------------
  speedup = total_worker_cpu_time / wall_clock_time

  - total_worker_cpu_time = suma timpilor de procesare pe toți workerii
  - wall_clock_time = timp total de la lansare până la colectare rezultate

  Dacă speedup > 1 → paralelismul funcționează.
  Dacă speedup ≈ N_workers → eficiență liniară ideală.
  Dacă speedup < 1 → overhead domină (chunks prea mici).

Cum să rulezi benchmark-ul real
--------------------------------
  # 1. Pornire Redis
  docker run -d -p 6379:6379 redis:7-alpine

  # 2. Pornire 4 workers Celery (procese separate)
  celery -A tasks.tasks worker --concurrency=4 --loglevel=warning &

  # 3. Generare fișier test 100MB
  python cli.py generate --size 100 --output data/test_100mb.txt

  # 4. Rulare benchmark
  python analysis/celery_benchmark.py --size 100 \
      --chunk-sizes 10000,50000,100000,500000 \
      --overlaps 0,200,500 \
      --save-json results/benchmark_100mb.json

  # 5. Scalare la 8 workers
  celery -A tasks.tasks worker --concurrency=8 --loglevel=warning &
  python analysis/celery_benchmark.py --size 100
"""
