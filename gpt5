# Проект: «Цифровой Экзокортекс» — Персональная База Знаний с RAG

Версия: 12.0 (Consolidated Hardened Blueprint, Post‑Review)  
Статус: Утверждено к разработке  
Дата: 2025‑10‑07

```
[//]: # Это **Тактическое Руководство по Реализации**. Он отвечает на вопрос **"КАК ИМЕННО?"**. Он берет стратегию Эксперта А и наполняет ее конкретными командами, параметрами, фрагментами кода и низкоуровневыми защитными механизмами
```
---

## 1) TL;DR

- Что это: персональная RAG‑система на базе SiYuan, Qdrant, SQLite FTS5 и кастомного Python‑пайплайна. Оптимизировано под VPS 4 vCPU / 8 GB RAM (Ubuntu 24.04).
- Цель: «второй мозг» с проверяемыми ответами, строгим цитированием и контролем галлюцинаций (RU/EN).
- Ключевые решения:
    - Кастомный RAGPipeline (без LlamaIndex в критическом пути).
    - Гибридный поиск: BM25 (FTS5 + RU‑лемматизация) + e5‑small 384d → RRF → bge‑reranker (ONNX).
    - Семантический кеш с первого дня.
    - Resilient LLM‑шлюз: OpenRouter + gemini‑cli (Gemini) + Qwen Code для кода, платный fallback с дневным лимитом, circuit breaker.
    - Полезная деградация: локальный экстрактивный QA (ONNX) и режим «только поиск».
    - ZRAM вместо/вдобавок swap, ограничение параллелизма реранкера.
- Сроки: MVP 4–5 недель.
- Стоимость: VPS ~10–20/мес+резервплатногоLLM 10–20/мес+резервплатногоLLM 5–10/мес.
- Примечание: ONNX‑модели — не локальные LLM; соответствуют ограничению «без локальных LLM».

---

## 2) Принципы и операционные политики

- Многоуровневый ретривал: BM25(FTS5+RU‑леммы) + векторы(Qdrant, 384d, int8) → RRF → кросс‑энкодер bge.
- Адаптивность: ef_search 64→96→128 при низкой уверенности до гейтинга.
- Контроль галлюцинаций: порог по откалиброванному score реранкера; «Недостаточно данных…» с причиной отказа.
- Обязательное цитирование: ответ — только валидный JSON {answer, citations[], confidence, status}.
- Деградация: LLM недоступны → локальный экстрактивный QA → при сомнении → «только поиск» (top‑snippets).
- Приватность: в LLM отправляются только отобранные фрагменты; при необходимости — маскирование PII.

---

## 3) Цели и метрики успеха

- Качество поиска:
    - Recall@15 ≥ 0.80 (целевое 0.85 с RU‑лемматизацией).
    - Fusion gain ≥ +0.15 vs лучший моно‑поиск.
    - MRR — мониторинг стабильности ранжирования.
- Groundedness: ≥ 0.80 (при бюджете — RAGAS на подвыборке) и/или Citation Precision ≥ 0.95.
- Производительность (P95, 4 vCPU):
    - Ретривал (Qdrant+FTS5+RRF) ≤ 1.5 с.
    - Реранк (50–80 кандидатов) ≤ 2.0 с.
    - LLM API ≤ 5.0 с; E2E (LLM режим) ≤ 9–12 с.
    - Конкурентность: max 1 активный реранкер.
- Надёжность:
    - Доля успешных ответов ≥ 85–90% с учётом fallback.
    - Ежемесячное успешное восстановление из бэкапа.

---

## 4) Технологический стек и конфигурация (MVP)

### 4.1 Хранилище знаний и индексация

- Источник: SiYuan (инкрементальная индексация по updated_at; позже — WebSocket).
- Чанкинг: 250–400 токенов, overlap 20–40, сохранение заголовков/section_path.
- Дедуп: SHA‑256; батчи 1–5k; транзакции.

SQL

```
-- Таблицы данных и FTS5
CREATE TABLE chunks (
  block_id TEXT PRIMARY KEY,
  doc_id   TEXT,
  section_path TEXT,
  lang     TEXT,
  content  TEXT,
  content_lemma TEXT,
  created_at TEXT,
  updated_at TEXT,
  content_hash TEXT
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  content, 
  content_lemma, 
  block_id UNINDEXED,
  tokenize='unicode61 remove_diacritics 2'
);
```

### 4.2 Лексический поиск (SQLite FTS5)

- RU: лемматизация (pymorphy3) + стоп‑лист; поиск по `content_lemma`.
- EN: поиск по `content`.
- PRAGMA‑тюнинг:

SQL

```
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=268435456; -- 256MB
```

### 4.3 Векторная память (Qdrant)

- multilingual‑e5‑small, 384d, cosine, scalar int8 (quantile=0.99).
- HNSW: m=16 (при запасе RAM — m=24), ef_construct=100, ef_search 64→96→128.
- Индексы payload: lang, doc_id, section.

YAML

```
# qdrant.yaml (фрагмент)
storage:
  performance:
    max_search_threads: 2
optimizers:
  memmap_threshold: 50000
collections:
  exocortex:
    vectors:
      size: 384
      distance: Cosine
    quantization_config:
      scalar:
        type: int8
        quantile: 0.99
```

### 4.4 Fusion (RRF)

- BM25 topK=200 + Vector topK=200 → RRF(k≈60) → top 60–80 кандидатов.

Python

```
def rrf_merge(vec_results, bm25_results, k=60):
    rank = {}
    for i, rid in enumerate(vec_results, 1):
        rank[rid] = rank.get(rid, 0.0) + 1.0 / (k + i)
    for i, rid in enumerate(bm25_results, 1):
        rank[rid] = rank.get(rid, 0.0) + 1.0 / (k + i)
    return [rid for rid, _ in sorted(rank.items(), key=lambda x: x[1], reverse=True)]
```

### 4.5 Реранкер (ONNX) и граф

- BAAI/bge‑reranker‑v2‑m3 (int8), ORT: `intra_op=2`, `inter_op=1`, `OMP_NUM_THREADS=2`; lazy‑load, TTL 10–15 мин; единственный инстанс.
- 1‑hop расширение через SQLite (recursive CTE), NetworkX — офлайн.

SQL

```
-- 1-hop расширение соседей
WITH RECURSIVE neigh(block) AS (
  SELECT :block_id
  UNION
  SELECT e.dst FROM edges e WHERE e.src = :block_id
  UNION
  SELECT e.src FROM edges e WHERE e.dst = :block_id
)
SELECT c.block_id, c.content 
FROM chunks c 
JOIN neigh n ON c.block_id = n.block
LIMIT 4;
```

### 4.6 Локальные ONNX‑модели

- Векторизация: multilingual‑e5‑small — постоянно в RAM.
- Экстрактивный QA fallback: xlm‑roberta‑base‑squad2 — lazy‑load с TTL.

Bash

```
# Рекомендуемые env для CPU
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ONNXRUNTIME_THREADING_ENABLE=1
```

### 4.7 LLM‑шлюз (Router)

- Primary: OpenRouter (в т.ч. Qwen‑Coder/Chat для кода), Gemini через gemini‑cli.
- Paid fallback: мини‑модель с дневным лимитом $1–2.
- Политики: token‑bucket per provider, health‑probes, circuit breaker, exponential backoff + jitter, anti‑thrash (≥30–60 с).

Python

```
class LLMRouter:
    def __init__(self, providers, budget_daily_usd=2.0):
        self.providers = providers  # [{'name': 'openrouter', ...}, ...]
        self.budget = budget_daily_usd

    async def generate(self, prompt, meta):
        for prov in self.sorted_providers():
            if not prov.circuit_closed(): 
                continue
            try:
                return await prov.call(prompt, timeout=8)
            except TransientError:
                prov.mark_failure()
                continue
        # Fallback: экстрактивный QA или "только поиск"
        return None
```

### 4.8 Семантический кеш

- Порог cosine ~0.90±0.02; TTL 1–7 дней; LRU. Хранит вектор запроса и валидный JSON‑ответ.

Python

```
class SemanticCache:
    def __init__(self, threshold=0.90, ttl_days=7, max_size=1000):
        self.threshold, self.ttl, self.max = threshold, ttl_days*86400, max_size
        self.store = {}  # key: id -> {emb, ts, response}

    def get(self, q_emb):
        best, score = None, 0.0
        now = time.time()
        for rec in self.store.values():
            if now - rec['ts'] > self.ttl: 
                continue
            s = cos_sim(q_emb, rec['emb'])
            if s > self.threshold and s > score:
                best, score = rec['response'], s
        return best
```

---

## 5) Поток запроса

text

```
Query
 ├─ Embed + LangDetect ──> Semantic Cache (hit? → return)
 ├─ Parallel Retrieval:
 │   ├─ Qdrant (top200, ef=64→96→128)
 │   └─ FTS5  (top200; RU via content_lemma)
 ├─ RRF (k=60) → 60–80 кандидатов
 ├─ Cross-Encoder Rerank (batch) → top-12
 ├─ 1-hop Graph Expansion (+2–4)
 ├─ LLM via Router → Strict JSON (auto-repair ≤2 tries)
 │    └─ If LLM unavailable → Extractive QA → else "search-only"
 └─ Cache save + Metrics + Logs → Response(JSON)
```

---

## 6) Дорожная карта (4–5 недель)

- Недели 1–2: инфра (Docker/Compose, Traefik TLS, UFW), ZRAM (zstd, 50–75%), мониторинг (Netdata/Prometheus‑экспорт); индексация SiYuan; FTS5 с RU‑леммами и стоп‑листом; Qdrant 384d int8; гибридный поиск + RRF; офлайн‑бенчмарки.
- Неделя 3: ONNX‑реранкер (lazy, TTL, батчинг), семафор 1 активный; 1‑hop CTE; семантический кеш (threshold/TTL/eviction).
- Неделя 4: LLM‑router (OpenRouter + gemini‑cli + платный fallback), circuit breaker/health‑пробы/anti‑thrash; JSON‑контракт + авто‑репейр; калибровка гейтинга (Platt/изотоника).
- Неделя 5: нагрузочные (до 10 параллельных), P50/P95 per stage; OOM‑дрилл с watchdog; E2E бэкап/восстановление; минимальный Streamlit/Gradio UI.

---

## 7) Надёжность, безопасность, эксплуатация

### ZRAM и лимиты потоков

Bash

```
# Установка и настройка ZRAM (пример)
sudo apt update && sudo apt install -y zram-tools
echo -e "ALGO=zstd\nPERCENT=50" | sudo tee /etc/default/zramswap
sudo systemctl enable --now zramswap

# Потоки для ORT/BLAS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

- Memory watchdog: при RAM>90% — выгрузка lazy‑моделей, частичная чистка кеша, переход в degraded mode.

### Безопасность

- Traefik TLS, UFW default‑deny (80/443 + SSH по белому IP).
- Изоляция в внутренней Docker‑сети; секреты через env/secrets; ротация ключей.

### Бэкапы

Bash

```
# Ежедневные бэкапы вне VPS (пример rsync+cron)
0 3 * * * rsync -avz /opt/exocortex/data/ user@backup:/bk/exocortex-$(date +\%F)/
```

- Qdrant snapshots + SQLite .backup; ежемесячные учения восстановления.

---

## 8) Контекст использования и чанкинг

- Объём: старт 50–100k чанков; к 3‑му месяцу 200–300k.
- Языки: 70% RU / 30% EN.
- Запросы: ~60% факты, ~40% обзор.
- Чанкинг: 250–400 токенов, overlap 20–40; поле lang; границы секций.
- Контекст LLM: ~3000–3500 токенов (12–16 чанков + инструкции).

---

## 9) Строгий JSON‑контракт ответа

JSON

```
{
  "answer": "string",
  "citations": [
    {"doc_id": "string", "block_id": "string", "score": 0.0, "char_span": [0, 0]}
  ],
  "confidence": 0.0,
  "status": "ok",
  "mode": "llm"
}
```

Правила: только валидный JSON; max_citations 6–8; при "no_data" — пустые answer/citations и confidence=0.0.

---

## 10) Ресурсный профиль (оценка RAM @300k чанков)

- Qdrant (384d int8 + HNSW + payload): ~0.5–1.0 GB
- e5‑small ONNX: ~130–400 MB (в RAM постоянно)
- bge‑reranker ONNX: ~400–600 MB (lazy, TTL)
- Экстрактивный QA ONNX: ~400–500 MB (lazy)
- Python+сервисы (Traefik, мониторинг): ~1.5–2.5 GB
- SQLite FTS5 + граф: ~200–300 MB
- Пики: ~3–5.5 GB — укладывается в 8 GB при ZRAM и ограничении параллелизма.

---

## 11) Go‑Live чек‑лист

- [ ]  Индексация: SiYuan → чанкинг → RU‑леммы/стоп‑лист → Qdrant(384d int8) + FTS5.
- [ ]  RRF: 200/200 → k=60 → top 60–80; реранк → top‑12; 1‑hop расширение (+2–4).
- [ ]  ONNX: e5‑small — постоянно; bge‑reranker и QA — lazy, TTL; ORT `intra=2`, `inter=1`; `OMP_NUM_THREADS=2`.
- [ ]  Router: OpenRouter + gemini‑cli (+ Qwen Code для задач по коду), дневной лимит платного fallback; health‑пробы, circuit breaker, anti‑thrash.
- [ ]  Семантический кеш: threshold ~0.90; TTL; LRU; метрики hit‑rate.
- [ ]  JSON‑схема: строгая валидация + авто‑репейр (≤2 попытки).
- [ ]  ZRAM включён; алерты; memory watchdog.
- [ ]  Бэкапы вне VPS; ежемесячный recovery‑drill.
- [ ]  Нагрузочные: до 10 параллельных; отчёт P50/P95 per stage; без OOM.

---

## 12) Известные риски и митигации

- P95 реранка > 2 с: уменьшить кандидатов (80→60→50), пред‑скоры (смесь норм. BM25/vec), батчинг.
- Нестабильность free LLM: router + circuit breaker, платный fallback, семкеш.
- RU‑recall без лемм: обязательно pymorphy3 и стоп‑лист.
- OOM при росте: ZRAM, lazy‑модели, memmap, ограничение параллелизма; при 500k+ — пересмотр m/квант./хранения.

---

## 13) Скелет пайплайна (референс)

Python

```
class RAGPipeline:
    def __init__(self, embedder, qdrant, fts5, reranker, graph, llm_router, semcache):
        self.embedder = embedder
        self.qdrant = qdrant
        self.fts5 = fts5
        self.reranker = reranker
        self.graph = graph
        self.llm = llm_router
        self.cache = semcache

    async def query(self, text: str, lang: str | None = None):
        q_emb = self.embedder.embed_query(text)
        if cached := self.cache.get(q_emb):
            return cached

        lang = lang or detect_lang(text)
        vec_ids = self.qdrant.search(q_emb, top_k=200, ef=64)
        bm25_ids = self.fts5.search(text, lang=lang, top_k=200)
        fused = rrf_merge(vec_ids, bm25_ids, k=60)[:80]

        cand = fetch_chunks(fused)
        ranked = await self.reranker.rank(text, cand)[:12]

        expanded = self.graph.expand_1hop([c.id for c in ranked], limit=4)
        final_ctx = assemble_context(ranked, expanded)

        prompt = build_prompt(text, final_ctx)
        resp = await self.llm.generate(prompt, meta={"lang": lang})
        out = ensure_valid_json(resp) or await self.llm.retry_repair(prompt)

        if not out:
            out = extractive_fallback(text, final_ctx) or search_only(final_ctx)

        self.cache.put(q_emb, out)
        return out
```
