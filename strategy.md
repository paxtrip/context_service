# Проект: «Цифровой Экзокортекс» — Персональная База Знаний с RAG

**Версия:** 11.0 (Expert-Reviewed Blueprint, Production-Ready)  
**Статус:** Утверждено к разработке (критические улучшения включены)  
**Дата:** 2024


```
[//]: # Это **Стратегический Блюпринт**. Он отвечает на вопросы **"ЧТО?"** и **"ПОЧЕМУ?"**. Он задает видение, принципы, метрики успеха и высокоуровневую архитектуру. Его сила в структуре, управлении проектом, постановке целей и обосновании решений.
```
---

## 1. TL;DR (Краткое содержание)

**Что это:** Персональная RAG-система на базе SiYuan, Qdrant и Python, критически оптимизированная для надежной работы на бюджетном VPS (4vCPU/8GB RAM).

**Цель:** Создать «второй мозг», который дает точные, проверяемые и быстрые ответы на русском и английском языках, с гарантированной защитой от галлюцинаций.

**Стек:**

- Гибридный поиск: **двухступенчатый** (BM25 с лемматизацией + векторы e5-small 384d) → RRF fusion → реранкинг bge-m3 (ONNX)
- **Graceful degradation:** Circuit breaker для LLM API → экстрактивный QA fallback → чистый поиск
- Доступ к LLM через интеллектуальную ротацию внешних API с резервным платным провайдером

**Сроки:** MVP — **4-5 недель** (включая калибровку на русскоязычных данных).

**Стоимость:** VPS $10-20/мес + платный LLM fallback $5-10/мес.

---

## 2. Ключевые принципы и операционные политики

### 2.1 Точность через мультиязычный гибридный поиск

**Ядро системы** — двухступенчатый гибридный поиск:

1. **Широкий поиск:**
    
    - Векторный (Qdrant): top-200, семантическая близость
    - Лексический (SQLite FTS5 + **pymorphy3**): top-200, лемматизированный BM25
2. **Fusion:** Reciprocal Rank Fusion (k=60) → top-80 кандидатов
    
3. **Reranking:** Cross-encoder (bge-reranker-v2-m3) → top-12 финальных чанков
    
4. **Графовое расширение:** 1-hop по связям документов → +2-4 контекстных чанка
    

**Критическое отличие от v10.0:** Обязательная **лемматизация русского языка** на уровне индексации и поиска. Без этого recall на русском снижается на 30-40%.

### 2.2 Многоуровневая стратегия отказоустойчивости

text

```
┌─────────────────────────────────────────────┐
│ Уровень 1: Гибридный поиск + LLM (основной) │
│   - Circuit breaker для ротации провайдеров │
│   - Семантический кеш (threshold=0.87)      │
└──────────────────┬──────────────────────────┘
                   │ Провайдеры недоступны
                   ▼
┌─────────────────────────────────────────────┐
│ Уровень 2: Экстрактивный QA (ONNX, локально)│
│   - xlm-roberta-base-squad2                 │
│   - Покрывает 60-70% фактических вопросов   │
└──────────────────┬──────────────────────────┘
                   │ Нет ответа с confidence > 0.3
                   ▼
┌─────────────────────────────────────────────┐
│ Уровень 3: Режим "только поиск"             │
│   - Топ-3 фрагмента + выдержки (highlight)  │
└─────────────────────────────────────────────┘
```

### 2.3 Контроль галлюцинаций (Gating)

- **Калибровка порога:** Platt scaling / изотоническая регрессия на валидационном наборе
- **Обязательное цитирование:** JSON-схема с валидацией на бэкенде
- **Confidence tracking:** Возврат метрик релевантности и причины отказа (low_coverage / low_agreement)

### 2.4 Адаптивное извлечение

При низкой уверенности первичного поиска (`max(scores) < 0.6`):

- `ef_search`: 64 → 96 → 128 (три ступени)
- Повторный RRF с расширенными кандидатами
- Если и это не помогает → fallback на уровень 2

---

## 3. Цели и метрики успеха

### 3.1 Золотой набор вопросов

**Размер:** Старт — 30 вопросов, цель — **80-120 вопросов** (70% RU / 30% EN).

**Типы:**

- 60% короткие фактические ("Какая столица X?")
- 40% обзорные ("Объясни концепцию Y")

### 3.2 Метрики качества

|Метрика|Целевое значение|Измерение|
|---|---|---|
|**Recall@15** (до реранкинга)|≥ 0.80|Доля вопросов с релевантным документом в топ-15|
|**Recall@12** (после реранкинга)|≥ 0.85|Финальный набор для LLM|
|**Groundedness (RAGAS)**|≥ 0.80|Оценка обоснованности ответа источниками|
|**Citation Precision**|≥ 0.95|Доля валидных цитирований|
|**Fusion Gain**|≥ +0.15|Δ Recall между гибридным и лучшим моно-поиском|

### 3.3 Latency Budget (P95 на 4vCPU)

|Этап|Целевое время|Допустимое время|
|---|---|---|
|Гибридный поиск (Qdrant + FTS5)|≤ 1.0 сек|1.5 сек|
|RRF fusion|≤ 0.3 сек|0.5 сек|
|Реранкинг (80 кандидатов)|≤ 1.5 сек|2.0 сек|
|Графовое расширение|≤ 0.4 сек|0.6 сек|
|LLM API (основной режим)|≤ 5.0 сек|8.0 сек|
|**Экстрактивный QA (fallback)**|≤ 2.0 сек|3.0 сек|
|**Итого (E2E, основной)**|≤ 8.2 сек|12.6 сек|
|**Итого (E2E, fallback)**|≤ 5.2 сек|7.6 сек|

---

## 4. Технологический стек и конфигурация (MVP)

### 4.1 Хранилище знаний: SiYuan

**Процесс индексации:**

- **Инкрементальная:** Поллинг по `updated_at` (интервал: 5 минут)
- **Чанкинг:** 250-400 токенов, overlap 20-40 токенов, с сохранением границ секций
- **Дедупликация:** SHA-256 хэш контента чанка
- **Батчинг:** 1000-5000 чанков за транзакцию

**Метаданные чанка:**

Python

```
{
    "block_id": str,        # ID блока SiYuan
    "doc_id": str,          # ID документа
    "section_path": str,    # Иерархия заголовков
    "lang": str,            # ru/en (auto-detect)
    "created_at": int,      # timestamp
    "updated_at": int,
    "content_hash": str     # SHA-256
}
```

---

### 4.2 Оркестратор: Thin Wrapper (Python)

**Архитектурное решение:**  
Отказ от полноценного LlamaIndex в пользу **минималистичной обертки** с выборочным использованием компонентов.

Python

```
# Структура проекта:
exocortex/
├── core/
│   ├── embedder.py          # ONNX e5-small (постоянно в RAM)
│   ├── reranker.py          # Lazy-loaded bge-m3
│   ├── extractive_qa.py     # ✨ НОВОЕ: xlm-roberta fallback
│   └── semantic_cache.py    # ✨ НОВОЕ: векторный кеш ответов
├── retrieval/
│   ├── vector_store.py      # Qdrant wrapper
│   ├── bm25_store.py        # ✨ УЛУЧШЕНО: SQLite FTS5 + лемматизация
│   ├── fusion.py            # RRF (k=60)
│   └── graph_expander.py    # 1-hop расширение
├── llm/
│   ├── router.py            # ✨ НОВОЕ: Circuit breaker + ротация
│   └── providers.py         # Адаптеры для API
└── pipeline.py              # Главный оркестратор
```

**Используем из LlamaIndex:**

- `resolve_embed_model()` — только для удобной загрузки ONNX моделей
- **Всё остальное** — собственная реализация (~500 строк кода)

**Выигрыш:** Экономия ~200 MB RAM, полный контроль потока выполнения.

---

### 4.3 Векторная память и гибридный поиск

#### 4.3.1 Векторная часть (Qdrant)

**Конфигурация коллекции:**

YAML

```
vectors:
  size: 384
  distance: Cosine

quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: true

hnsw_config:
  m: 16                    # Баланс точность/память
  ef_construct: 100
  
optimizers_config:
  memmap_threshold: 50000  # Используем mmap для больших коллекций
  
performance:
  max_search_threads: 2    # Не больше половины vCPU
```

**Payload индексы:**

Python

```
# Для быстрой фильтрации
client.create_payload_index(
    collection_name="chunks",
    field_name="lang",
    field_schema="keyword"
)
client.create_payload_index(
    collection_name="chunks",
    field_name="doc_id",
    field_schema="keyword"
)
```

**Адаптивный поиск:**

Python

```
async def adaptive_search(query_vector, confidence_threshold=0.6):
    # Попытка 1: быстрый поиск
    results = await qdrant.search(
        vector=query_vector,
        limit=200,
        search_params={"ef": 64}
    )
    
    if max(r.score for r in results) < confidence_threshold:
        # Попытка 2: средняя точность
        results = await qdrant.search(
            vector=query_vector,
            limit=200,
            search_params={"ef": 96}
        )
    
    if max(r.score for r in results) < confidence_threshold:
        # Попытка 3: максимальная точность
        results = await qdrant.search(
            vector=query_vector,
            limit=200,
            search_params={"ef": 128}
        )
    
    return results
```

#### 4.3.2 Лексическая часть (SQLite FTS5 + лемматизация) ✨

**Критическое изменение:** Индексация и поиск с лемматизацией для русского языка.

**Схема таблицы:**

SQL

```
-- Основная таблица чанков
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    block_id TEXT UNIQUE NOT NULL,
    doc_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_lemma TEXT NOT NULL,  -- ✨ НОВОЕ: лемматизированный текст
    lang TEXT NOT NULL,
    section_path TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    content_hash TEXT
);

-- FTS5 полнотекстовый индекс
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,                      -- Оригинальный текст (для EN)
    content_lemma,                -- Лемматизированный (для RU)
    block_id UNINDEXED,
    tokenize='unicode61 remove_diacritics 2'
);

-- Индексы для графа и фильтрации
CREATE INDEX idx_doc_id ON chunks(doc_id);
CREATE INDEX idx_lang ON chunks(lang);
CREATE INDEX idx_hash ON chunks(content_hash);
```

**Конфигурация лемматизатора:**

Python

```
from pymorphy3 import MorphAnalyzer
import re

class RussianLemmatizer:
    def __init__(self):
        self.morph = MorphAnalyzer()
        # Русские стоп-слова (частицы, предлоги)
        self.stopwords = set([
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со',
            'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да',
            'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только',
            'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
            'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'при'
        ])
    
    def lemmatize(self, text: str) -> str:
        """Лемматизация с удалением стоп-слов"""
        tokens = re.findall(r'\w+', text.lower())
        lemmas = []
        
        for token in tokens:
            if token in self.stopwords:
                continue
            parsed = self.morph.parse(token)[0]
            lemmas.append(parsed.normal_form)
        
        return ' '.join(lemmas)
```

**PRAGMA-оптимизации:**

SQL

```
PRAGMA journal_mode = WAL;           -- Параллельные чтения
PRAGMA synchronous = NORMAL;         -- Баланс скорость/надежность
PRAGMA temp_store = MEMORY;          -- Временные таблицы в RAM
PRAGMA mmap_size = 268435456;        -- 256 MB memory-mapped I/O
PRAGMA cache_size = -64000;          -- 64 MB кеш страниц
```

**Поиск с учетом языка:**

Python

```
async def bm25_search(query: str, lang: str, top_k: int = 200):
    # Лемматизация только для русского
    if lang == 'ru':
        query_processed = lemmatizer.lemmatize(query)
        column = 'content_lemma'
    else:
        query_processed = query.lower()
        column = 'content'
    
    results = cursor.execute(f"""
        SELECT 
            block_id,
            rank,
            snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
        FROM chunks_fts
        WHERE {column} MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query_processed, top_k))
    
    return results.fetchall()
```

#### 4.3.3 Reciprocal Rank Fusion (RRF)

Python

```
def reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    k: int = 60,
    top_n: int = 80
) -> list:
    """
    Объединение результатов с RRF.
    
    Args:
        k: Параметр сглаживания (60 эмпирически оптимален)
        top_n: Количество финальных кандидатов для реранкинга
    """
    scores = {}
    
    for rank, result in enumerate(vector_results, 1):
        doc_id = result.id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result['block_id']
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Сортируем по скору, берем топ-N
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]
```

---

### 4.4 Локальные ML-модели (ONNX)

#### 4.4.1 Векторизация (постоянно в RAM)

**Модель:** `intfloat/multilingual-e5-small` (384 dimensions)

**Конфигурация ONNX Runtime:**

Python

```
import onnxruntime as ort

session = ort.InferenceSession(
    "e5-small.onnx",
    providers=["CPUExecutionProvider"],
    sess_options={
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
        "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL,
        "graph_optimization_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    }
)

# Переменные окружения
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
```

**Использование префиксов:**

Python

```
def embed_query(text: str):
    return model.encode(f"query: {text}")

def embed_passage(text: str):
    return model.encode(f"passage: {text}")
```

**Потребление памяти:** ~400 MB (постоянно)

#### 4.4.2 Реранкинг (lazy-loaded пул)

**Модель:** `BAAI/bge-reranker-v2-m3`

Python

```
from datetime import datetime, timedelta

class LazyReranker:
    def __init__(self, model_path: str, idle_timeout: int = 600):
        self.model_path = model_path
        self.model = None
        self.last_used = None
        self.idle_timeout = timedelta(seconds=idle_timeout)
    
    def _load_model(self):
        self.model = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            sess_options={"intra_op_num_threads": 2}
        )
        self.last_used = datetime.now()
    
    def _unload_if_idle(self):
        if (self.model and self.last_used and 
            datetime.now() - self.last_used > self.idle_timeout):
            del self.model
            self.model = None
    
    async def rerank(self, query: str, documents: list, top_k: int = 12):
        self._unload_if_idle()
        
        if self.model is None:
            self._load_model()
        
        self.last_used = datetime.now()
        
        # Реранкинг
        scores = self.model.run(
            None,
            {"query": query, "documents": documents}
        )
        
        # Сортируем и возвращаем топ-K
        ranked = sorted(
            zip(documents, scores[0]),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]
```

**Потребление памяти:** ~600 MB (когда активна)

#### 4.4.3 Экстрактивный QA Fallback ✨ НОВОЕ

**Модель:** `deepset/xlm-roberta-base-squad2` (поддержка RU/EN)

Python

```
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer

class ExtractiveFallback:
    def __init__(self, model_name: str = "deepset/xlm-roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForQuestionAnswering.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider"
        )
    
    def answer(
        self,
        question: str,
        contexts: list[str],
        confidence_threshold: float = 0.3
    ) -> dict | None:
        """
        Попытка экстракции ответа из контекстов.
        
        Returns:
            {"text": str, "score": float, "context_id": int} или None
        """
        for idx, context in enumerate(contexts[:3]):  # Проверяем топ-3
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            outputs = self.model(**inputs)
            
            # Декодируем ответ
            answer_start = outputs.start_logits.argmax()
            answer_end = outputs.end_logits.argmax() + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0][answer_start:answer_end]
                )
            )
            
            # Вычисляем confidence
            score = (
                outputs.start_logits[0][answer_start] +
                outputs.end_logits[0][answer_end - 1]
            ).item() / 2
            
            if score > confidence_threshold:
                return {
                    "text": answer.strip(),
                    "score": score,
                    "context_id": idx,
                    "mode": "extractive"
                }
        
        return None
```

**Потребление памяти:** ~450 MB (lazy-loaded, аналогично реранкеру)

---

### 4.5 Графовая память (SQLite + NetworkX)

**Схема графа:**

SQL

```
CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY,
    source_block_id TEXT NOT NULL,
    target_block_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,  -- 'parent', 'child', 'ref', 'sibling', 'prev', 'next'
    weight REAL DEFAULT 1.0,
    UNIQUE(source_block_id, target_block_id, edge_type)
);

CREATE INDEX idx_source ON graph_edges(source_block_id, edge_type);
CREATE INDEX idx_target ON graph_edges(target_block_id, edge_type);
```

**1-hop расширение:**

Python

```
async def expand_context(seed_blocks: list[str], max_neighbors: int = 4):
    """
    Расширяет контекст на 1 шаг по графу связей.
    Приоритет: parent > ref > sibling > next
    """
    expanded = []
    
    for block_id in seed_blocks:
        neighbors = cursor.execute("""
            SELECT target_block_id, edge_type, weight
            FROM graph_edges
            WHERE source_block_id = ?
            ORDER BY 
                CASE edge_type
                    WHEN 'parent' THEN 1
                    WHEN 'ref' THEN 2
                    WHEN 'sibling' THEN 3
                    WHEN 'next' THEN 4
                    ELSE 5
                END,
                weight DESC
            LIMIT ?
        """, (block_id, max_neighbors))
        
        expanded.extend([row[0] for row in neighbors.fetchall()])
    
    return expanded
```

**Offline анализ (NetworkX):**

Python

```
import networkx as nx

def analyze_graph_offline(db_path: str):
    """Периодический анализ для вычисления PageRank и кластеров"""
    G = nx.DiGraph()
    
    # Загрузка из SQLite
    edges = cursor.execute("SELECT source_block_id, target_block_id, weight FROM graph_edges")
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    
    # PageRank для оценки важности блоков
    pagerank = nx.pagerank(G, weight='weight')
    
    # Сохраняем в отдельную таблицу
    cursor.executemany(
        "INSERT OR REPLACE INTO block_metrics (block_id, pagerank) VALUES (?, ?)",
        pagerank.items()
    )
```

---

### 4.6 LLM провайдеры и ротация ✨ УЛУЧШЕНО

#### 4.6.1 Circuit Breaker + Интеллектуальная ротация

Python

```
from circuitbreaker import circuit
from datetime import datetime, timedelta
import asyncio

class LLMProvider:
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        is_free: bool,
        rate_limit: str  # "30/10min"
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.is_free = is_free
        self.rate_limit = rate_limit
        
        # Метрики здоровья
        self.errors_count = 0
        self.total_requests = 0
        self.latencies = []  # Последние 100 запросов
        self.last_error = None

class LLMRouter:
    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self.semantic_cache = SemanticCache(threshold=0.87, ttl_days=7)
    
    def _get_provider_score(self, provider: LLMProvider) -> float:
        """
        Скоринг провайдера: чем меньше, тем лучше.
        Учитывает ошибки и латентность.
        """
        error_penalty = provider.errors_count * 10
        latency_penalty = (
            sum(provider.latencies[-20:]) / len(provider.latencies[-20:])
            if provider.latencies else 0
        )
        free_bonus = -5 if provider.is_free else 0  # Предпочитаем бесплатные
        
        return error_penalty + latency_penalty + free_bonus
    
    @circuit(failure_threshold=3, recovery_timeout=60, expected_exception=Exception)
    async def _call_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        timeout: float = 8.0
    ) -> str:
        """Вызов с circuit breaker"""
        start_time = datetime.now()
        
        try:
            async with asyncio.timeout(timeout):
                response = await self._make_api_request(provider, prompt)
            
            # Обновляем метрики
            latency = (datetime.now() - start_time).total_seconds()
            provider.latencies.append(latency)
            provider.total_requests += 1
            provider.errors_count = max(0, provider.errors_count - 0.5)  # Decay
            
            return response
            
        except Exception as e:
            provider.errors_count += 1
            provider.last_error = str(e)
            raise
    
    async def generate(
        self,
        prompt: str,
        query_embedding: np.ndarray = None
    ) -> dict | None:
        """
        Главный метод генерации с кешем и ротацией.
        """
        # Шаг 1: Проверка семантического кеша
        if query_embedding is not None:
            cached = self.semantic_cache.get(query_embedding)
            if cached:
                return {"answer": cached, "source": "cache"}
        
        # Шаг 2: Сортируем провайдеров по здоровью
        sorted_providers = sorted(
            self.providers,
            key=self._get_provider_score
        )
        
        # Шаг 3: Пробуем провайдеров по очереди
        last_error = None
        for provider in sorted_providers:
            try:
                response = await self._call_provider(provider, prompt)
                
                # Успех - сохраняем в кеш
                if query_embedding is not None:
                    self.semantic_cache.set(query_embedding, response)
                
                return {
                    "answer": response,
                    "source": provider.name,
                    "is_free": provider.is_free
                }
                
            except CircuitBreakerError:
                # Провайдер временно отключен, пробуем следующий
                continue
            except Exception as e:
                last_error = e
                continue
        
        # Все провайдеры недоступны
        return None
```

#### 4.6.2 Конфигурация провайдеров

Python

```
providers = [
    LLMProvider(
        name="groq-llama3-70b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        is_free=True,
        rate_limit="30/10min"
    ),
    LLMProvider(
        name="openrouter-free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        is_free=True,
        rate_limit="10/min"
    ),
    LLMProvider(
        name="openai-gpt4o-mini",  # ✨ Платный fallback
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        is_free=False,
        rate_limit="500/min"
    )
]

router = LLMRouter(providers)
```

**Бюджет:** Платный провайдер срабатывает только при отказе бесплатных. При 3k токенов/запрос и цене GPT-4o-mini $0.15/1M токенов → $0.00045/запрос. Бюджет $5/мес = ~11,000 fallback-запросов.

---

### 4.7 Семантический кеш ✨ НОВОЕ

Python

```
import numpy as np
from datetime import datetime, timedelta
from collections import OrderedDict

class SemanticCache:
    """
    LRU-кеш с семантическим поиском по косинусной близости.
    """
    def __init__(
        self,
        threshold: float = 0.87,
        ttl_days: int = 7,
        max_size: int = 1000
    ):
        self.threshold = threshold
        self.ttl = timedelta(days=ttl_days)
        self.max_size = max_size
        self.cache = OrderedDict()  # {query_id: (embedding, response, timestamp)}
    
    def _cleanup_expired(self):
        """Удаляем устаревшие записи"""
        now = datetime.now()
        expired = [
            qid for qid, (_, _, ts) in self.cache.items()
            if now - ts > self.ttl
        ]
        for qid in expired:
            del self.cache[qid]
    
    def get(self, query_embedding: np.ndarray) -> str | None:
        """Поиск похожего запроса в кеше"""
        self._cleanup_expired()
        
        for qid, (cached_emb, response, ts) in self.cache.items():
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            
            if similarity >= self.threshold:
                # Перемещаем в конец (LRU)
                self.cache.move_to_end(qid)
                return response
        
        return None
    
    def set(self, query_embedding: np.ndarray, response: str):
        """Сохранение ответа в кеш"""
        qid = hash(query_embedding.tobytes())
        
        # LRU eviction
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[qid] = (query_embedding, response, datetime.now())
```

**Impact:** Ожидаемое снижение LLM-запросов на 30-40% после накопления кеша.

---

## 5. Главный Pipeline (интеграция всех компонентов)

Python

```
class ExocortexRAG:
    def __init__(self):
        self.embedder = ONNXEmbedder("e5-small.onnx")
        self.vector_store = QdrantClient(...)
        self.bm25_store = BM25Store(db_path="chunks.db")
        self.reranker = LazyReranker("bge-reranker-v2-m3.onnx")
        self.graph = GraphExpander(db_path="chunks.db")
        self.llm_router = LLMRouter(providers)
        self.extractive_qa = ExtractiveFallback()
        self.semantic_cache = SemanticCache()
    
    async def query(self, user_query: str, lang: str = "ru") -> dict:
        """
        Главный метод запроса с graceful degradation.
        """
        trace_id = uuid.uuid4()
        start_time = datetime.now()
        
        # Этап 1: Векторизация запроса
        query_embedding = self.embedder.encode_query(user_query)
        
        # Этап 2: Двухступенчатый поиск
        vector_results = await self.vector_store.search(
            vector=query_embedding,
            limit=200,
            ef_search=64  # Адаптивно повысится при необходимости
        )
        
        bm25_results = await self.bm25_store.search(
            query=user_query,
            lang=lang,
            top_k=200
        )
        
        # Этап 3: RRF Fusion
        fused_results = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=60,
            top_n=80
        )
        
        # Этап 4: Reranking
        reranked = await self.reranker.rerank(
            query=user_query,
            documents=[r['content'] for r in fused_results],
            top_k=12
        )
        
        # Этап 5: Графовое расширение
        seed_blocks = [r['block_id'] for r in reranked]
        expanded_blocks = await self.graph.expand_context(
            seed_blocks,
            max_neighbors=4
        )
        
        final_contexts = reranked + expanded_blocks[:4]
        
        # Этап 6: Генерация ответа (с fallback)
        response = await self._generate_with_fallback(
            query=user_query,
            contexts=final_contexts,
            query_embedding=query_embedding,
            trace_id=trace_id
        )
        
        # Метрики
        latency = (datetime.now() - start_time).total_seconds()
        
        return {
            **response,
            "trace_id": str(trace_id),
            "latency_ms": latency * 1000,
            "num_contexts": len(final_contexts)
        }
    
    async def _generate_with_fallback(
        self,
        query: str,
        contexts: list,
        query_embedding: np.ndarray,
        trace_id: uuid.UUID
    ) -> dict:
        """
        Трехуровневая стратегия генерации ответа.
        """
        # Уровень 1: LLM через ротацию провайдеров
        prompt = self._build_prompt(query, contexts)
        llm_response = await self.llm_router.generate(
            prompt=prompt,
            query_embedding=query_embedding
        )
        
        if llm_response:
            answer = self._parse_json_response(llm_response['answer'])
            if answer and answer.get('confidence', 0) > 0.6:
                return {
                    "answer": answer['answer'],
                    "citations": answer['citations'],
                    "confidence": answer['confidence'],
                    "mode": "llm",
                    "llm_provider": llm_response['source']
                }
        
        # Уровень 2: Экстрактивный QA (локально)
        extractive_answer = self.extractive_qa.answer(
            question=query,
            contexts=[c['content'] for c in contexts]
        )
        
        if extractive_answer:
            return {
                "answer": extractive_answer['text'],
                "citations": [contexts[extractive_answer['context_id']]],
                "confidence": extractive_answer['score'],
                "mode": "extractive"
            }
        
        # Уровень 3: Только поиск с выдержками
        return {
            "answer": "",
            "contexts": contexts[:3],
            "snippets": [c['snippet'] for c in contexts[:3]],
            "confidence": 0.0,
            "mode": "search_only",
            "message": "Релевантные фрагменты найдены, но автоматический ответ невозможен"
        }
    
    def _build_prompt(self, query: str, contexts: list) -> str:
        """
        Промпт с жестким требованием JSON-формата.
        """
        context_text = "\n\n".join([
            f"[{i+1}] {c['content']} (ID: {c['block_id']})"
            for i, c in enumerate(contexts)
        ])
        
        return f"""Ты — ассистент персональной базы знаний. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленные контексты.

ВАЖНО:
- Если в контекстах нет информации для ответа, верни пустой answer и confidence=0.0
- Каждое утверждение ОБЯЗАТЕЛЬНО подтверждай цитатой
- Ответ должен быть СТРОГО в JSON-формате без дополнительного текста

КОНТЕКСТЫ:
{context_text}

ВОПРОС: {query}

ФОРМАТ ОТВЕТА (только валидный JSON):
{{
  "answer": "текст ответа с фактами из контекстов",
  "citations": [
    {{"block_id": "ID блока", "quote": "точная цитата", "relevance": 0.95}}
  ],
  "confidence": 0.85
}}"""
    
    def _parse_json_response(self, raw_response: str) -> dict | None:
        """Парсинг и валидация JSON-ответа от LLM"""
        try:
            # Извлекаем JSON из markdown-блоков (если есть)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                raw_response = json_match.group(1)
            
            data = json.loads(raw_response)
            
            # Валидация схемы
            required_keys = {'answer', 'citations', 'confidence'}
            if not required_keys.issubset(data.keys()):
                return None
            
            # Валидация цитат
            for cite in data['citations']:
                if 'block_id' not in cite:
                    return None
            
            return data
            
        except json.JSONDecodeError:
            return None
```

---

## 6. Дорожная карта развития (Roadmap)

### 📅 Фаза 1: MVP (4-5 недель)

#### Неделя 1: Инфраструктура и критические компоненты

- **День 1-2:**
    
    - [ ]  Настройка VPS (Ubuntu 24.04)
    - [ ]  Установка zram (4GB, zstd compression)
    - [ ]  Docker + Docker Compose
    - [ ]  Настройка UFW, Traefik с TLS
    - [ ]  Мониторинг (Netdata или cAdvisor)
- **День 3-5:**
    
    - [ ]  ✨ Интеграция pymorphy3
    - [ ]  SQLite FTS5: схема + лемматизация + стоп-слова RU
    - [ ]  Бенчмарк лемматизации на 10k блоков
    - [ ]  Тестирование BM25-поиска на RU
- **День 6-7:**
    
    - [ ]  Настройка Qdrant (коллекция + квантизация)
    - [ ]  ONNX e5-small: загрузка + бенчмарк (latency, RAM)
    - [ ]  Первичная индексация тестового набора (1000 блоков)

#### Неделя 2: Поисковый Pipeline

- **День 8-10:**
    
    - [ ]  Реализация адаптивного векторного поиска (ef_search: 64→96→128)
    - [ ]  Интеграция BM25 + векторный поиск
    - [ ]  RRF fusion (k=60, экспериментально проверить 40/60/80)
- **День 11-12:**
    
    - [ ]  ONNX bge-reranker-v2-m3
    - [ ]  Lazy-loading пул с таймаутом 10 минут
    - [ ]  Бенчмарк реранкинга: 50/80/100 кандидатов → выбор оптимума
- **День 13-14:**
    
    - [ ]  ✨ Экстрактивный QA: xlm-roberta-base-squad2 ONNX
    - [ ]  Интеграция в fallback-логику
    - [ ]  Тестирование на 20 фактических вопросах

#### Неделя 3: LLM и Граф

- **День 15-17:**
    
    - [ ]  LLMRouter: circuit breaker + скоринг провайдеров
    - [ ]  Интеграция Groq + OpenRouter + OpenAI (fallback)
    - [ ]  ✨ Семантический кеш (threshold=0.87, TTL=7d)
    - [ ]  Стресс-тест ротации (искусственные отказы)
- **День 18-19:**
    
    - [ ]  JSON-схема ответа + валидатор
    - [ ]  Промпт-инжиниринг (тестирование на разных моделях)
    - [ ]  Обработка edge cases (невалидный JSON, отказ от ответа)
- **День 20-21:**
    
    - [ ]  Граф: схема SQLite + индексы
    - [ ]  1-hop расширение контекста
    - [ ]  A/B тест: с графовым расширением vs без

#### Неделя 4: Калибровка и Тестирование

- **День 22-24:**
    
    - [ ]  Золотой набор: 30 вопросов (21 RU / 9 EN, факт/обзор)
    - [ ]  Офлайн-метрики: Recall@15, Recall@12, Fusion Gain
    - [ ]  Калибровка gating-порога (Platt scaling)
- **День 25-26:**
    
    - [ ]  Load testing: 10 параллельных запросов, измерение P95 latency
    - [ ]  Тестирование OOM-сценариев (мониторинг swap/zram)
    - [ ]  Streamlit UI для ручного тестирования
- **День 27-28:**
    
    - [ ]  Настройка бэкапов (ежедневно, на внешний storage)
    - [ ]  Учение восстановления из бэкапа
    - [ ]  Финальная проверка по чек-листу

#### Неделя 5: Буфер и Документация

- **День 29-35:**
    - [ ]  Отладка выявленных проблем
    - [ ]  Оптимизация узких мест (если P95 > budget)
    - [ ]  Документация: README, API-спецификация, runbook
    - [ ]  Подготовка к Go-Live

---

### 📅 Фаза 2: Оптимизация и Мониторинг (2-3 недели)

- **Увеличение золотого набора:** До 80-120 вопросов
- **Плановая регуляризация:**
    - Расширение семантического кеша (threshold tuning)
    - Динамическая калибровка RRF-весов
- **WebSocket API SiYuan:** Real-time индексация вместо поллинга
- **Advanced мониторинг:**
    - Prometheus + Grafana
    - Алерты на OOM, high latency, LLM failures
- **PII-маскирование:** Регэкспы для email/телефонов перед отправкой в LLM

---

### 📅 Фаза 3: "Взросление" архитектуры (4-6 недель)

- **Полный отказ от LlamaIndex:** Миграция на прямые вызовы
- **MCP/HTTP API:** REST/WebSocket сервер для интеграции с агентами
- **Персонализация:**
    - Адаптация промптов на основе фидбека
    - User-specific контекстные фильтры
- **Продвинутый граф:**
    - PageRank для весов блоков
    - Кластеризация документов
    - Временное ранжирование (новизна vs стабильность)

---

## 7. Надежность, безопасность и эксплуатация

### 7.1 Управление ресурсами VPS

#### Zram вместо swap ✨

Bash

```
# Установка zram-tools
sudo apt install zram-tools

# Конфигурация /etc/default/zramswap
ALGO=zstd              # Быстрое сжатие
PERCENT=50             # 4GB для 8GB RAM
PRIORITY=100

# Применение
sudo service zramswap reload

# Проверка
sudo zramctl
```

**Обоснование:** Zram в 2-3 раза быстрее дискового swap, так как использует сжатую RAM вместо I/O.

#### Переменные окружения для ONNX

Bash

```
# docker-compose.yml
environment:
  - OMP_NUM_THREADS=2
  - MKL_NUM_THREADS=2
  - ONNXRUNTIME_MAX_THREADS=2
```

#### Мониторинг памяти

Python

```
import psutil
import logging

async def memory_watchdog():
    """Автоматический graceful restart при критическом потреблении RAM"""
    while True:
        mem = psutil.virtual_memory()
        
        if mem.percent > 90:
            logging.critical(f"RAM usage critical: {mem.percent}%")
            # Выгружаем lazy-loaded модели
            reranker.unload()
            extractive_qa.unload()
        
        await asyncio.sleep(30)
```

---

### 7.2 Безопасность

#### Сетевая изоляция

YAML

```
# docker-compose.yml
networks:
  internal:
    driver: bridge
    internal: true  # Изолированная сеть для БД
  
  external:
    driver: bridge

services:
  qdrant:
    networks:
      - internal
  
  app:
    networks:
      - internal
      - external
  
  traefik:
    networks:
      - external
```

#### TLS и Firewall

Bash

```
# UFW правила
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp   # SSH (только с белого IP)
sudo ufw allow 80/tcp   # HTTP → HTTPS redirect
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Traefik автоматически получает Let's Encrypt сертификаты
```

#### Secrets Management

Bash

```
# .env (не коммитится в Git)
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
QDRANT_API_KEY=...

# Docker secrets (production)
echo "gsk_..." | docker secret create groq_api_key -
```

---

### 7.3 Бэкапы и Восстановление

#### Стратегия бэкапов

Bash

```
#!/bin/bash
# backup.sh - Ежедневный бэкап (cron: 0 3 * * *)

DATE=$(date +%Y%m%d)
BACKUP_DIR="/mnt/backup-volume"  # Внешний volume (не на VPS!)

# Qdrant snapshot
docker exec qdrant curl -X POST http://localhost:6333/collections/chunks/snapshots

# SQLite + граф
sqlite3 /opt/exocortex/data/chunks.db ".backup ${BACKUP_DIR}/chunks_${DATE}.db"

# Конфигурация
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz /opt/exocortex/config/

# Ротация (храним 30 последних дней + 12 месячных)
find ${BACKUP_DIR} -name "*.db" -mtime +30 -delete
```

#### Учения восстановления (ежемесячно)

Bash

```
# restore-drill.sh
# 1. Создать тестовый Docker Compose stack
# 2. Восстановить последний бэкап
# 3. Запустить валидацию (golden set)
# 4. Задокументировать результаты
```

---

### 7.4 Наблюдаемость

#### Структурированные логи

Python

```
import structlog

logger = structlog.get_logger()

# В каждом запросе
logger.info(
    "query_completed",
    trace_id=str(trace_id),
    query_length=len(user_query),
    num_contexts=len(final_contexts),
    mode=response['mode'],
    latency_ms=latency,
    llm_provider=response.get('llm_provider'),
    cache_hit=response.get('source') == 'cache'
)
```

#### Метрики (Prometheus-совместимые)

Python

```
from prometheus_client import Counter, Histogram, Gauge

# Определение метрик
query_counter = Counter('rag_queries_total', 'Total queries', ['mode', 'lang'])
latency_histogram = Histogram('rag_latency_seconds', 'Query latency', ['stage'])
cache_hit_rate = Gauge('rag_cache_hit_rate', 'Semantic cache hit rate')

# Использование
query_counter.labels(mode='llm', lang='ru').inc()
latency_histogram.labels(stage='reranking').observe(1.2)
```

---

## 8. Контекст использования (для калибровки)

|Параметр|Старт|Через 3 месяца|Через 12 месяцев|
|---|---|---|---|
|**Объем индекса**|50-100k чанков|200-300k|500k-1M|
|**Размер вектора БД**|~40 MB|~115 MB|~290 MB|
|**Запросов в день**|10-20|50-100|200-300|
|**Доля кеш-хитов**|5-10% (холодный старт)|30-40%|50-60%|

**Языковое распределение:** 70% русский, 30% английский

**Типы запросов:**

- 60% короткие фактические ("Когда основан X?")
- 40% обзорные ("Как работает Y?")

**Бюджет контекста LLM:** ~3000 токенов (12 чанков по ~200 токенов + 4 графовых расширения)

---

## 9. Чек-лист готовности к Go-Live MVP

### Критические компоненты (блокеры запуска)

#### Инфраструктура

- [ ]  VPS настроен (Ubuntu 24.04, 4vCPU/8GB RAM)
- [ ]  ✨ Zram 4GB (zstd, percent=50) вместо swap
- [ ]  Docker + Docker Compose установлены
- [ ]  Traefik настроен с TLS (Let's Encrypt)
- [ ]  UFW firewall активен (22/80/443)
- [ ]  Мониторинг развернут (Netdata/cAdvisor)

#### Поиск и Индексация

- [ ]  ✨ **Русская лемматизация (pymorphy3) работает**
- [ ]  SQLite FTS5 настроен с content_lemma
- [ ]  Стоп-слова RU загружены
- [ ]  Qdrant: коллекция 384d + scalar int8 квантизация
- [ ]  HNSW индекс: m=16, ef_construct=100
- [ ]  Payload индексы созданы (lang, doc_id)

#### ML-модели (ONNX)

- [ ]  e5-small загружена, бенчмарк пройден (<300ms на запрос)
- [ ]  OMP_NUM_THREADS=2 установлен
- [ ]  bge-reranker в lazy-pool (idle timeout 10 мин)
- [ ]  ✨ xlm-roberta-base-squad2 (экстрактивный QA) работает
- [ ]  Бенчмарк реранкинга: 80 кандидатов за <2 сек

#### Pipeline

- [ ]  Двухступенчатый поиск: 200→80→12 реализован
- [ ]  RRF k=60 протестирован на золотом наборе
- [ ]  Адаптивный ef_search (64→96→128) работает
- [ ]  Графовое расширение 1-hop (+4 чанка)
- [ ]  JSON-схема ответа валидируется

#### LLM и Fallback

- [ ]  ✨ Circuit breaker для провайдеров настроен
- [ ]  Groq + OpenRouter + OpenAI (платный fallback) работают
- [ ]  ✨ Семантический кеш (threshold=0.87, TTL=7d) активен
- [ ]  Экстрактивный QA fallback протестирован
- [ ]  Режим "только поиск" возвращает snippets

#### Тестирование и Калибровка

- [ ]  Золотой набор: минимум 30 вопросов (70% RU / 30% EN)
- [ ]  Recall@15 ≥ 0.80 достигнут
- [ ]  Groundedness ≥ 0.80 (на выборке)
- [ ]  Latency P95 в пределах budget (E2E <13 сек)
- [ ]  Load test: 10 параллельных запросов без OOM
- [ ]  Калибровка gating-порога выполнена

#### Эксплуатация

- [ ]  Бэкапы настроены (ежедневно, внешний storage)
- [ ]  Восстановление из бэкапа протестировано
- [ ]  Структурированные логи с trace_id
- [ ]  Memory watchdog активен (alert при >90% RAM)
- [ ]  ✨ Streamlit UI для тестирования развернут

---

## 10. Известные риски и митигации

|Риск|Вероятность|Impact|Митигация|
|---|---|---|---|
|**OOM при индексации 300k чанков**|Средняя|Высокий|Батчинг по 5k, zram, мониторинг, автоунлоад моделей|
|**Реранкинг >2 сек на 4vCPU**|Средняя|Средний|Сократить кандидатов до 60, использовать int8 ONNX|
|**Бесплатные LLM API нестабильны**|Высокая|Средний|Circuit breaker + платный fallback ($5/мес)|
|**Recall <0.8 на русском без лемматизации**|✨ Высокая|Критический|**Обязательная лемматизация pymorphy3**|
|**VPS перегрузка при 50+ запросах/день**|Низкая|Средний|Rate limiting, семантический кеш (снижает на 40%)|

---

## 11. Дополнительные ресурсы

### Референс-код (для быстрого старта)

#### RU-лемматизация + FTS5

Python

```
# lemmatizer.py
from pymorphy3 import MorphAnalyzer
import re

class RussianLemmatizer:
    # (см. раздел 4.3.2 для полной реализации)
    pass
```

#### RRF Fusion

Python

```
# fusion.py
def reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_n=80):
    # (см. раздел 4.3.3)
    pass
```

#### Circuit Breaker для LLM

Python

```
# llm_router.py
from circuitbreaker import circuit

class LLMRouter:
    # (см. раздел 4.6.1 для полной реализации)
    pass
```

### Рекомендуемые настройки Docker Compose

YAML

```
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - internal
    deploy:
      resources:
        limits:
          memory: 2G

  app:
    build: .
    volumes:
      - ./data/sqlite:/app/data
    environment:
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
    env_file:
      - .env
    networks:
      - internal
      - external
    depends_on:
      - qdrant
    deploy:
      resources:
        limits:
          memory: 4G

  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=your@email.com"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./data/traefik:/letsencrypt
    networks:
      - external

networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge
```

---

## 12. Заключение и следующие шаги

### Ключевые улучшения v11.0 vs v10.0

1. ✨ **Русская лемматизация** — критическое улучшение для 70% контента
2. ✨ **Экстрактивный QA fallback** — надежность +40%, независимость от LLM API
3. ✨ **Circuit breaker** — устойчивость к отказам провайдеров
4. ✨ **Семантический кеш** — снижение LLM-запросов на 30-40%
5. ✨ **Zram вместо swap** — в 2-3 раза быстрее при memory pressure
6. ✨ **Thin wrapper** — экономия ~200 MB RAM, контроль архитектуры

### Ожидаемые результаты MVP

- **Recall@15:** 0.82-0.85 (с лемматизацией)
- **Groundedness:** 0.80-0.88
- **Latency P95 (E2E):** 9-11 секунд (основной режим), 5-7 секунд (fallback)
- **Доля успешных ответов:** 85-90% (LLM + экстрактивный)
- **Потребление RAM:** 5-6 GB в пике, 3-4 GB в idle
- **Надежность:** 99%+ uptime (благодаря fallback)

### Первые шаги (сегодня)

1. Форкнуть репозиторий структуры проекта
2. Настроить VPS + zram
3. Протестировать pymorphy3 на sample-данных из SiYuan
4. Создать первые 10 вопросов для золотого набора

---

**Версия документа:** 11.0  
**Статус:** Production-Ready Blueprint  
**Дата последнего обновления:** 2024  
**Следующий обзор:** После завершения недели 2 (оценка прогресса pipeline)

---

_Этот документ является живым артефактом. Все изменения архитектуры, метрик и рисков должны отражаться здесь с обоснованием._
