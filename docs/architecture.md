# Архитектура системы

## Обзор

Система состоит из трёх независимых компонентов: **ML-пайплайн** (обучение модели), **Backend** (FastAPI сервис) и **Frontend** (Streamlit интерфейс).

```
Пользователь
     │
     ▼
┌─────────────────┐        ┌──────────────────────────────────┐
│  Frontend       │ HTTP   │  Backend (FastAPI)               │
│  Streamlit      │◄──────►│  POST /forward                   │
│  :8501          │        │  GET  /model-info                │
└─────────────────┘        │  GET  /history                   │
                           │  GET  /stats                     │
                           │  GET  /health                    │
                           └──────────┬───────────────────────┘
                                      │
                           ┌──────────▼───────────┐
                           │  ViT-B/16 model      │
                           │  service/artifacts/  │
                           │    model.pth         │
                           │    classes.json      │
                           │    preprocessing_    │
                           │    config.json       │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │  SQLite DB           │
                           │  service/history.db  │
                           └──────────────────────┘
```

---

## Компонент 1: ML-пайплайн (обучение)

**Инструменты:** PyTorch, MLflow, MinIO (S3), Hydra, Kaggle  
**Датасет:** HAM10000 (ISIC 2018), 10 015 изображений, 7 классов

### Предобработка данных

- Сплит по `lesion_id` (не по фото) - устраняет data leakage (`src/make_splits.py`)
- Resize до 224×224 пикселей
- Нормализация по средним/стд ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Аугментации при обучении: RandomResizedCrop, HorizontalFlip, ColorJitter

### Финальная модель: ViT-B/16

```
Входное изображение (3×224×224)
        │
        ▼ разбивка на патчи 16×16 → 196 патчей
        ▼ позиционные эмбеддинги
        ▼ 12 слоёв Self-Attention (Transformer Encoder)
        ▼ CLS токен → Linear(768, num_classes=7)
        ▼ Dropout(0.3)
        ▼ Softmax → вероятности 7 классов
```

### Порог меланомы

Если P(mel) ≥ 0.14 → предсказываем меланому (recall mel: 47% → 79%)  
Реализован в `src/threshold_tuning.py` и `service/app/main.py`.

### MLflow трекинг

- Сервер: SQLite backend + MinIO (S3) для артефактов
- Эксперимент: `skin_lesion_isic2018`
- Финальный run: `vit_b16_final`
- Зарегистрированная модель: `skin_lesion_vit_b16`, стадия `PRD`

---

## Компонент 2: Backend (FastAPI)

**Файлы:** `service/app/main.py`, `service/database.py`  
**Запуск:** `uvicorn service.app.main:app --host 0.0.0.0 --port 8000`

### При старте сервиса

1. Загружается `service/artifacts/classes.json`: список 7 классов
2. Загружается `service/artifacts/preprocessing_config.json`: параметры предобработки
3. Загружается `service/artifacts/model.pth`: веса ViT-B/16
4. Инициализируется SQLite база данных

### Эндпоинты

| Метод | URL           | Описание           |
| ----- | ------------- | ------------------ |
| POST  | `/forward`    | Анализ изображения |
| GET   | `/model-info` | Метаданные модели  |
| GET   | `/history`    | История запросов   |
| GET   | `/stats`      | Статистика времени |
| GET   | `/health`     | Статус сервиса     |

### /forward

**Хедеры запроса:**

- `x-top-k` - сколько вариантов диагноза вернуть (по умолчанию 3)
- `x-mode` - `global` или `windows` (по умолчанию `global`)
- `x-return-probs` - вернуть ли вероятности по всем классам (true/false)

**Форматы входа:**

- `multipart/form-data` с полем `image`
- `application/json` с полем `image_b64` (base64)

**Пример ответа:**

```json
{
  "predicted_class": "mel",
  "confidence": 0.72,
  "top_k": [
    { "class": "mel", "probability": 0.72 },
    { "class": "bkl", "probability": 0.18 },
    { "class": "nv", "probability": 0.07 }
  ],
  "mode": "global",
  "elapsed_ms": 145.3
}
```

### Режимы инференса

**Global:** изображение ресайзится до 224×224, подаётся целиком. Быстро.

**Windows:** изображение разбивается на перекрывающиеся фрагменты 224×224 с шагом 112px. Каждый фрагмент прогоняется через модель, вероятности усредняются по всем фрагментам.

### Логирование (SQLite)

Таблица `request_history`:
| Поле | Тип | Описание |
|------|-----|----------|
| timestamp | DateTime | Время запроса |
| elapsed_ms | Float | Время обработки (мс) |
| image_width/height | Integer | Размер входного изображения |
| predicted_class | String | Предсказанный класс |
| confidence | Float | Вероятность топ-1 |
| top3_classes | String | "mel,nv,bkl" |
| top3_probs | String | "0.72,0.18,0.07" |
| mode | String | "global" / "windows" |
| model_name | String | "skin-lesion-classifier" |
| architecture | String | "ViT-B/16" |
| stage | String | "PRD" |

---

## Компонент 3: Frontend (Streamlit)

**Файл:** `frontend/streamlit_app.py`  
**Запуск:** `streamlit run frontend/streamlit_app.py`

### Вкладки интерфейса

1. **Анализ изображения** - загрузка, выбор top-k и режима, результат с вероятностями
2. **История запросов** - таблица всех прошлых запросов из БД
3. **Статистика сервиса** - время обработки (mean, p50, p95, p99)
4. **Проверка состояния** - healthcheck
5. **Информация о модели** - данные из `/model-info`

---

## Структура файлов

```
├── src/
│   ├── common.py               # модели, датасет, обучение
│   ├── make_splits.py          # сплит по lesion_id
│   ├── threshold_tuning.py     # подбор порога меланомы
│   ├── dl_experiments.py       # обучение с MLflow
│   └── dl_demonstration.py     # демонстрационный скрипт
│
├── service/
│   ├── app/main.py             # FastAPI backend
│   ├── database.py             # SQLAlchemy модель + SQLite
│   ├── artifacts/
│   │   ├── model.pth           # веса ViT-B/16
│   │   ├── classes.json        # список 7 классов
│   │   └── preprocessing_config.json  # параметры предобработки
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
│
└── notebooks/
    ├── EDA.ipynb
    ├── ML_checkpoint3.ipynb
    ├── ML_checkpoint5.ipynb
    ├── ML_checkpoint6.ipynb
    └── 04-ml-flow-ipynb.ipynb
```

---

## Технологии

| Слой                  | Технология                      |
| --------------------- | ------------------------------- |
| Модель                | PyTorch, torchvision (ViT-B/16) |
| Трекинг экспериментов | MLflow + MinIO                  |
| Backend               | FastAPI + uvicorn               |
| База данных           | SQLite + SQLAlchemy             |
| Frontend              | Streamlit + Plotly              |
| Контейнеризация       | Docker                          |
| Данные                | HAM10000 (ISIC 2018)            |
