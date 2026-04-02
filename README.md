# Cloth recsys

Сервис рекомендаций одежды. Пользователь оценивает вещи из каталога H&M, а система на основе лайков подбирает похожие товары тремя способами: по визуальному сходству, по поведению похожих покупателей и через обученную модель ранжирования CatBoost.

## Как это работает

```
H&M images
    ↓
EfficientNet-B0 → эмбеддинг
    ↓
FAISS-поиск → топ потенциально релевантных объектов + косинусные скоры
    ↓
CatBoostRanker → ранжирование моделью
    ↓
Streamlit UI — лайки, три вкладки рекомендаций
```

### Encoder (`encoder/`)

Модель на базе предобученной **EfficientNet-B0**, последний слой заменён головой `Linear(1280→512) → BatchNorm1d → L2-normalize`. Используется `TripletMarginLoss`: вещи с одним `product_code` должны быть близко, с разными далеко

### Ranker (`ranker/`)

Двухэтапный пайплайн для сокращения вычислительной сложности: retrieval (FAISS) → ranking (CatBoostRanker)

**Признаки:**

| # | Признак | Тип |
|---|---------|-----|
| 0 | `faiss_score` | косинусное сходство из FAISS |
| 1–5 | `item_product_type`, `colour_group`, `department`, `section`, `garment_group` | категориальные |
| 6 | `item_global_freq` | log1p(число покупок товара) |
| 7 | `user_cat_purchase_count` | кол-во покупок пользователя в той же категории |
| 8 | `user_dept_purchase_count` | кол-во покупок пользователя в том же отделе |
| 9–11 | `user_avg_price`, `item_avg_price`, `price_diff` | ценовые сигналы |

Модель обучается с временным сплитом: история до `2020-08-25` — признаки, покупки после — объекты для обучения, loss: `YetiRank`

### Engine (`engine/`)

Класс `FashionEngine` реализует три стратегии:

**Визуальное сходство** (`get_visually_similar` / вкладка "Ваш стиль") — FAISS `IndexFlatIP`: поиск ближайших соседей по косинусному расстоянию

**Look-alike** (`get_lookalike_recommendations` / вкладка "С этим сочетают") — усредняет эмбеддинги лайкнутых вещей, находит топ-50 покупателей с похожим профилем, агрегирует их покупки по частоте

**Ранжирование** (`get_ranked_recommendations` / вкладка "AI Подбор") — FAISS извлекает 200 кандидатов, затем `FashionRanker` ранжирует их с помощью обученной модели

### UI (`ui/`)

Streamlit-приложение с тремя вкладками рекомендаций

## Данные

Датасет [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

## Запуск

### 1. Подготовка окружения

Скопируйте содержимое датасета в папку с проектом, затем установите библиотеки:

```bash
pip install torch torchvision pandas Pillow tqdm faiss-cpu streamlit catboost
```

### 2. Обучение encoder

```bash
python encoder/train.py
```

### 3. Извлечение эмбеддингов

```bash
python encoder/extract_embeddings.py --checkpoint checkpoints/model_final.pt
```

### 4. Обучение ранжировщика

```bash
python ranker/train.py
```

### 5. Запуск UI

```bash
streamlit run ui/app.py
```

## Технологии

- **PyTorch** — модель и обучение encoder
- **FAISS** — точный поиск ближайших соседей
- **CatBoost** — обученная модель ранжирования (YetiRank)
- **pandas**, **numpy** — обработка данных
- **Streamlit** — веб-интерфейс
