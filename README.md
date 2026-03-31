# Cloth recsys

Сервис рекомендаций одежды. Пользователь оценивает вещи из каталога H&M, а система на основе лайков подбирает похожие товары двумя способами: по визуальному сходству и по поведению похожих покупателей.

## Как это работает

```
H&M images
    ↓
EfficientNet-B0 → эмбеддинг
    ↓
FAISS-поиск + Коллаборативная фильтрация
    ↓
Streamlit UI — лайки, рекомендации
```

### Encoder (`encoder/`)

Модель на базе предобученной **EfficientNet-B0**, последний слой заменён головой `Linear(1280→512) → BatchNorm1d → L2-normalize`. Используется `TripletMarginLoss`: вещи с одним `product_code` должны быть близко, с разными далеко.

### Engine (`engine/`)

Класс `FashionEngine` реализует две стратегии:

**Визуальное сходство** (`get_visually_similar` или вкладка "Ваш стиль") — FAISS `IndexFlatIP`: точный поиск ближайших соседей по косинусному сходству среди 105k векторов. Возвращает топ вещей.

**Look-alike** (`get_lookalike_recommendations` или вкладка "С этим сочетают") — усредняет эмбеддинги лайкнутых вещей в эмбеддинг вкуса пользователя, находит топ покупателей с похожим профилем, собирает их покупки и ранжирует по частоте.

### UI (`ui/`)

Streamlit-приложение

## Данные

Датасет [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

## Запуск

### 1. Подготовка окружения

Скопируйте содержимое датасета в папку с проектом, далее установите библиотеки

```bash
pip install torch torchvision pandas Pillow tqdm faiss-cpu streamlit
```

### 2. Обучение модели

```bash
python encoder/train.py
```

### 3. Извлечение эмбеддингов

```bash
python encoder/extract_embeddings.py --checkpoint checkpoints/model_final.pt
```

### 4. Запуск UI

```bash
streamlit run ui/app.py
```

## Технологии

- **PyTorch** — модель и обучение
- **FAISS** — приближённый поиск ближайших соседей
- **pandas**, **numpy** — обработка данных
- **Streamlit** — веб-интерфейс
