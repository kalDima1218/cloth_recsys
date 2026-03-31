import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "engine"))

from engine import FashionEngine

DATA_ROOT        = _PROJECT_ROOT
ARTICLES_CSV     = DATA_ROOT / "articles.csv"
TRANSACTIONS_CSV = DATA_ROOT / "transactions_train.csv"
EMBEDDINGS_NPZ   = DATA_ROOT / "outputs" / "embeddings.npz"
ENGINE_CACHE     = DATA_ROOT / "engine_cache"

COLD_START_N = 10
GRID_COLS    = 4
RECS_PER_TAB = 8
BATCH_N      = 10


def article_image_path(article_id: int) -> Path:
    s = str(int(article_id)).zfill(10)
    return DATA_ROOT / "images" / s[:3] / f"{s}.jpg"


def load_image(article_id: int) -> Optional[Image.Image]:
    try:
        return Image.open(article_image_path(article_id)).convert("RGB")
    except (FileNotFoundError, OSError):
        return None


@st.cache_resource(show_spinner="Загрузка FAISS-индексов...")
def load_engine() -> FashionEngine:
    if (ENGINE_CACHE / "item.faiss").exists():
        return FashionEngine.load(str(ENGINE_CACHE))
    if not EMBEDDINGS_NPZ.exists():
        st.error(f"Не найден файл эмбеддингов: **{EMBEDDINGS_NPZ}**")
        st.stop()
    return FashionEngine(str(EMBEDDINGS_NPZ), str(TRANSACTIONS_CSV))


@st.cache_data(show_spinner=False)
def load_articles() -> pd.DataFrame:
    cols = ["article_id", "prod_name", "product_group_name", "product_type_name"]
    return pd.read_csv(ARTICLES_CSV, usecols=cols).set_index("article_id")


@st.cache_data(show_spinner=False)
def load_prices() -> Dict[int, float]:
    if not TRANSACTIONS_CSV.exists():
        return {}
    df = pd.read_csv(TRANSACTIONS_CSV, usecols=["article_id", "price"])
    return df.groupby("article_id")["price"].mean().to_dict()


def _sample_random_batch(engine: FashionEngine, n: int = BATCH_N) -> List[int]:
    seen = set(st.session_state.pool)
    candidates = [aid for aid in engine._art_id_to_idx.keys()
                  if aid not in seen and article_image_path(aid).exists()]
    return random.sample(candidates, min(n, len(candidates)))


def _build_cold_start_pool(engine: FashionEngine) -> List[int]:
    articles = load_articles()
    df = articles[articles.index.isin(engine._art_id_to_idx.keys())].copy()
    df = df[df.index.map(lambda aid: article_image_path(int(aid)).exists())]

    if TRANSACTIONS_CSV.exists():
        counts = pd.read_csv(TRANSACTIONS_CSV, usecols=["article_id"])["article_id"].value_counts()
        df["_pop"] = df.index.map(counts).fillna(0)
        df = df.sort_values("_pop", ascending=False).head(5_000)

    groups = df["product_group_name"].unique().tolist()
    random.shuffle(groups)

    result: List[int] = []
    for group in groups:
        if len(result) >= COLD_START_N:
            break
        subset = df[df["product_group_name"] == group]
        if not subset.empty:
            result.append(int(subset.index[0]))

    for aid in df[~df.index.isin(result)].index:
        if len(result) >= COLD_START_N:
            break
        result.append(int(aid))

    return result


def init_session() -> None:
    if "pool" not in st.session_state:
        engine = load_engine()
        st.session_state.pool         = _build_cold_start_pool(engine)
        st.session_state.current_idx  = 0
        st.session_state.liked_ids    = []
        st.session_state.disliked_ids = []
        st.session_state.style_limit  = RECS_PER_TAB
        st.session_state.look_limit   = RECS_PER_TAB


def _article_info(article_id: int, articles: pd.DataFrame, prices: Dict) -> Dict:
    try:
        row = articles.loc[int(article_id)]
        name     = row["prod_name"]
        category = row.get("product_type_name") or row.get("product_group_name") or "—"
    except KeyError:
        name, category = f"Артикул {article_id}", "—"
    return {"name": name, "category": category}


def render_swipe_card(article_id: int, articles: pd.DataFrame, prices: Dict) -> None:
    info = _article_info(article_id, articles, prices)
    img  = load_image(article_id)

    img_col, info_col = st.columns([3, 2], gap="large")

    with img_col:
        if img:
            st.image(img, width="stretch")
        else:
            st.markdown(
                "<div style='height:420px;background:#f5f5f5;border-radius:16px;"
                "display:flex;align-items:center;justify-content:center;'>"
                "<span style='font-size:64px'>👗</span></div>",
                unsafe_allow_html=True,
            )

    with info_col:
        st.markdown(f"## {info['name']}")
        st.markdown(f"<span style='color:#888;font-size:.9rem'>{info['category']}</span>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#bbb'>article_id: {article_id}</small>", unsafe_allow_html=True)
        st.markdown("<br>" * 3, unsafe_allow_html=True)

        like_col, dislike_col = st.columns(2, gap="small")
        with like_col:
            if st.button("👍  Нравится", key="btn_like", width="stretch", type="primary"):
                st.session_state.liked_ids.append(article_id)
                st.session_state.current_idx += 1
                st.rerun()
        with dislike_col:
            if st.button("👎  Пропустить", key="btn_dislike", width="stretch"):
                st.session_state.disliked_ids.append(article_id)
                st.session_state.current_idx += 1
                st.rerun()


def render_product_grid(
    article_ids: List[int],
    articles: pd.DataFrame,
    prices: Dict,
    key_prefix: str = "grid",
) -> None:
    if not article_ids:
        st.info("Рекомендаций пока нет — поставьте больше лайков.")
        return

    liked_set    = set(st.session_state.liked_ids)
    disliked_set = set(st.session_state.disliked_ids)

    cols = st.columns(GRID_COLS, gap="small")
    for i, aid in enumerate(article_ids):
        with cols[i % GRID_COLS]:
            img = load_image(aid)
            if img:
                st.image(img, width="stretch")
            else:
                st.markdown(
                    "<div style='height:160px;background:#f5f5f5;border-radius:8px;"
                    "display:flex;align-items:center;justify-content:center;'>"
                    "<span>👗</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption(f"**{_article_info(aid, articles, prices)['name']}**")

            if aid in liked_set:
                st.markdown("<center>✅</center>", unsafe_allow_html=True)
            elif aid in disliked_set:
                st.markdown("<center>❌</center>", unsafe_allow_html=True)
            else:
                b1, b2 = st.columns(2, gap="small")
                with b1:
                    if st.button("👍", key=f"{key_prefix}_like_{aid}", width="stretch", help="Нравится"):
                        st.session_state.liked_ids.append(aid)
                        st.rerun()
                with b2:
                    if st.button("👎", key=f"{key_prefix}_dis_{aid}", width="stretch", help="Не нравится"):
                        st.session_state.disliked_ids.append(aid)
                        st.rerun()


def render_recommendations(engine: FashionEngine, articles: pd.DataFrame, prices: Dict) -> None:
    liked: List[int] = st.session_state.liked_ids
    if not liked:
        return

    st.markdown("---")
    tab_style, tab_look = st.tabs(["👗 Ваш стиль", "🛍 С этим сочетают"])
    seen = set(liked) | set(st.session_state.disliked_ids)

    with tab_style:
        st.markdown("##### Похожие вещи — от каждой понравившейся")
        valid_liked = [a for a in liked if a in engine._art_id_to_idx]
        if valid_liked:
            limit        = st.session_state.style_limit
            results      = []
            visited      = set(seen)
            sources      = valid_liked.copy()
            random.shuffle(sources)
            source_cycle = sources * (limit // len(sources) + 1)

            for source_id in source_cycle:
                if len(results) >= limit:
                    break
                vec_src   = engine._embeddings_f16[engine._art_id_to_idx[source_id]].astype(np.float32)
                neighbors = engine.get_visually_similar(vec_src, top_k=15)
                fresh     = [a for a in neighbors if a not in visited]
                if not fresh:
                    continue
                results.append(fresh[0])
                visited.add(fresh[0])

            render_product_grid(results, articles, prices, key_prefix="style")
            if st.button("🔀 Обновить", key="refresh_style"):
                st.rerun()
        else:
            st.warning("Нет лайкнутых товаров в индексе.")

    with tab_look:
        st.markdown("##### Покупают люди с похожим вкусом")
        limit     = st.session_state.look_limit
        lookalike = engine.get_lookalike_recommendations(liked, top_k=limit + 10)
        lookalike = [a for a in lookalike if a not in seen][:limit]
        render_product_grid(lookalike, articles, prices, key_prefix="look")
        if st.button("Показать ещё", key="more_look"):
            st.session_state.look_limit += RECS_PER_TAB
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Fashion Tinder", page_icon="👗", layout="centered", initial_sidebar_state="collapsed")

    st.markdown(
        """
        <style>
        .stButton > button {
            border-radius: 28px;
            font-size: 1.05rem;
            padding: .55rem 1.2rem;
            transition: transform .1s;
        }
        .stButton > button:active { transform: scale(.97); }
        [data-testid="stTabs"] button { font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0'>👗 Fashion Tinder</h1>"
        "<p style='text-align:center;color:#888;margin-top:4px'>Оцените вещи — получите персональные рекомендации</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    engine   = load_engine()
    articles = load_articles()
    prices   = load_prices()

    init_session()

    pool = st.session_state.pool
    idx  = st.session_state.current_idx

    if idx < len(pool):
        st.progress(idx / len(pool), text=f"Оценено {idx} из {len(pool)}")
        st.markdown("")
        render_swipe_card(pool[idx], articles, prices)
    else:
        liked_n    = len(st.session_state.liked_ids)
        disliked_n = len(st.session_state.disliked_ids)
        st.success(f"✅ Пул оценён! Понравилось: **{liked_n}**, пропущено: **{disliked_n}**. Посмотрите рекомендации ниже или загрузите ещё вещи.")
        if st.button("🔄  Ещё вещей", type="primary", width="content"):
            st.session_state.pool.extend(_sample_random_batch(engine))
            st.rerun()

    liked_n    = len(st.session_state.liked_ids)
    disliked_n = len(st.session_state.disliked_ids)
    if liked_n or disliked_n:
        m1, m2 = st.columns(2)
        m1.metric("👍 Нравится", liked_n)
        m2.metric("👎 Пропущено", disliked_n)

    render_recommendations(engine, articles, prices)


if __name__ == "__main__":
    main()
