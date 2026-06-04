import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #1a1d2e !important; }
section[data-testid="stSidebar"] { background-color: #f8f9fb !important; }
section[data-testid="stSidebar"] * { color: #1a1d2e !important; }

.section-title { font-size: 1.5rem; font-weight: 800; color: #2d3a8c; margin-bottom: 4px; }
.sub-label { color: #64748b; font-size: 0.88rem; margin-bottom: 20px; }

/* Tarjeta de pelicula seleccionada */
.base-card {
    background: #eef2ff;
    border: 2px solid #4338ca;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.base-card h3 { color: #2d3a8c; margin: 0 0 6px; font-size: 1.1rem; font-weight: 800; }
.base-card p  { color: #374151; margin: 2px 0; font-size: 0.85rem; }

/* Tarjetas de recomendacion */
.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px;
    margin-top: 12px;
}
.rec-card {
    background: #f8f9fb;
    border: 1px solid #e0e3ed;
    border-radius: 10px;
    overflow: hidden;
    transition: box-shadow .2s, border-color .2s;
}
.rec-card:hover { box-shadow: 0 4px 16px rgba(67,56,202,.15); border-color: #4338ca; }
.rec-card img {
    width: 100%;
    height: 260px;
    object-fit: cover;
    background: #e0e3ed;
    display: block;
}
.rec-card .card-body { padding: 10px 12px 12px; }
.rec-card .card-rank { color: #4338ca; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.rec-card .card-title { color: #1a1d2e; font-size: 0.88rem; font-weight: 700; margin: 2px 0 4px; line-height: 1.3; }
.rec-card .card-meta  { color: #64748b; font-size: 0.75rem; margin: 0; }
.rec-card .card-score { color: #2d3a8c; font-size: 0.78rem; font-weight: 600; margin-top: 4px; }
.sim-bar { height: 4px; background: #4338ca; border-radius: 2px; margin-top: 6px; }

/* Fuente del dataset */
.dataset-note {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 10px 14px;
    color: #166534;
    font-size: 0.82rem;
    margin-bottom: 16px;
}
div[data-testid="stMetric"] {
    background: #f8f9fb !important;
    border: 1px solid #e0e3ed !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
</style>
"""

# ─── Poster placeholder via IMDB ──────────────────────────────────────────────
def _poster_url(imdb_id: str) -> str:
    """
    Construye la URL del poster de IMDB usando su servicio de imagenes publico.
    Si no carga, el navegador muestra el fallback del HTML.
    """
    # IMDB tiene posters accesibles en esta ruta (tamaño M = 182x268)
    return f"https://img.omdbapi.com/?i={imdb_id}&apikey=trilogy&h=300"


def _omdb_poster(imdb_id: str) -> str:
    """URL alternativa con OMDb (sin key devuelve placeholder bonito)."""
    return f"https://via.placeholder.com/182x268/4338ca/ffffff?text=IMDB"


# ─── Carga y preparacion del dataset ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/imdb_top_movies_1980_2026.csv")
    df = df.dropna(subset=["title", "genres"])
    df["genres"]         = df["genres"].fillna("Unknown")
    df["average_rating"] = df["average_rating"].fillna(df["average_rating"].median())
    df["num_votes"]      = df["num_votes"].fillna(0)
    df["year"]           = df["year"].fillna(0).astype(int)
    df["runtime_minutes"]= df["runtime_minutes"].fillna(df["runtime_minutes"].median()).astype(int)
    # Texto combinado para TF-IDF
    df["combined"] = (
        df["genres"].str.replace(",", " ") + " " +
        df["title"] + " " +
        df["year"].astype(str)
    )
    return df.reset_index(drop=True)


@st.cache_data
def build_similarity(df):
    tfidf  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = tfidf.fit_transform(df["combined"])
    # Normalizar rating para agregarlo a la similitud
    scaler = MinMaxScaler()
    rating_norm = scaler.fit_transform(df[["average_rating"]]).flatten()
    sim    = cosine_similarity(matrix)
    # Boost por rating: similitud * 0.85 + rating_normalizado * 0.15
    sim_boosted = sim * 0.85 + np.outer(np.ones(len(df)), rating_norm) * 0.15
    return sim_boosted


def _genre_list(df: pd.DataFrame) -> list:
    genres = set()
    for g in df["genres"].dropna():
        for part in g.split(","):
            genres.add(part.strip())
    return sorted(genres)


def run():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sistema de Recomendacion de Peliculas</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-label">Basado en el dataset real de IMDB Top Movies 1980-2026 '
        '| Algoritmo: TF-IDF + Cosine Similarity con boost por calificacion</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dataset-note">'
        'Fuente de datos: <b>data/imdb_top_movies_1980_2026.csv</b> — '
        'Dataset real con las mejores peliculas de IMDB entre 1980 y 2026.'
        '</div>',
        unsafe_allow_html=True,
    )

    df  = load_data()
    sim = build_similarity(df)

    # ── Sidebar de configuracion ──────────────────────────────────────────────
    st.sidebar.markdown("### Configuracion")

    # Filtro de genero (opcional)
    all_genres   = _genre_list(df)
    genre_filter = st.sidebar.multiselect("Filtrar por genero", all_genres)

    # Filtro de anio
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider("Rango de anios", min_yr, max_yr, (1990, max_yr))

    # Rating minimo
    min_rating = st.sidebar.slider("Calificacion minima", 5.0, 9.5, 7.0, step=0.1)

    # Numero de recomendaciones
    n_recs = st.sidebar.slider("Numero de recomendaciones", 3, 12, 6)

    # ── Pelicula base ─────────────────────────────────────────────────────────
    # Pre-filtrar el selector por los filtros activos
    df_filtered = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1]) &
        (df["average_rating"] >= min_rating)
    ]
    if genre_filter:
        mask = df_filtered["genres"].apply(
            lambda g: any(gf in g for gf in genre_filter)
        )
        df_filtered = df_filtered[mask]

    if df_filtered.empty:
        st.warning("No hay peliculas que coincidan con los filtros. Ajuste los criterios.")
        return

    titles_sorted = df_filtered.sort_values("average_rating", ascending=False)["title"].tolist()
    selected_title = st.selectbox(
        "Selecciona una pelicula que te guste:",
        titles_sorted,
        index=0,
    )

    if st.button("Recomendar peliculas similares", type="primary"):
        # Indice en el df original
        base_idx = df[df["title"] == selected_title].index[0]
        base_row = df.iloc[base_idx]

        # Scores de similitud
        scores = list(enumerate(sim[base_idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Excluir la pelicula base
        scores = [(i, s) for i, s in scores if i != base_idx]

        # Aplicar filtros a las recomendaciones
        def _passes(idx):
            row = df.iloc[idx]
            if row["average_rating"] < min_rating:
                return False
            if row["year"] < year_range[0] or row["year"] > year_range[1]:
                return False
            if genre_filter and not any(gf in row["genres"] for gf in genre_filter):
                return False
            return True

        scores = [(i, s) for i, s in scores if _passes(i)][:n_recs]

        # ── Pelicula base ──────────────────────────────────────────────────────
        st.markdown("---")
        genres_display = base_row["genres"].replace(",", " | ")
        st.markdown(f"""
        <div class="base-card">
            <h3>Seleccionada: {base_row['title']} ({base_row['year']})</h3>
            <p>Generos: {genres_display}</p>
            <p>Calificacion IMDB: <b>{base_row['average_rating']}</b> &nbsp;|&nbsp;
               Votos: <b>{int(base_row['num_votes']):,}</b> &nbsp;|&nbsp;
               Duracion: <b>{base_row['runtime_minutes']} min</b></p>
            <p><a href="{base_row['imdb_url']}" target="_blank" style="color:#4338ca;">
               Ver en IMDB</a></p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metricas globales ──────────────────────────────────────────────────
        recs_df = pd.DataFrame([
            {
                "title":          df.iloc[i]["title"],
                "year":           df.iloc[i]["year"],
                "genres":         df.iloc[i]["genres"],
                "average_rating": df.iloc[i]["average_rating"],
                "num_votes":      df.iloc[i]["num_votes"],
                "runtime_minutes":df.iloc[i]["runtime_minutes"],
                "imdb_id":        df.iloc[i]["imdb_id"],
                "imdb_url":       df.iloc[i]["imdb_url"],
                "similarity":     round(s, 4),
            }
            for i, s in scores
        ])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recomendaciones",         len(recs_df))
        c2.metric("Rating promedio",          f"{recs_df['average_rating'].mean():.2f}")
        c3.metric("Similitud promedio",       f"{recs_df['similarity'].mean():.2%}")
        c4.metric("Duracion promedio (min)",  f"{recs_df['runtime_minutes'].mean():.0f}")

        # ── Tarjetas de peliculas ─────────────────────────────────────────────
        st.subheader("Peliculas recomendadas")

        # Mostrar en columnas (3 por fila)
        cols_per_row = 3
        rows = [scores[i:i+cols_per_row] for i in range(0, len(scores), cols_per_row)]

        for row_items in rows:
            cols = st.columns(cols_per_row)
            for col, (idx, sim_score) in zip(cols, row_items):
                row = df.iloc[idx]
                poster_url = f"https://img.omdbapi.com/?i={row['imdb_id']}&apikey=trilogy&h=300"
                genres_str = row["genres"].replace(",", " | ")
                bar_w      = int(sim_score * 100)

                with col:
                    # Poster via IMDB (carga desde internet)
                    st.markdown(f"""
                    <div class="rec-card">
                        <img
                          src="{poster_url}"
                          onerror="this.src='https://placehold.co/182x268/4338ca/ffffff?text={row['title'][:15].replace(' ', '+')}'"
                          alt="{row['title']}"
                        />
                        <div class="card-body">
                            <div class="card-rank">Similitud: {sim_score:.0%}</div>
                            <div class="card-title">{row['title']}</div>
                            <div class="card-meta">{row['year']} &nbsp;|&nbsp; {row['runtime_minutes']} min</div>
                            <div class="card-meta">{genres_str}</div>
                            <div class="card-score">IMDB: {row['average_rating']} &nbsp;({int(row['num_votes']):,} votos)</div>
                            <div class="sim-bar" style="width:{bar_w}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='text-align:center;margin-top:4px'>"
                        f"<a href='{row['imdb_url']}' target='_blank' "
                        f"style='color:#4338ca;font-size:0.78rem;'>Ver en IMDB</a></div>",
                        unsafe_allow_html=True,
                    )

        # ── Grafico de similitud ───────────────────────────────────────────────
        st.markdown("---")
        fig = px.bar(
            recs_df.sort_values("similarity"),
            x="similarity",
            y="title",
            orientation="h",
            color="average_rating",
            color_continuous_scale="Blues",
            title="Puntuacion de similitud con la pelicula seleccionada",
            labels={"similarity": "Similitud", "title": "Pelicula", "average_rating": "Rating IMDB"},
            template="simple_white",
        )
        fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

        # ── Tabla completa ─────────────────────────────────────────────────────
        with st.expander("Ver tabla completa de recomendaciones"):
            show_df = recs_df[["title", "year", "genres", "average_rating",
                                "num_votes", "runtime_minutes", "similarity"]].copy()
            show_df.columns = ["Titulo", "Anio", "Generos", "Rating IMDB",
                                "Votos", "Duracion (min)", "Similitud"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    else:
        # Estadisticas del dataset al inicio
        st.markdown("---")
        st.subheader("Estadisticas del dataset IMDB")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de peliculas", f"{len(df):,}")
        c2.metric("Anio mas antiguo",    df["year"].min())
        c3.metric("Rating promedio",     f"{df['average_rating'].mean():.2f}")
        c4.metric("Generos distintos",   len(_genre_list(df)))

        # Top 10 peliculas
        st.subheader("Top 10 peliculas mejor calificadas")
        top10 = df.nlargest(10, "average_rating")[
            ["title", "year", "genres", "average_rating", "num_votes", "runtime_minutes"]
        ].copy()
        top10.columns = ["Titulo", "Anio", "Generos", "Rating", "Votos", "Duracion (min)"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

        # Distribucion de generos
        genre_counts = {}
        for g in df["genres"].dropna():
            for part in g.split(","):
                part = part.strip()
                genre_counts[part] = genre_counts.get(part, 0) + 1
        genre_df = pd.DataFrame(
            list(genre_counts.items()), columns=["Genero", "Peliculas"]
        ).sort_values("Peliculas", ascending=False).head(15)

        fig = px.bar(
            genre_df, x="Genero", y="Peliculas",
            title="Distribucion de peliculas por genero",
            color="Peliculas", color_continuous_scale="Blues",
            template="simple_white",
        )
        fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
