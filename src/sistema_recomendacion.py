"""
Sistema de Recomendacion de Videojuegos.

Recomendador basado en contenido sobre el dataset `games.csv`. A partir de un
juego que le guste al usuario, sugiere juegos parecidos usando:

    - TF-IDF sobre las caracteristicas del juego (genero, plataforma, etc.).
    - Similitud del coseno entre esos vectores.
    - Un pequeno ajuste (boost) por la nota de Metacritic.

Ademas intenta mostrar la portada real de cada juego obtenida de Wikipedia
(sin API key); si no la encuentra, genera una portada de respaldo con el titulo.

El punto de entrada es `run()`, invocado desde `Inicio.py`.
"""

import urllib.parse
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Estilo sobrio coherente con el resto del proyecto (acento azul #1e40af).
LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #0f172a !important; }
section[data-testid="stSidebar"] { background-color: #f8fafc !important; }
.section-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
.section-title::after { content:""; display:block; width:42px; height:3px;
    background:#1e40af; border-radius:2px; margin-top:8px; }
.sub-label { color: #64748b; font-size: 0.9rem; margin: 10px 0 18px; }

/* Tarjeta del juego seleccionado */
.base-card {
    background: #f1f5f9; border: 1px solid #1e40af; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 18px;
}
.base-card h3 { color: #1e3a8a; margin: 0 0 6px; font-size: 1.1rem; font-weight: 700; }
.base-card p  { color: #334155; margin: 2px 0; font-size: 0.86rem; }

/* Tarjetas de recomendacion */
.rec-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    overflow: hidden; transition: box-shadow .2s, border-color .2s;
}
.rec-card:hover { box-shadow: 0 4px 16px rgba(30,64,175,.15); border-color: #1e40af; }
.rec-card img { width: 100%; height: 240px; object-fit: cover;
    background: #e2e8f0; display: block; }
.rec-card .card-body { padding: 10px 12px 12px; }
.rec-card .card-rank { color: #1e40af; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em; }
.rec-card .card-title { color: #0f172a; font-size: 0.9rem; font-weight: 700;
    margin: 2px 0 4px; line-height: 1.3; }
.rec-card .card-meta  { color: #64748b; font-size: 0.76rem; margin: 0; }
.rec-card .card-score { color: #1e3a8a; font-size: 0.78rem; font-weight: 600; margin-top: 4px; }
.sim-bar { height: 4px; background: #1e40af; border-radius: 2px; margin-top: 6px; }

div[data-testid="stMetric"] {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; padding: 12px 16px !important;
}
</style>
"""

PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"


# ──────────────────────────────────────────────────────────────────────────────
# IMAGENES (portadas)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_cover(title):
    """Intenta obtener la URL de la portada de un juego desde Wikipedia (sin API key).

    Devuelve la URL de la miniatura si existe, o None si no la encuentra o falla la
    conexion. El resultado se cachea para no repetir la peticion por cada juego.
    """
    try:
        slug = urllib.parse.quote(title.strip().replace(" ", "_"))
        url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        r = requests.get(url, timeout=6, headers={"User-Agent": "lab-recsys/1.0"})
        if r.status_code == 200:
            thumb = r.json().get("thumbnail", {})
            if thumb and thumb.get("source"):
                return thumb["source"]
    except Exception:
        pass  # sin internet o titulo no encontrado: se usara el respaldo
    return None


def placeholder_cover(title):
    """Genera la URL de una portada de respaldo con el titulo del juego."""
    txt = urllib.parse.quote(title[:22])
    return f"https://placehold.co/300x400/1e40af/ffffff?text={txt}"


def cover_for(title):
    """Devuelve la portada real si existe; si no, la portada de respaldo."""
    return fetch_cover(title) or placeholder_cover(title)


# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACION DEL DATASET
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_games():
    """Carga `games.csv`, lo limpia y construye el texto para TF-IDF.

    El dataset trae el mismo juego en varias plataformas; se conserva una sola fila
    por titulo (la de mayor nota de Metacritic) para que las recomendaciones no se
    repitan. Se crea la columna `combined` con las caracteristicas que definen a
    cada juego, dando mas peso al genero.
    """
    df = pd.read_csv("data/games.csv")
    df = df.dropna(subset=["title", "genre"])

    # Una fila por titulo: la de mayor metacritic_score.
    df = (df.sort_values("metacritic_score", ascending=False)
            .drop_duplicates(subset=["title"])
            .reset_index(drop=True))

    # Texto combinado para la similitud. El genero se repite para que pese mas.
    def make_combined(row):
        tokens = [str(row["genre"])] * 3
        tokens += [str(row["platform_type"]), str(row["platform_maker"]),
                   str(row["publisher_tier"]), str(row["esrb_rating"])]
        if row.get("online_multiplayer") == 1: tokens.append("multiplayer")
        if row.get("is_sequel") == 1:          tokens.append("sequel")
        if row.get("vr_support") == 1:         tokens.append("vr")
        return " ".join(tokens)

    df["combined"] = df.apply(make_combined, axis=1)
    return df


@st.cache_data
def build_similarity(df):
    """Calcula la matriz de similitud entre todos los juegos.

    Combina la similitud del coseno de los vectores TF-IDF (85%) con la nota de
    Metacritic normalizada (15%), de modo que, a igual parecido, se prefieran los
    juegos mejor valorados.
    """
    # TF-IDF convierte el texto `combined` de cada juego en un vector numerico.
    # Da mas peso a las palabras distintivas y menos a las muy comunes.
    #   - stop_words="english": ignora palabras vacias en ingles.
    #   - ngram_range=(1,2): considera palabras sueltas y pares de palabras.
    #   - max_features=3000: limita el vocabulario a los 3000 terminos mas frecuentes.
    tfidf  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    matrix = tfidf.fit_transform(df["combined"])

    # Similitud del coseno: mide el angulo entre cada par de vectores. Da una matriz
    # NxN donde la posicion [i, j] indica que tan parecidos son los juegos i y j (1 = identicos).
    sim    = cosine_similarity(matrix)

    # Nota de Metacritic llevada al rango 0-1 para poder sumarla a la similitud.
    meta_norm = MinMaxScaler().fit_transform(df[["metacritic_score"]]).flatten()
    # Resultado final = 85% parecido por contenido + 15% calidad (Metacritic).
    # np.outer repite la nota de cada juego en todas las columnas para poder sumarla.
    sim_boosted = sim * 0.85 + np.outer(np.ones(len(df)), meta_norm) * 0.15
    return sim_boosted


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ
# ──────────────────────────────────────────────────────────────────────────────
def run():
    """Punto de entrada: configura los filtros, recibe el juego base y recomienda."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sistema de Recomendacion de Videojuegos</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-label">Basado en el dataset games.csv | '
        'Algoritmo: TF-IDF + Similitud del coseno con ajuste por Metacritic</div>',
        unsafe_allow_html=True,
    )

    df  = load_games()           # catalogo de juegos (una fila por titulo)
    sim = build_similarity(df)    # matriz de similitud entre todos los juegos

    # ── Filtros (barra lateral) ────────────────────────────────────────────────
    st.sidebar.markdown("### Configuracion")
    # Lista de generos disponibles para el filtro (orden alfabetico, sin repetidos).
    all_genres   = sorted(df["genre"].dropna().unique())
    genre_filter = st.sidebar.multiselect("Filtrar por genero", all_genres)

    # Rango de anios disponible segun el dataset; por defecto desde el ano 2000.
    min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider("Rango de anios", min_yr, max_yr, (2000, max_yr))

    # Nota minima de Metacritic y cantidad de recomendaciones a mostrar.
    min_meta = st.sidebar.slider("Metacritic minimo", 0, 100, 70, step=5)
    n_recs   = st.sidebar.slider("Numero de recomendaciones", 3, 12, 6)

    # ── Juego base ─────────────────────────────────────────────────────────────
    # Se aplican los filtros al catalogo para que el selector solo ofrezca juegos
    # que cumplan los criterios (ano y Metacritic).
    df_filtered = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1]) &
        (df["metacritic_score"] >= min_meta)
    ]
    # Filtro adicional por genero, solo si el usuario eligio alguno.
    if genre_filter:
        df_filtered = df_filtered[df_filtered["genre"].isin(genre_filter)]

    # Si ningun juego cumple los filtros, se avisa y se detiene aqui.
    if df_filtered.empty:
        st.warning("No hay juegos que coincidan con los filtros. Ajuste los criterios.")
        return

    # Selector del juego base, ordenado por mejor Metacritic primero.
    titles_sorted = df_filtered.sort_values("metacritic_score", ascending=False)["title"].tolist()
    selected_title = st.selectbox("Selecciona un juego que te guste:", titles_sorted, index=0)

    # Mientras no se pulse el boton, se muestra la vista inicial de estadisticas.
    if not st.button("Recomendar juegos similares", type="primary"):
        _landing(df)
        return

    # ── Calculo de recomendaciones ─────────────────────────────────────────────
    # Posicion (indice) del juego base dentro del catalogo y su fila completa.
    base_idx = df[df["title"] == selected_title].index[0]
    base_row = df.iloc[base_idx]

    # sim[base_idx] son las similitudes del juego base con todos los demas. Se
    # emparejan con su indice, se ordenan de mayor a menor y se excluye el propio base.
    scores = sorted(enumerate(sim[base_idx]), key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != base_idx]

    # Las recomendaciones tambien deben cumplir los filtros activos. Esta funcion
    # interna devuelve True solo si el juego pasa los tres criterios.
    def _passes(idx):
        row = df.iloc[idx]
        if row["metacritic_score"] < min_meta:                  return False   # Metacritic
        if not (year_range[0] <= row["year"] <= year_range[1]): return False   # rango de anios
        if genre_filter and row["genre"] not in genre_filter:   return False   # genero
        return True

    # Se conservan solo los que pasan los filtros y se toman los n primeros (mas similares).
    scores = [(i, s) for i, s in scores if _passes(i)][:n_recs]

    # ── Tarjeta del juego seleccionado (con portada) ───────────────────────────
    st.markdown("---")
    cov = cover_for(base_row["title"])      # portada real o de respaldo
    col_img, col_info = st.columns([1, 3])  # imagen estrecha a la izquierda, datos a la derecha
    with col_img:
        st.image(cov, use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div class="base-card">
            <h3>Seleccionado: {base_row['title']} ({int(base_row['year'])})</h3>
            <p>Genero: {base_row['genre']} &nbsp;|&nbsp; Plataforma: {base_row['platform']}</p>
            <p>Metacritic: <b>{int(base_row['metacritic_score'])}</b> &nbsp;|&nbsp;
               Nota de usuarios: <b>{base_row['user_score']}</b> &nbsp;|&nbsp;
               Editora: <b>{base_row['publisher']}</b></p>
            <p>Ventas globales: <b>{base_row['global_sales_million']:.2f}</b> millones &nbsp;|&nbsp;
               Clasificacion ESRB: <b>{base_row['esrb_rating']}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # Puede ocurrir que tras los filtros no quede ninguna recomendacion.
    if not scores:
        st.warning("No se encontraron juegos similares con los filtros actuales.")
        return

    # ── Metricas globales de la recomendacion ──────────────────────────────────
    # Se arma un DataFrame con los datos de los juegos recomendados para las metricas,
    # el grafico y la tabla.
    recs_df = pd.DataFrame([{
        "title":      df.iloc[i]["title"],
        "year":       int(df.iloc[i]["year"]),
        "genre":      df.iloc[i]["genre"],
        "platform":   df.iloc[i]["platform"],
        "metacritic": int(df.iloc[i]["metacritic_score"]),
        "user_score": df.iloc[i]["user_score"],
        "sales":      df.iloc[i]["global_sales_million"],
        "similarity": round(s, 4),
    } for i, s in scores])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recomendaciones",     len(recs_df))
    c2.metric("Metacritic promedio", f"{recs_df['metacritic'].mean():.0f}")
    c3.metric("Similitud promedio",  f"{recs_df['similarity'].mean():.0%}")
    c4.metric("Ventas promedio (M)", f"{recs_df['sales'].mean():.2f}")

    # ── Tarjetas de juegos recomendados (con portada) ──────────────────────────
    st.subheader("Juegos recomendados")
    cols_per_row = 3   # cuantas tarjetas por fila
    # Se parte la lista de recomendaciones en grupos de 3 para distribuirlas en filas.
    filas = [scores[i:i + cols_per_row] for i in range(0, len(scores), cols_per_row)]
    for fila in filas:
        cols = st.columns(cols_per_row)   # crea las columnas de esta fila
        for col, (idx, sim_score) in zip(cols, fila):
            row = df.iloc[idx]
            portada  = cover_for(row["title"])         # portada real o de respaldo
            fallback = placeholder_cover(row["title"]) # respaldo si la imagen no carga
            bar_w = int(sim_score * 100)               # ancho de la barrita de similitud (%)
            with col:
                # Tarjeta HTML: imagen + datos. onerror cambia a la portada de respaldo
                # si la imagen real no se puede cargar en el navegador.
                st.markdown(f"""
                <div class="rec-card">
                    <img src="{portada}" onerror="this.src='{fallback}'" alt="{row['title']}" />
                    <div class="card-body">
                        <div class="card-rank">Similitud: {sim_score:.0%}</div>
                        <div class="card-title">{row['title']}</div>
                        <div class="card-meta">{int(row['year'])} &nbsp;|&nbsp; {row['platform']}</div>
                        <div class="card-meta">{row['genre']}</div>
                        <div class="card-score">Metacritic: {int(row['metacritic_score'])}
                            &nbsp;|&nbsp; Usuarios: {row['user_score']}</div>
                        <div class="sim-bar" style="width:{bar_w}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Grafico de similitud ───────────────────────────────────────────────────
    # Barras horizontales: cada juego recomendado y su nivel de similitud con el base.
    # El color indica la nota de Metacritic.
    st.markdown("---")
    fig = px.bar(
        recs_df.sort_values("similarity"),
        x="similarity", y="title", orientation="h",
        color="metacritic", color_continuous_scale="Blues",
        title="Similitud con el juego seleccionado",
        labels={"similarity": "Similitud", "title": "Juego", "metacritic": "Metacritic"},
        template=PLOTLY_THEME,
    )
    fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla completa ─────────────────────────────────────────────────────────
    # Tabla con todas las recomendaciones y sus datos, con encabezados en espanol.
    with st.expander("Ver tabla completa de recomendaciones"):
        show = recs_df.copy()
        show.columns = ["Titulo", "Anio", "Genero", "Plataforma", "Metacritic",
                        "Nota usuarios", "Ventas (M)", "Similitud"]
        st.dataframe(show, use_container_width=True, hide_index=True)


def _landing(df):
    """Vista inicial (antes de recomendar): estadisticas generales del catalogo.

    Se muestra mientras el usuario aun no ha pulsado el boton de recomendar, para que
    la pantalla no quede vacia y se conozca el dataset.
    """
    st.markdown("---")
    st.subheader("Estadisticas del catalogo de juegos")

    # Cuatro indicadores generales del catalogo.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de juegos",    f"{len(df):,}")
    c2.metric("Anio mas antiguo",   int(df["year"].min()))
    c3.metric("Metacritic promedio", f"{df['metacritic_score'].mean():.0f}")
    c4.metric("Generos distintos",  df["genre"].nunique())

    # Los 10 juegos con mejor nota de Metacritic.
    st.subheader("Top 10 juegos mejor valorados (Metacritic)")
    top10 = df.nlargest(10, "metacritic_score")[
        ["title", "year", "genre", "platform", "metacritic_score", "user_score"]
    ].copy()
    top10.columns = ["Titulo", "Anio", "Genero", "Plataforma", "Metacritic", "Nota usuarios"]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    # Cuantos juegos hay por genero (los 15 generos mas frecuentes).
    genre_df = (df["genre"].value_counts().head(15)
                .reset_index())
    genre_df.columns = ["Genero", "Juegos"]
    fig = px.bar(genre_df, x="Genero", y="Juegos",
                 title="Distribucion de juegos por genero",
                 color="Juegos", color_continuous_scale="Blues",
                 template=PLOTLY_THEME)
    fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
