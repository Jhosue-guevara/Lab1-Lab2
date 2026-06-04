import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html
import re

# ─── Estilos modo claro (sin emojis) ──────────────────────────────────────────
CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #f5f7fa; color: #1a1d2e; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0e3ed;
}
section[data-testid="stSidebar"] * { color: #1a1d2e !important; }

.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #2d3a8c;
    margin-bottom: 4px;
}
.sub-label {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 20px;
}

/* Tarjeta de opinion */
.opinion-card {
    background: #ffffff;
    border: 1px solid #dde1ee;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.opinion-text {
    color: #374151;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-top: 6px;
}
.badge-pos {
    display: inline-block;
    background: #dcfce7;
    color: #16a34a;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}
.badge-neu {
    display: inline-block;
    background: #fef9c3;
    color: #a16207;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}
.badge-neg {
    display: inline-block;
    background: #fee2e2;
    color: #dc2626;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}
.conc-box {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 16px;
    color: #374151;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #dde1ee;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .metric-val { font-size: 1.8rem; font-weight: 800; color: #2d3a8c; }
.metric-card .metric-lbl { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }

/* Inputs y botones en blanco */
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    color: #1a1d2e !important;
    border: 1px solid #d1d5db !important;
}
.stSelectbox div[data-baseweb="select"] {
    background: #ffffff !important;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dde1ee;
    border-radius: 10px;
    padding: 12px;
}
</style>
"""

# ─── Carga del modelo (cache para no recargar) ────────────────────────────────
@st.cache_resource
def cargar_modelo():
    from sentiment_analysis_spanish import sentiment_analysis
    return sentiment_analysis.SentimentAnalysisSpanish()


# ─── Fuentes de scraping ──────────────────────────────────────────────────────
SOURCES = {
    "Opiniones de demo (offline)": None,
    "Reddit r/Spain (foro)": {
        "url": "https://old.reddit.com/r/Spain/",
        "selector": "p.title a.title",
    },
    "BBC Noticias Mundo": {
        "url": "https://www.bbc.com/mundo",
        "selector": "h3",
    },
    "CNN en Espanol": {
        "url": "https://cnnespanol.cnn.com/",
        "selector": "span.container__headline-text",
    },
}

DEMO_OPINIONS = [
    "Me encanta este producto, es absolutamente increible y lo recomiendo a todos.",
    "Pesimo servicio, nunca mas compro aqui. Una terrible experiencia.",
    "Esta bien, nada especial. Cumple con lo que promete pero no mas.",
    "La calidad es excelente y el precio muy justo. Muy satisfecho con la compra.",
    "Muy decepcionante. Esperaba mucho mas basandome en las opiniones. Perdida de dinero.",
    "Bastante bueno en general. Algunos detalles menores pero nada que arruine la experiencia.",
    "Soporte al cliente horrible. Nunca respondieron mis correos. Empresa a evitar.",
    "Excelente construccion y las instrucciones fueron claras y faciles de seguir.",
    "No esta mal, no esta bien. Hace lo que promete pero no te va a sorprender.",
    "Extremadamente satisfecho con esta compra. Perfecto en todos los aspectos.",
    "El producto es bueno pero el envio fue muy lento y frustrante la espera.",
    "Increible experiencia. El servicio al cliente fue profesional y amable en todo momento.",
    "Muy decepcionante. La calidad es baja y el precio demasiado elevado para lo que es.",
    "Excelente relacion calidad-precio. Lo recomiendo ampliamente a todos mis conocidos.",
    "Dificil de usar y las instrucciones son confusas. Necesita mejorar bastante.",
    "Funciona muy bien, llego rapido y el empaque estaba perfecto. Excelente vendedor.",
    "Totalmente inaceptable. El producto llego roto y no me quisieron hacer el cambio.",
    "Producto de calidad media. Ni muy bueno ni muy malo, simplemente del monton.",
]


def _scrape(source_key: str):
    if source_key == "Opiniones de demo (offline)":
        return DEMO_OPINIONS, None
    cfg = SOURCES[source_key]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(cfg["url"], headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        elements = soup.select(cfg["selector"])
        texts = [html.unescape(e.get_text(strip=True)) for e in elements if e.get_text(strip=True)]
        texts = [t for t in texts if len(t) > 20][:25]
        return texts, None
    except Exception as e:
        return [], str(e)


def _classify(score: float):
    """Clasifica el score del modelo (0=negativo, 1=positivo)."""
    if score > 0.6:
        return "Positivo", score * 100
    elif score < 0.4:
        return "Negativo", (1 - score) * 100
    else:
        return "Neutro", 50.0


def run():
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis de Sentimientos y Scraping</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Extrae opiniones y clasifica su sentimiento con un modelo de lenguaje en espanol</div>',
                unsafe_allow_html=True)

    # ── Pestanas ──────────────────────────────────────────────────────────────
    tab_texto, tab_scraping = st.tabs(["Analizar texto propio", "Scraping y analisis masivo"])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: TEXTO PROPIO
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_texto:
        st.subheader("Analisis de texto individual")
        st.write("Escribe un comentario para evaluar si su tono es positivo o negativo de forma local.")

        texto_usuario = st.text_area(
            "Introduce el texto a analizar:",
            value="Me encanta esta aplicacion! Es muy facil de usar y el diseno es fantastico.",
            height=120,
            key="texto_individual"
        )

        if st.button("Analizar Sentimiento", type="primary", key="btn_individual"):
            if not texto_usuario.strip():
                st.warning("Por favor, escribe algo para poder analizarlo.")
            else:
                with st.spinner("Cargando modelo y analizando texto..."):
                    nlp = cargar_modelo()
                    p_positivo = nlp.sentiment(texto_usuario)

                sentimiento, confianza = _classify(p_positivo)

                st.markdown("---")
                st.subheader("Resultado del Analisis")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sentimiento", sentimiento)
                with c2:
                    st.metric("Confianza", f"{confianza:.1f}%")
                with c3:
                    st.metric("Puntaje del modelo", f"{p_positivo:.4f}")

                # Barra visual de sentimiento
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_positivo,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Escala de sentimiento (0 = Negativo | 1 = Positivo)", "font": {"color": "#374151"}},
                    gauge={
                        "axis": {"range": [0, 1], "tickcolor": "#374151"},
                        "bar": {"color": "#4f46e5"},
                        "steps": [
                            {"range": [0, 0.4], "color": "#fee2e2"},
                            {"range": [0.4, 0.6], "color": "#fef9c3"},
                            {"range": [0.6, 1], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "#1a1d2e", "width": 3},
                            "thickness": 0.75,
                            "value": p_positivo,
                        },
                    },
                    number={"font": {"color": "#2d3a8c"}},
                ))
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font={"color": "#374151"},
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Como funciona este puntaje"):
                    st.write(f"El modelo asigna una probabilidad matematica de **{p_positivo:.4f}** al texto analizado.")
                    st.write("- Un valor cercano a **1.0** significa que el modelo considera el comentario **positivo**.")
                    st.write("- Un valor cercano a **0.0** significa que el modelo lo considera **negativo**.")
                    st.write("- Un valor entre **0.4 y 0.6** se clasifica como **neutro o mixto**.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: SCRAPING MASIVO
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_scraping:
        st.subheader("Extraccion y analisis masivo de opiniones")

        col_src, col_btn = st.columns([3, 1])
        with col_src:
            source = st.selectbox("Fuente de opiniones", list(SOURCES.keys()))
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Extraer y Analizar", type="primary", key="btn_scraping")

        if run_btn:
            with st.spinner("Extrayendo opiniones..."):
                texts, error = _scrape(source)

            if error:
                st.warning(f"No se pudo conectar al sitio: {error}. Se usaran las opiniones de demo.")
                texts = DEMO_OPINIONS

            if not texts:
                st.error("No se encontraron opiniones. Intente con otra fuente.")
                return

            with st.spinner(f"Analizando {len(texts)} opiniones con el modelo en espanol..."):
                nlp = cargar_modelo()
                results = []
                for t in texts:
                    score = nlp.sentiment(t)
                    sentimiento, confianza = _classify(score)
                    results.append({
                        "Opinion": t,
                        "Sentimiento": sentimiento,
                        "Puntaje": round(score, 4),
                        "Confianza (%)": round(confianza, 1),
                    })

            df = pd.DataFrame(results)

            st.success(f"Se analizaron **{len(df)}** opiniones de '{source}'.")
            st.markdown("---")

            # Metricas globales
            counts = df["Sentimiento"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de opiniones", len(df))
            c2.metric("Positivas", counts.get("Positivo", 0))
            c3.metric("Neutras",   counts.get("Neutro",   0))
            c4.metric("Negativas", counts.get("Negativo", 0))

            # Graficas
            color_map = {
                "Positivo": "#16a34a",
                "Neutro":   "#a16207",
                "Negativo": "#dc2626",
            }
            col_a, col_b = st.columns(2)

            with col_a:
                fig_pie = px.pie(
                    df, names="Sentimiento",
                    title="Distribucion de sentimientos",
                    color="Sentimiento",
                    color_discrete_map=color_map,
                    template="simple_white",
                )
                fig_pie.update_layout(
                    paper_bgcolor="#ffffff",
                    font={"color": "#374151"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                fig_hist = px.histogram(
                    df, x="Puntaje", nbins=20,
                    color="Sentimiento",
                    title="Distribucion de puntajes del modelo",
                    color_discrete_map=color_map,
                    template="simple_white",
                )
                fig_hist.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f9fafb",
                    font={"color": "#374151"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Tabla de resultados
            st.markdown("---")
            st.subheader("Opiniones analizadas")

            for _, row in df.iterrows():
                s = row["Sentimiento"]
                if s == "Positivo":
                    badge = f'<span class="badge-pos">Positivo</span>'
                elif s == "Negativo":
                    badge = f'<span class="badge-neg">Negativo</span>'
                else:
                    badge = f'<span class="badge-neu">Neutro</span>'

                st.markdown(f"""
                <div class="opinion-card">
                    {badge}
                    <small style="color:#6b7280; margin-left:8px;">
                        Puntaje: {row['Puntaje']} | Confianza: {row['Confianza (%)']:.1f}%
                    </small>
                    <p class="opinion-text">{html.escape(row['Opinion'])}</p>
                </div>
                """, unsafe_allow_html=True)

            # Conclusion
            dominant = df["Sentimiento"].mode()[0]
            pct = counts.get(dominant, 0) / len(df) * 100
            avg_score = df["Puntaje"].mean()
            tono = "favorable" if avg_score > 0.6 else "desfavorable" if avg_score < 0.4 else "neutral"

            st.markdown(f"""
            <div class="conc-box">
                <strong>Conclusion del analisis:</strong><br>
                De las <strong>{len(df)}</strong> opiniones analizadas de <em>{source}</em>,
                el sentimiento predominante es <strong>{dominant}</strong>
                ({pct:.1f}% de las opiniones).
                El puntaje promedio del modelo es <strong>{avg_score:.4f}</strong>,
                lo que indica un tono general <strong>{tono}</strong>.
            </div>
            """, unsafe_allow_html=True)

            # Descarga de resultados
            st.markdown("---")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar resultados en CSV",
                csv_bytes,
                file_name="analisis_sentimientos.csv",
                mime="text/csv",
            )
