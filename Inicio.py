import streamlit as st

st.set_page_config(
    page_title="Ciencia de Datos | Lab 1 & 2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos globales MODO CLARO ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fondo blanco en toda la app */
.stApp {
    background-color: #ffffff !important;
    color: #1a1d2e !important;
}

/* Sidebar blanco */
section[data-testid="stSidebar"] {
    background-color: #f8f9fb !important;
    border-right: 1px solid #e0e3ed !important;
}
section[data-testid="stSidebar"] * {
    color: #1a1d2e !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    padding: 4px 0;
}

/* Fondo blanco en bloques y contenedores */
[data-testid="block-container"] {
    background-color: #ffffff !important;
}
.stMarkdown, .stText, .stDataFrame {
    background-color: transparent !important;
}

/* Encabezados y texto */
h1, h2, h3, h4, h5, h6 {
    color: #1a1d2e !important;
}

/* Cards de la home */
.home-card {
    background: #f8f9fb;
    border: 1px solid #e0e3ed;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.home-card h4 {
    color: #2d3a8c;
    margin: 0 0 8px 0;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.home-card p {
    color: #374151;
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Badge de tecnologias */
.badge {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #4338ca;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 4px 4px 0 0;
}

/* Metricas blancas */
div[data-testid="stMetric"] {
    background: #f8f9fb !important;
    border: 1px solid #e0e3ed !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
div[data-testid="stMetric"] label {
    color: #6b7280 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1a1d2e !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    color: #1a1d2e !important;
    border: 1px solid #d1d5db !important;
}
.stSelectbox [data-baseweb="select"] {
    background: #ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #374151 !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4338ca !important;
    border-bottom: 2px solid #4338ca !important;
}

/* DataFrames / tablas */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar con navegacion ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navegacion")
    st.markdown("---")
    page = st.radio(
        "Seleccione una opcion:",
        [
            "Inicio",
            "Analisis Exploratorio",
            "Aprendizaje Automatico",
            "Sistema de Recomendacion",
            "Carga de Archivos",
            "Analisis de Sentimientos",
            "Prompts de IA",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Ciencia de Datos - Ciclo I-2026")

# ── Enrutamiento ──────────────────────────────────────────────────────────────
if page == "Inicio":

    st.title("Plataforma de Ciencia de Datos")
    st.markdown(
        "**Laboratorio I y II** · Tecnica Electiva I · "
        "Ingenieria en Sistemas y Redes Informaticas · Ciclo I-2026"
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="home-card">
            <h4>Analisis Exploratorio</h4>
            <p>Exploracion profunda de los datasets mediante estadisticas descriptivas,
            visualizaciones interactivas y validacion de hipotesis.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="home-card">
            <h4>Carga de Archivos</h4>
            <p>Carga dinamica de archivos CSV o Excel desde el explorador
            con visualizaciones automaticas de los datos.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="home-card">
            <h4>Aprendizaje Automatico</h4>
            <p>Entrenamiento interactivo de modelos de regresion y clasificacion
            con control de parametros y metricas en tiempo real.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="home-card">
            <h4>Analisis de Sentimientos</h4>
            <p>Extraccion y analisis de opiniones con clasificacion automatica
            de sentimientos usando modelo en espanol.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="home-card">
            <h4>Sistema de Recomendacion</h4>
            <p>Motor de recomendacion de videojuegos basado en similitud
            de contenido usando TF-IDF y cosine similarity.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="home-card">
            <h4>Prompts de IA</h4>
            <p>Interfaz de lenguaje natural que responde preguntas sobre
            los datasets usando procesamiento de texto inteligente.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Datasets utilizados")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="home-card">
            <h4>Video Games Sales & Ratings</h4>
            <p>Catalogo de videojuegos con plataforma, genero, anio, publisher,
            puntuaciones de Metacritic y usuarios, ventas por region y recaudacion estimada.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="home-card">
            <h4>Retail Fraud Detection 100K</h4>
            <p>Dataset de transacciones de comercio minorista con indicadores
            de fraude, montos, categorias de productos y datos del cliente.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Tecnologias")
    techs = ["Python 3.9+", "Streamlit", "Pandas", "NumPy", "Scikit-learn",
             "Matplotlib", "Seaborn", "Plotly", "sentiment-analysis-spanish",
             "BeautifulSoup4"]
    badges = "".join(f'<span class="badge">{t}</span>' for t in techs)
    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

elif page == "Analisis Exploratorio":
    from src.analisis_exploratorio import run
    run()

elif page == "Aprendizaje Automatico":
    from src.aprendizaje_automatico import run
    run()

elif page == "Sistema de Recomendacion":
    from src.sistema_recomendacion import run
    run()

elif page == "Carga de Archivos":
    from src.carga_archivos import run
    run()

elif page == "Analisis de Sentimientos":
    from src.sentimientos import run
    run()

elif page == "Prompts de IA":
    from src.prompts_ia import run
    run()
