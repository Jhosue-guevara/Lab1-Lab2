"""
Prompts de IA (consulta de datos en lenguaje natural).

Ofrece una interfaz tipo chat donde el usuario escribe preguntas en espanol sobre
un conjunto de datos y recibe la respuesta calculada sobre el. Puede usar uno de
los datasets del proyecto o subir su propio archivo CSV/Excel.

Ejemplos de preguntas que entiende:
    - Cuantas columnas tiene el dataset?
    - Cual es la media del campo metacritic_score?
    - Cual es el mayor valor del campo year?
    - Tambien saluda ("hola") y explica que puede hacer ("ayuda").

El "motor de IA" es un interprete de reglas: identifica la intencion de la pregunta
y el campo mencionado, y ejecuta la operacion correspondiente con pandas. No es un
modelo de lenguaje entrenado, pero responde en lenguaje natural.

El punto de entrada es `run()`, invocado desde `Inicio.py`.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np

# Estilo sobrio coherente con el resto del proyecto (acento azul #1e40af).
LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #0f172a !important; }
section[data-testid="stSidebar"] { background-color: #f8fafc !important; }
.section-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; }
.section-title::after { content:""; display:block; width:42px; height:3px;
    background:#1e40af; border-radius:2px; margin-top:8px; }
.sub-label { color: #64748b; font-size: 0.9rem; margin: 10px 0 18px; }
div[data-testid="stMetric"] {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; padding: 12px 16px !important;
}
</style>
"""


@st.cache_data
def load_games():
    """Carga (cacheada) el dataset de videojuegos."""
    return pd.read_csv("data/games.csv")


@st.cache_data
def load_fraud():
    """Carga (cacheada) el dataset de deteccion de fraude."""
    return pd.read_csv("data/retail_fraud_detection_100k.csv")


# Datasets del proyecto disponibles para consultar.
DATASETS = {
    "Videojuegos (games.csv)": load_games,
    "Fraude (retail_fraud_detection)": load_fraud,
}

# Preguntas de ejemplo que se muestran como botones.
SUGGESTIONS = [
    "Cuantas filas tiene el dataset?",
    "Cuantas columnas tiene?",
    "Cuales son los nombres de las columnas?",
    "Cual es la media del campo metacritic_score?",
    "Cual es el maximo del campo year?",
    "Cual es el valor mas frecuente del campo genre?",
]


# ──────────────────────────────────────────────────────────────────────────────
# MOTOR DE RESPUESTAS
# ──────────────────────────────────────────────────────────────────────────────
def _col_en_pregunta(df, q):
    """Busca cual columna del dataset menciona la pregunta.

    Compara el texto con el nombre de cada columna (con espacios o guiones bajos) y,
    si no hay coincidencia directa, con las palabras que componen el nombre.
    Devuelve el nombre real de la columna, o None si no reconoce ninguna.
    """
    q_norm = q.lower()
    mejor, mejor_largo = None, 0
    # Coincidencia con el nombre completo de la columna (mas fiable).
    for c in df.columns:
        for variante in {c.lower(), c.lower().replace("_", " ")}:
            if variante in q_norm and len(variante) > mejor_largo:
                mejor, mejor_largo = c, len(variante)
    if mejor:
        return mejor
    # Respaldo: coincidencia por alguna palabra del nombre (ej. "amount" en "transaction_amount").
    for c in df.columns:
        palabras = [w for w in re.split(r"[_\s]+", c.lower()) if len(w) > 2]
        if any(re.search(r"\b" + re.escape(w) + r"\b", q_norm) for w in palabras):
            return c
    return None


def _fmt(v):
    """Formatea un numero: sin decimales si es entero, con 4 decimales si no."""
    try:
        return f"{int(v):,}" if float(v) == int(v) else f"{v:,.4f}"
    except (ValueError, TypeError, OverflowError):
        return str(v)


def _answer(question, df):
    """Interpreta la pregunta y devuelve la respuesta en texto (Markdown).

    Detecta primero saludos y mensajes sociales; luego intenciones de consulta de
    datos (filas, columnas, media, maximo, etc.) y, para las que se refieren a un
    campo, identifica la columna mencionada con `_col_en_pregunta`.
    """
    q = re.sub(r"[?!¿¡]", "", question.lower()).strip()

    # ── Mensajes sociales ──────────────────────────────────────────────────────
    if re.search(r"\b(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|saludos|que tal)\b", q):
        return ("Hola. Soy tu asistente de datos. Puedo responder preguntas sobre el "
                "dataset cargado. Por ejemplo: *cuantas columnas tiene?* o *cual es la "
                "media del campo metacritic_score?*. Escribe *ayuda* para ver todo lo que puedo hacer.")
    if re.search(r"\b(gracias|muchas gracias|genial|perfecto)\b", q):
        return "Con gusto. Si tienes otra pregunta sobre los datos, aqui estoy."
    if re.search(r"\b(adios|chao|hasta luego|nos vemos|bye)\b", q):
        return "Hasta luego. Vuelve cuando quieras consultar mas datos."
    # Preguntas sobre como usar la aplicacion (subir archivos / cambiar de dataset).
    if re.search(r"(subir|cargar|importar|usar)\s+(un\s+)?(csv|archivo|excel|fichero|datos)|"
                 r"puedo\s+subir|c[oó]mo\s+subo|mi\s+propio\s+archivo", q):
        return ("Si. En la parte de arriba, en **Fuente de datos**, elige "
                "**Subir mi archivo (CSV/Excel)** y selecciona tu archivo. Despues "
                "podras hacerme preguntas sobre el (cuantas columnas tiene, la media de "
                "un campo, etc.).")
    if re.search(r"(qu[eé]|cu[aá]les)\s+(datasets?|datos|fuentes?).*(hay|tienes|disponibles)|"
                 r"cambiar\s+(de\s+)?dataset|otro\s+dataset", q):
        return ("Arriba puedes elegir la **Fuente de datos**: Videojuegos, Fraude, o "
                "subir tu propio archivo CSV/Excel.")
    if re.search(r"(ayuda|que puedes hacer|que sabes hacer|opciones|help|como funciona)", q):
        return (
            "Puedo responder preguntas como:\n\n"
            "- *Cuantas filas / columnas tiene el dataset?*\n"
            "- *Cuales son los nombres de las columnas?*\n"
            "- *Cual es la media / mediana / maximo / minimo / suma del campo [nombre]?*\n"
            "- *Cual es la desviacion estandar del campo [nombre]?*\n"
            "- *Cual es el valor mas frecuente del campo [nombre]?*\n"
            "- *Cuantos valores unicos tiene el campo [nombre]?*\n"
            "- *Cuantos nulos tiene el campo [nombre]?* / *Cuantos nulos hay en total?*\n"
            "- *Cuales son los tipos de datos?* / *Muestrame el resumen estadistico*"
        )

    # ── Consultas sobre el dataset completo ────────────────────────────────────
    if re.search(r"cu[aá]ntas?\s+filas|cu[aá]ntos?\s+registros|cu[aá]ntos?\s+datos|n[uú]mero\s+de\s+filas|rows", q):
        return f"El dataset tiene **{len(df):,} filas** (registros)."

    if re.search(r"cu[aá]ntas?\s+columnas|cu[aá]ntos?\s+campos|n[uú]mero\s+de\s+columnas|columns", q):
        return (f"El dataset tiene **{df.shape[1]} columnas**:\n\n"
                + "\n".join(f"- `{c}`" for c in df.columns))

    if re.search(r"nombres?\s+de\s+(las\s+)?columnas|qu[eé]\s+columnas|cu[aá]les\s+son\s+los\s+campos|lista.*column", q):
        return "Las columnas del dataset son:\n\n" + "\n".join(f"- `{c}`" for c in df.columns)

    if re.search(r"tipos?\s+de\s+datos?|datatypes?|tipo\s+de\s+campo", q):
        return "**Tipos de datos:**\n\n" + "\n".join(f"- `{c}`: {t}" for c, t in df.dtypes.items())

    if re.search(r"describe|resumen\s+estad[ií]stico|summary|estad[ií]sticas?\s+generales?", q):
        return f"**Resumen estadistico:**\n```\n{df.describe().to_string()}\n```"

    if re.search(r"total\s+de\s+nulos?|cu[aá]ntos?\s+nulos?\s+(hay\s+)?(en\s+)?total|nulos?\s+totales?", q):
        total  = df.isnull().sum().sum()
        by_col = df.isnull().sum()
        by_col = by_col[by_col > 0]
        if by_col.empty:
            return "El dataset **no tiene valores nulos**."
        detalle = "\n".join(f"- `{c}`: {v}" for c, v in by_col.items())
        return f"El dataset tiene **{total:,} valores nulos** en total:\n\n{detalle}"

    # ── Consultas sobre un campo especifico ────────────────────────────────────
    # Para todas estas se necesita identificar de que columna habla la pregunta.
    col = _col_en_pregunta(df, q)

    # Nulos de un campo (se revisa antes que las metricas numericas).
    if re.search(r"nulos?|missing|vac[ií]os?|faltantes?", q):
        if col:
            n, pct = df[col].isnull().sum(), df[col].isnull().mean() * 100
            return f"El campo `{col}` tiene **{n:,} valores nulos** ({pct:.2f}%)."
        return "No reconoci el campo. Escribe el nombre exacto de una columna."

    # Valores unicos de un campo.
    if re.search(r"[uú]nicos?|unique|distintos?", q):
        if col:
            n = df[col].nunique()
            ejemplos = ", ".join(str(v) for v in df[col].dropna().unique()[:10])
            return f"El campo `{col}` tiene **{n} valores unicos**. Ejemplos: {ejemplos}{'...' if n > 10 else ''}."
        return "No reconoci el campo. Escribe el nombre exacto de una columna."

    # Valor mas frecuente (moda) de un campo.
    if re.search(r"m[aá]s\s+frecuente|m[aá]s\s+com[uú]n|moda|most\s+common", q):
        if col:
            moda = df[col].mode()
            if len(moda):
                veces = (df[col] == moda[0]).sum()
                return f"El valor mas frecuente del campo `{col}` es **{moda[0]}** (aparece {veces:,} veces)."
        return "No reconoci el campo. Escribe el nombre exacto de una columna."

    # Metricas numericas: cada intencion se mapea a una operacion de pandas.
    operaciones = [
        (r"media|promedio|mean|average",       "media",                lambda s: s.mean()),
        (r"mediana|median",                    "mediana",              lambda s: s.median()),
        (r"m[aá]ximo|mayor\s+valor|max",       "valor maximo",         lambda s: s.max()),
        (r"m[ií]nimo|menor\s+valor|min",       "valor minimo",         lambda s: s.min()),
        (r"desviaci[oó]n\s+est[aá]ndar|std|desv", "desviacion estandar", lambda s: s.std()),
        (r"suma|total\s+del\s+campo|sum",      "suma",                 lambda s: s.sum()),
    ]
    for patron, nombre, func in operaciones:
        if re.search(patron, q):
            if not col:
                return "No reconoci el campo. Escribe el nombre exacto de una columna."
            if pd.api.types.is_numeric_dtype(df[col]):
                # Se capitaliza el nombre de la operacion para una redaccion neutral.
                return f"**{nombre.capitalize()}** del campo `{col}`: **{_fmt(func(df[col]))}**."
            # Para maximo/minimo de texto tiene sentido el orden alfabetico.
            if nombre in ("valor maximo", "valor minimo"):
                val = df[col].dropna().astype(str).max() if nombre == "valor maximo" \
                      else df[col].dropna().astype(str).min()
                return f"**{nombre.capitalize()}** (alfabetico) del campo `{col}`: **'{val}'**."
            return f"El campo `{col}` no es numerico, no se puede calcular la {nombre}."

    # ── No se entendio la pregunta ─────────────────────────────────────────────
    return (
        "No entendi la pregunta. Escribe *ayuda* para ver ejemplos, o prueba con:\n\n"
        "- *Cuantas columnas tiene el dataset?*\n"
        "- *Cual es la media del campo [nombre]?*\n"
        "- *Cual es el maximo del campo [nombre]?*"
    )


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ (CHAT)
# ──────────────────────────────────────────────────────────────────────────────
def run():
    """Punto de entrada: elige la fuente de datos y muestra el chat de consultas."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prompts de IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Haz preguntas en lenguaje natural sobre un dataset o sobre tu propio archivo</div>',
                unsafe_allow_html=True)

    # ── Fuente de datos: un dataset del proyecto o un archivo propio ────────────
    # Se coloca arriba, en el area principal, para que sea facil de encontrar.
    fuente = st.radio("Fuente de datos",
                      list(DATASETS.keys()) + ["Subir mi archivo (CSV/Excel)"],
                      horizontal=True)

    if fuente == "Subir mi archivo (CSV/Excel)":
        archivo = st.file_uploader("Cargue su archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if archivo is None:
            st.info("Seleccione un archivo CSV o Excel para empezar a preguntar sobre el.")
            return
        try:
            df = (pd.read_csv(archivo) if archivo.name.lower().endswith("csv")
                  else pd.read_excel(archivo))
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return
        fuente_id = archivo.name
    else:
        df = DATASETS[fuente]()
        fuente_id = fuente

    # Resumen de la fuente activa y lista de columnas disponibles.
    st.caption(f"Fuente activa: **{fuente_id}** — {len(df):,} filas, {df.shape[1]} columnas.")
    with st.expander("Ver columnas disponibles para preguntar"):
        st.write(", ".join(f"`{c}`" for c in df.columns))

    # El historial se reinicia si se cambia de fuente de datos (las respuestas
    # anteriores ya no aplicarian al nuevo dataset).
    if st.session_state.get("prompts_fuente") != fuente_id:
        st.session_state.prompts_fuente = fuente_id
        st.session_state.prompts_chat = []

    # Botones con preguntas sugeridas.
    st.markdown("**Preguntas sugeridas:**")
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.prompts_chat.append(("user", sug))
                st.session_state.prompts_chat.append(("assistant", _answer(sug, df)))
                st.rerun()

    st.markdown("---")

    # Mensaje de bienvenida cuando aun no hay conversacion.
    if not st.session_state.prompts_chat:
        with st.chat_message("assistant"):
            st.markdown("Hola. Preguntame lo que quieras sobre los datos. "
                        "Escribe *ayuda* si no sabes por donde empezar.")

    # Historial de la conversacion, con burbujas de chat.
    for rol, contenido in st.session_state.prompts_chat:
        with st.chat_message(rol):
            st.markdown(contenido)

    # Caja de entrada del chat (aparece fija en la parte inferior).
    pregunta = st.chat_input("Escribe tu pregunta sobre el dataset...")
    if pregunta:
        st.session_state.prompts_chat.append(("user", pregunta))
        st.session_state.prompts_chat.append(("assistant", _answer(pregunta, df)))
        st.rerun()

    # Boton para reiniciar la conversacion.
    if st.session_state.prompts_chat and st.button("Limpiar conversacion"):
        st.session_state.prompts_chat = []
        st.rerun()
