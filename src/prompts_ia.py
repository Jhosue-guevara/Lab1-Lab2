import streamlit as st
import pandas as pd
import numpy as np
import re

LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #1a1d2e !important; }
section[data-testid="stSidebar"] { background-color: #f8f9fb !important; }
.section-title { font-size: 1.5rem; font-weight: 800; color: #2d3a8c; margin-bottom: 4px; }
.sub-label { color: #64748b; font-size: 0.88rem; margin-bottom: 20px; }
.chat-bubble-user {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 60px;
    color: #1a1d2e;
    font-size: 0.92rem;
}
.chat-bubble-ai {
    background: #f8f9fb;
    border: 1px solid #e0e3ed;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 8px 60px 8px 0;
    color: #1a1d2e;
    font-size: 0.92rem;
}
.chat-label-user { text-align: right; color: #4338ca; font-size: 0.75rem; font-weight: 700; }
.chat-label-ai   { color: #374151; font-size: 0.75rem; font-weight: 700; }
div[data-testid="stMetric"] {
    background: #f8f9fb !important;
    border: 1px solid #e0e3ed !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
</style>
"""

@st.cache_data
def load_imdb():
    return pd.read_csv("data/imdb_top_movies_1980_2026.csv")

@st.cache_data
def load_fraud():
    return pd.read_csv("data/retail_fraud_detection_100k.csv")

DATASETS = {
    "IMDB Top Movies 1980-2026": load_imdb,
    "Retail Fraud Detection 100K": load_fraud,
}


def _answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    q = re.sub(r"[?!]", "", q).strip()

    if re.search(r"cu[aá]ntas?\s+filas|cu[aá]ntos?\s+registros|cu[aá]ntos?\s+datos|tama[nñ]o|rows", q):
        return f"El dataset tiene **{len(df):,} filas** (registros)."

    if re.search(r"cu[aá]ntas?\s+columnas|cu[aá]ntos?\s+campos|columns", q):
        return f"El dataset tiene **{df.shape[1]} columnas**:\n\n" + "\n".join(f"- `{c}`" for c in df.columns)

    if re.search(r"nombres?\s+de\s+(las\s+)?columnas|qu[eé]\s+columnas|cuales\s+son\s+los\s+campos|list.*column", q):
        return "Las columnas del dataset son:\n\n" + "\n".join(f"- `{c}`" for c in df.columns)

    mean_match = re.search(r"(media|promedio|mean|average)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if mean_match:
        hint = mean_match.group(3).strip().replace(" ", "_")
        col  = _find_col(df, hint)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"La **media** del campo `{col}` es **{df[col].mean():,.4f}**."
        elif col:
            return f"El campo `{col}` es categorico, no se puede calcular la media."
        return f"No encontre un campo relacionado con '{hint}'."

    max_match = re.search(r"(m[aá]ximo|mayor\s+valor|max)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if max_match:
        hint = max_match.group(3).strip().replace(" ", "_")
        col  = _find_col(df, hint)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"El **valor maximo** del campo `{col}` es **{df[col].max():,.4f}**."
        elif col:
            return f"El **valor maximo** del campo `{col}` es **'{df[col].dropna().astype(str).max()}'**."
        return f"No encontre un campo relacionado con '{hint}'."

    min_match = re.search(r"(m[ií]nimo|menor\s+valor|min)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if min_match:
        hint = min_match.group(3).strip().replace(" ", "_")
        col  = _find_col(df, hint)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"El **valor minimo** del campo `{col}` es **{df[col].min():,.4f}**."
        elif col:
            return f"El **valor minimo** del campo `{col}` es **'{df[col].dropna().astype(str).min()}'**."
        return f"No encontre un campo relacionado con '{hint}'."

    std_match = re.search(r"(desviaci[oó]n\s+est[aá]ndar|std|desv)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if std_match:
        hint = std_match.group(3).strip().replace(" ", "_")
        col  = _find_col(df, hint)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return f"La **desviacion estandar** del campo `{col}` es **{df[col].std():,.4f}**."
        return f"No encontre el campo numerico '{hint}'."

    uniq_match = re.search(r"(valores?\s+[uú]nicos?|cu[aá]ntos?\s+[uú]nicos?|unique)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if uniq_match:
        hint    = uniq_match.group(3).strip().replace(" ", "_")
        col     = _find_col(df, hint)
        if col:
            n      = df[col].nunique()
            sample = ", ".join(str(v) for v in df[col].dropna().unique()[:10])
            return f"El campo `{col}` tiene **{n} valores unicos**. Ejemplos: {sample}{'...' if n > 10 else ''}."
        return f"No encontre un campo relacionado con '{hint}'."

    null_match = re.search(r"(nulos?|missing|vacios?|faltantes?)\s+(del?\s+campo\s+)?([a-z_\s]+)", q)
    if null_match:
        hint = null_match.group(3).strip().replace(" ", "_")
        col  = _find_col(df, hint)
        if col:
            n   = df[col].isnull().sum()
            pct = df[col].isnull().mean() * 100
            return f"El campo `{col}` tiene **{n:,} valores nulos** ({pct:.2f}%)."
        return f"No encontre el campo '{hint}'."

    if re.search(r"tipos?\s+de\s+datos?|datatypes?|tipo\s+de\s+campo", q):
        lines = [f"- `{c}`: {str(t)}" for c, t in df.dtypes.items()]
        return "**Tipos de datos:**\n\n" + "\n".join(lines)

    if re.search(r"describe|resumen\s+estad[ií]stico|summary|estad[ií]sticas?\s+generales?", q):
        return f"**Resumen estadistico:**\n```\n{df.describe().to_string()}\n```"

    if re.search(r"total\s+de\s+nulos?|cu[aá]ntos?\s+nulos?\s+totales?", q):
        total  = df.isnull().sum().sum()
        by_col = df.isnull().sum()
        by_col = by_col[by_col > 0]
        if by_col.empty:
            return "El dataset **no tiene valores nulos**."
        detail = "\n".join(f"- `{c}`: {v}" for c, v in by_col.items())
        return f"El dataset tiene **{total:,} valores nulos** en total:\n\n{detail}"

    return (
        "No entendi la pregunta. Puedes preguntarme:\n\n"
        "- *Cuantas filas tiene el dataset?*\n"
        "- *Cuantas columnas tiene?*\n"
        "- *Cual es la media del campo [nombre]?*\n"
        "- *Cual es el maximo del campo [nombre]?*\n"
        "- *Cuantos valores unicos tiene el campo [nombre]?*\n"
        "- *Cuantos nulos tiene el campo [nombre]?*\n"
        "- *Cuales son los tipos de datos?*\n"
        "- *Muéstrame el resumen estadistico*"
    )


def _find_col(df, hint):
    hint_clean = hint.strip().lower().replace(" ", "_")
    lower_map  = {c.lower(): c for c in df.columns}
    if hint_clean in lower_map:
        return lower_map[hint_clean]
    for col_lower, col_orig in lower_map.items():
        if hint_clean in col_lower or col_lower in hint_clean:
            return col_orig
    words = [w for w in re.split(r"[\s_]+", hint_clean) if len(w) > 2]
    for col_lower, col_orig in lower_map.items():
        if any(w in col_lower for w in words):
            return col_orig
    return None


SUGGESTIONS = [
    "Cuantas filas tiene el dataset?",
    "Cuantas columnas tiene?",
    "Cuales son los nombres de las columnas?",
    "Cual es la media del campo rating?",
    "Cual es el maximo del campo year?",
    "Cuantos valores unicos tiene el campo genre?",
    "Cuantos nulos tiene el campo gross?",
    "Cuales son los tipos de datos?",
    "Muestrame el resumen estadistico",
]


def run():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prompts de IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Realiza preguntas en lenguaje natural sobre los datasets</div>',
                unsafe_allow_html=True)

    ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    df = DATASETS[ds_name]()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Preguntas sugeridas:**")
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                st.session_state.chat_history.append({"role": "ai", "content": _answer(sug, df)})

    st.markdown("---")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown('<div class="chat-label-user">Tu</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-label-ai">Asistente de Datos</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-ai">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Escribe tu pregunta sobre el dataset...",
            placeholder="Ej: Cual es la media del campo rating?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Enviar", type="primary")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user",  "content": user_input})
        st.session_state.chat_history.append({"role": "ai",    "content": _answer(user_input, df)})
        st.rerun()

    if st.button("Limpiar conversacion"):
        st.session_state.chat_history = []
        st.rerun()
