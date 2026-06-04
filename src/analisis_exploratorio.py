import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #1a1d2e !important; }
section[data-testid="stSidebar"] { background-color: #f8f9fb !important; }
.section-title { font-size: 1.5rem; font-weight: 800; color: #2d3a8c; margin-bottom: 4px; }
.sub-label { color: #64748b; font-size: 0.88rem; margin-bottom: 20px; }
.conc-box {
    background: #eef2ff;
    border-left: 4px solid #4338ca;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 12px;
    color: #374151;
}
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

PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"


def run():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Exploratorio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Exploracion interactiva de los datasets del proyecto</div>', unsafe_allow_html=True)

    ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    df = DATASETS[ds_name]()

    submenu = st.sidebar.radio(
        "Submenu",
        [
            "Descripcion del dataset",
            "Descripcion de los campos",
            "Navegador del dataset",
            "Buscador de registros",
            "Graficador exploratorio",
            "Hipotesis",
        ],
    )

    # ── 1. Descripcion del dataset ─────────────────────────────────────────────
    if submenu == "Descripcion del dataset":
        st.subheader("Descripcion del dataset")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Campos numericos", df.select_dtypes(include=np.number).shape[1])
        c4.metric("Campos categoricos", df.select_dtypes(exclude=np.number).shape[1])

        st.markdown("---")
        st.markdown("**Tipos de datos por columna**")
        dtype_df = pd.DataFrame({
            "Columna":  df.dtypes.index,
            "Tipo":     df.dtypes.values.astype(str),
            "Nulos":    df.isnull().sum().values,
            "% Nulos":  (df.isnull().mean() * 100).round(2).values,
            "Unicos":   df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Vista previa (primeras 5 filas)**")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("**Resumen estadistico general**")
        st.dataframe(
            df.describe(include="all").T.reset_index().rename(columns={"index": "Campo"}),
            use_container_width=True
        )

    # ── 2. Descripcion de los campos ──────────────────────────────────────────
    elif submenu == "Descripcion de los campos":
        st.subheader("Descripcion de los campos")
        campo = st.selectbox("Seleccione un campo", df.columns.tolist())

        if pd.api.types.is_numeric_dtype(df[campo]):
            st.markdown("**Tipo:** Cuantitativo numerico")
            desc = df[campo].describe()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Conteo",   f"{desc['count']:.0f}")
            c2.metric("Media",    f"{desc['mean']:.4f}")
            c3.metric("Desv. Est.", f"{desc['std']:.4f}")
            c4.metric("Minimo",   f"{desc['min']:.4f}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Q1 (25%)", f"{desc['25%']:.4f}")
            c6.metric("Mediana",  f"{desc['50%']:.4f}")
            c7.metric("Q3 (75%)", f"{desc['75%']:.4f}")
            c8.metric("Maximo",   f"{desc['max']:.4f}")

            st.markdown(f"**Valores nulos:** {df[campo].isnull().sum()} ({df[campo].isnull().mean()*100:.1f}%)")
            st.markdown(f"**Valores unicos:** {df[campo].nunique()}")

            fig = px.histogram(df, x=campo, nbins=50,
                               title=f"Distribucion de {campo}",
                               color_discrete_sequence=["#4338ca"],
                               template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Tipo:** Categorico / Texto")
            st.markdown(f"**Valores nulos:** {df[campo].isnull().sum()} ({df[campo].isnull().mean()*100:.1f}%)")
            st.markdown(f"**Valores unicos:** {df[campo].nunique()}")
            vc = df[campo].value_counts().reset_index()
            vc.columns = [campo, "Frecuencia"]
            vc["% del Total"] = (vc["Frecuencia"] / len(df) * 100).round(2)
            st.markdown("**Posibles valores (top 30):**")
            st.dataframe(vc.head(30), use_container_width=True, hide_index=True)

    # ── 3. Navegador del dataset ───────────────────────────────────────────────
    elif submenu == "Navegador del dataset":
        st.subheader("Navegador del dataset completo")
        st.caption(f"Total de registros: {len(df):,}")

        with st.expander("Filtros rapidos"):
            cols_filter = st.multiselect("Columnas a mostrar", df.columns.tolist(),
                                         default=df.columns[:8].tolist())
            num_rows = st.slider("Filas a mostrar", 10, 500, 50, step=10)

        cols_to_show = cols_filter if cols_filter else df.columns.tolist()
        st.dataframe(df[cols_to_show].head(num_rows), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV completo", csv_bytes,
                           file_name=f"{ds_name}.csv", mime="text/csv")

    # ── 4. Buscador de registros (Bonus) ──────────────────────────────────────
    elif submenu == "Buscador de registros":
        st.subheader("Buscador de registros por codigo (Bonus)")

        id_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ["id", "title", "transaction", "name"])]
        if not id_cols:
            id_cols = [df.columns[0]]

        search_col = st.selectbox("Campo de busqueda", id_cols)
        search_val = st.text_input(f"Ingrese el valor a buscar en '{search_col}'")

        if search_val:
            mask   = df[search_col].astype(str).str.contains(search_val, case=False, na=False)
            result = df[mask]
            st.success(f"{len(result)} registro(s) encontrado(s).")
            if not result.empty:
                st.dataframe(result, use_container_width=True)
        else:
            st.info("Ingrese un valor para buscar registros.")

    # ── 5. Graficador exploratorio ─────────────────────────────────────────────
    elif submenu == "Graficador exploratorio":
        st.subheader("Graficador Exploratorio")
        campo = st.selectbox("Seleccione un campo para graficar", df.columns.tolist())

        if pd.api.types.is_numeric_dtype(df[campo]):
            tipo_grafico = st.radio("Tipo de grafico", ["Histograma", "Box Plot", "Violin Plot"], horizontal=True)

            if tipo_grafico == "Histograma":
                fig = px.histogram(df, x=campo, nbins=60,
                                   title=f"Histograma de {campo}",
                                   color_discrete_sequence=["#4338ca"],
                                   template=PLOTLY_THEME, marginal="box")
            elif tipo_grafico == "Box Plot":
                fig = px.box(df, y=campo, title=f"Box Plot de {campo}",
                             color_discrete_sequence=["#7c3aed"],
                             template=PLOTLY_THEME)
            else:
                fig = px.violin(df, y=campo, box=True,
                                title=f"Violin Plot de {campo}",
                                color_discrete_sequence=["#0369a1"],
                                template=PLOTLY_THEME)

            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            top_n = st.slider("Top N categorias", 5, 30, 15)
            vc = df[campo].value_counts().head(top_n).reset_index()
            vc.columns = [campo, "Frecuencia"]
            fig = px.bar(vc, x=campo, y="Frecuencia",
                         title=f"Top {top_n} valores de '{campo}'",
                         color="Frecuencia",
                         color_continuous_scale="Blues",
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
                              xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

    # ── 6. Hipotesis ──────────────────────────────────────────────────────────
    elif submenu == "Hipotesis":
        st.subheader("Validacion de Hipotesis")
        if ds_name == "IMDB Top Movies 1980-2026":
            _hipotesis_imdb(df)
        else:
            _hipotesis_fraud(df)


# ─── HIPOTESIS IMDB ────────────────────────────────────────────────────────────
def _hipotesis_imdb(df):
    hip = st.selectbox("Seleccione una hipotesis", [
        "H1: Las peliculas con mayor numero de votos tienen calificacion mas alta",
        "H2: Las peliculas producidas despues del 2000 tienen mayor recaudacion",
        "H3: La duracion de la pelicula influye en su calificacion",
    ])

    rating_col  = _find_col(df, ["rating", "imdb_rating", "score"])
    votes_col   = _find_col(df, ["votes", "numvotes", "vote"])
    year_col    = _find_col(df, ["year", "release_year", "releaseyear"])
    gross_col   = _find_col(df, ["gross", "revenue", "box_office", "worldwide"])
    runtime_col = _find_col(df, ["runtime", "duration", "minutes"])

    if hip.startswith("H1"):
        st.markdown("**H1: Las peliculas con mayor numero de votos tienen calificacion mas alta**")
        if rating_col and votes_col:
            sub = df[[rating_col, votes_col]].dropna()
            corr, pval = stats.pearsonr(sub[votes_col], sub[rating_col])
            c1, c2 = st.columns(2)
            c1.metric("Correlacion de Pearson", f"{corr:.4f}")
            c2.metric("P-value", f"{pval:.4e}")

            fig = px.scatter(
                sub.sample(min(2000, len(sub)), random_state=42),
                x=votes_col, y=rating_col, trendline="ols",
                title="Votos vs Calificacion",
                color_discrete_sequence=["#4338ca"],
                template=PLOTLY_THEME, opacity=0.5,
            )
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

            conclusion = "VALIDADA" if abs(corr) > 0.3 and pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> La hipotesis H1 queda <b>{conclusion}</b>. '
                f'La correlacion de Pearson es <b>{corr:.3f}</b> con un p-value de <b>{pval:.2e}</b>. '
                + ("Existe una relacion estadisticamente significativa entre popularidad y calificacion."
                   if conclusion == "VALIDADA"
                   else "No hay evidencia suficiente de una correlacion significativa.")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron las columnas necesarias.")

    elif hip.startswith("H2"):
        st.markdown("**H2: Las peliculas producidas despues del 2000 tienen mayor recaudacion**")
        if year_col and gross_col:
            sub = df[[year_col, gross_col]].dropna()
            sub[year_col]  = pd.to_numeric(sub[year_col],  errors="coerce")
            sub[gross_col] = pd.to_numeric(sub[gross_col], errors="coerce")
            sub = sub.dropna()
            antes   = sub.loc[sub[year_col] <= 2000, gross_col]
            despues = sub.loc[sub[year_col] >  2000, gross_col]

            c1, c2, c3 = st.columns(3)
            c1.metric("Media antes 2000",   f"${antes.mean():,.0f}")
            c2.metric("Media despues 2000", f"${despues.mean():,.0f}")
            _, pval = stats.ttest_ind(antes, despues, equal_var=False)
            c3.metric("P-value (t-test)", f"{pval:.4e}")

            sub["Periodo"] = sub[year_col].apply(lambda y: "Antes 2000" if y <= 2000 else "Despues 2000")
            fig = px.box(sub, x="Periodo", y=gross_col,
                         title="Recaudacion antes vs. despues del 2000",
                         color="Periodo",
                         color_discrete_sequence=["#4338ca", "#0369a1"],
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

            conclusion = "VALIDADA" if despues.mean() > antes.mean() and pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> La hipotesis H2 queda <b>{conclusion}</b>. '
                f'La media de recaudacion despues del 2000 es <b>${despues.mean():,.0f}</b> '
                f'versus <b>${antes.mean():,.0f}</b> antes del 2000 (p={pval:.2e}).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron las columnas necesarias.")

    else:
        st.markdown("**H3: La duracion de la pelicula influye en su calificacion**")
        if runtime_col and rating_col:
            sub = df[[runtime_col, rating_col]].dropna()
            sub[runtime_col] = pd.to_numeric(sub[runtime_col], errors="coerce")
            sub[rating_col]  = pd.to_numeric(sub[rating_col],  errors="coerce")
            sub = sub.dropna()
            corr, pval = stats.pearsonr(sub[runtime_col], sub[rating_col])
            c1, c2 = st.columns(2)
            c1.metric("Correlacion de Pearson", f"{corr:.4f}")
            c2.metric("P-value", f"{pval:.4e}")

            fig = px.scatter(
                sub.sample(min(2000, len(sub)), random_state=7),
                x=runtime_col, y=rating_col, trendline="ols",
                title="Duracion vs Calificacion",
                color_discrete_sequence=["#0369a1"],
                template=PLOTLY_THEME, opacity=0.5,
            )
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

            conclusion = "VALIDADA" if abs(corr) > 0.2 and pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> H3 queda <b>{conclusion}</b>. '
                f'La correlacion entre duracion y calificacion es <b>{corr:.3f}</b> (p={pval:.2e}).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron las columnas necesarias.")


# ─── HIPOTESIS FRAUD ───────────────────────────────────────────────────────────
def _hipotesis_fraud(df):
    hip = st.selectbox("Seleccione una hipotesis", [
        "H1: Las transacciones fraudulentas tienen montos significativamente mayores",
        "H2: Existe diferencia en la tasa de fraude entre categorias de productos",
    ])

    fraud_col  = _find_col(df, ["is_fraud", "fraud", "fraud_indicator", "fraudulent"])
    amount_col = _find_col(df, ["amount", "transaction_amount", "total", "price"])
    cat_col    = _find_col(df, ["category", "product_category", "merchant_category"])

    if hip.startswith("H1"):
        st.markdown("**H1: Las transacciones fraudulentas tienen montos significativamente mayores**")
        if fraud_col and amount_col:
            sub = df[[fraud_col, amount_col]].dropna()
            sub[fraud_col] = pd.to_numeric(sub[fraud_col], errors="coerce")
            fraudulent = sub.loc[sub[fraud_col] == 1, amount_col]
            legitimate = sub.loc[sub[fraud_col] == 0, amount_col]

            c1, c2, c3 = st.columns(3)
            c1.metric("Media Fraude",    f"${fraudulent.mean():,.2f}")
            c2.metric("Media Legitima",  f"${legitimate.mean():,.2f}")
            _, pval = stats.mannwhitneyu(fraudulent, legitimate, alternative="greater")
            c3.metric("P-value (Mann-Whitney)", f"{pval:.4e}")

            sub["Tipo"] = sub[fraud_col].map({0: "Legitima", 1: "Fraudulenta"})
            fig = px.box(sub, x="Tipo", y=amount_col,
                         title="Monto por tipo de transaccion",
                         color="Tipo",
                         color_discrete_sequence=["#4338ca", "#dc2626"],
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

            conclusion = "VALIDADA" if fraudulent.mean() > legitimate.mean() and pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> H1 queda <b>{conclusion}</b>. '
                f'El monto promedio de transacciones fraudulentas (${fraudulent.mean():,.2f}) '
                f'es mayor al de legitimas (${legitimate.mean():,.2f}) con p={pval:.2e}.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron columnas necesarias.")

    else:
        st.markdown("**H2: Existe diferencia en la tasa de fraude entre categorias de productos**")
        if fraud_col and cat_col:
            sub = df[[fraud_col, cat_col]].dropna()
            sub[fraud_col] = pd.to_numeric(sub[fraud_col], errors="coerce")
            rates = sub.groupby(cat_col)[fraud_col].mean().sort_values(ascending=False).reset_index()
            rates.columns = [cat_col, "Tasa de Fraude"]
            rates["Tasa (%)"] = (rates["Tasa de Fraude"] * 100).round(2)

            fig = px.bar(rates.head(20), x=cat_col, y="Tasa (%)",
                         title="Tasa de fraude por categoria",
                         color="Tasa (%)",
                         color_continuous_scale="Blues",
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
                              xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rates, use_container_width=True, hide_index=True)

            variance = rates["Tasa de Fraude"].var()
            conclusion = "VALIDADA" if variance > 0.0001 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> H2 queda <b>{conclusion}</b>. '
                f'Las tasas de fraude varian entre categorias (varianza={variance:.6f}), '
                f'indicando que el tipo de producto influye en la probabilidad de fraude.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron columnas necesarias.")


def _find_col(df, candidates):
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, col_orig in lower_cols.items():
            if cand in col_lower:
                return col_orig
    return None
