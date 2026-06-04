import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

LIGHT_CSS = """
<style>
.stApp { background-color: #ffffff !important; color: #1a1d2e !important; }
section[data-testid="stSidebar"] { background-color: #f8f9fb !important; }
.section-title { font-size: 1.5rem; font-weight: 800; color: #2d3a8c; margin-bottom: 4px; }
.sub-label { color: #64748b; font-size: 0.88rem; margin-bottom: 20px; }
div[data-testid="stMetric"] {
    background: #f8f9fb !important;
    border: 1px solid #e0e3ed !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
</style>
"""

PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"


def run():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Carga de Archivos y Analisis de Datos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Cargue un archivo CSV o Excel para explorar sus datos y generar visualizaciones automaticas</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Seleccione un archivo de datos",
        type=["csv", "xlsx", "xls", "tsv"],
        help="Formatos soportados: CSV, Excel (.xlsx/.xls), TSV",
    )

    if uploaded is None:
        st.info("Arrastre un archivo aqui o haga clic en 'Browse files' para cargar sus datos.")
        return

    try:
        ext = uploaded.name.split(".")[-1].lower()
        if ext in ("csv", "tsv"):
            sep = "\t" if ext == "tsv" else ","
            df = pd.read_csv(uploaded, sep=sep, on_bad_lines="skip")
        else:
            xls   = pd.ExcelFile(uploaded)
            sheet = (st.selectbox("Hoja de Excel", xls.sheet_names)
                     if len(xls.sheet_names) > 1 else xls.sheet_names[0])
            df = pd.read_excel(uploaded, sheet_name=sheet)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return

    st.success(f"Archivo cargado: **{uploaded.name}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas",               f"{df.shape[0]:,}")
    c2.metric("Columnas",             df.shape[1])
    c3.metric("Campos numericos",     df.select_dtypes(include=np.number).shape[1])
    c4.metric("Valores nulos totales",f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Vista previa", "Estadisticas", "Graficos"])

    with tab1:
        n = st.slider("Filas a mostrar", 5, min(200, len(df)), 20)
        st.dataframe(df.head(n), use_container_width=True)

    with tab2:
        st.markdown("**Tipos de datos y nulos**")
        dtype_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo":    df.dtypes.values.astype(str),
            "Nulos":   df.isnull().sum().values,
            "% Nulos": (df.isnull().mean() * 100).round(2).values,
            "Unicos":  df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            st.markdown("**Estadisticas descriptivas**")
            st.dataframe(
                num_df.describe().T.reset_index().rename(columns={"index": "Campo"}),
                use_container_width=True,
            )

    with tab3:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = df.select_dtypes(include="object").columns.tolist()

        st.subheader("Generador de graficos")
        chart_type = st.selectbox("Tipo de grafico", [
            "Histograma", "Box Plot", "Scatter Plot", "Barras (categorico)",
            "Linea de tiempo", "Mapa de calor de correlaciones", "Pie Chart",
        ])

        if chart_type == "Histograma":
            if numeric_cols:
                col  = st.selectbox("Campo numerico", numeric_cols)
                bins = st.slider("Numero de barras", 10, 100, 30)
                fig  = px.histogram(df, x=col, nbins=bins,
                                    color_discrete_sequence=["#4338ca"],
                                    template=PLOTLY_THEME, marginal="box")
                fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas numericas.")

        elif chart_type == "Box Plot":
            if numeric_cols:
                col   = st.selectbox("Campo numerico", numeric_cols)
                group = st.selectbox("Agrupar por (opcional)", ["Ninguno"] + cat_cols)
                grp   = None if group == "Ninguno" else group
                fig   = px.box(df, y=col, x=grp, color=grp,
                               template=PLOTLY_THEME,
                               color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas numericas.")

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col     = st.selectbox("Eje X", numeric_cols, index=0)
                y_col     = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1))
                color_col = st.selectbox("Color por", ["Ninguno"] + cat_cols + numeric_cols)
                clr       = None if color_col == "Ninguno" else color_col
                fig       = px.scatter(
                    df.sample(min(2000, len(df)), random_state=42),
                    x=x_col, y=y_col, color=clr, opacity=0.6,
                    template=PLOTLY_THEME,
                    trendline="ols" if clr is None else None,
                )
                fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Se necesitan al menos 2 columnas numericas.")

        elif chart_type == "Barras (categorico)":
            if cat_cols:
                col   = st.selectbox("Campo categorico", cat_cols)
                top_n = st.slider("Top N categorias", 5, 30, 15)
                vc    = df[col].value_counts().head(top_n).reset_index()
                vc.columns = [col, "Frecuencia"]
                fig = px.bar(vc, x=col, y="Frecuencia", color="Frecuencia",
                             color_continuous_scale="Blues", template=PLOTLY_THEME)
                fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
                                  xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas categoricas.")

        elif chart_type == "Linea de tiempo":
            if numeric_cols:
                date_cols = [c for c in df.columns
                             if any(k in c.lower() for k in ["date","time","year","month","fecha"])]
                x_col = st.selectbox("Eje X (fecha/tiempo)",
                                     date_cols if date_cols else df.columns.tolist())
                y_col = st.selectbox("Eje Y (valor)", numeric_cols)
                fig   = px.line(df.sort_values(x_col), x=x_col, y=y_col,
                                template=PLOTLY_THEME,
                                color_discrete_sequence=["#4338ca"])
                fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Mapa de calor de correlaciones":
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig  = px.imshow(corr, text_auto=".2f",
                                 color_continuous_scale="RdBu_r",
                                 title="Mapa de correlaciones",
                                 template=PLOTLY_THEME)
                fig.update_layout(paper_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Se necesitan al menos 2 columnas numericas.")

        elif chart_type == "Pie Chart":
            if cat_cols:
                col   = st.selectbox("Campo categorico", cat_cols)
                top_n = st.slider("Top N categorias", 3, 20, 8)
                vc    = df[col].value_counts().head(top_n).reset_index()
                vc.columns = [col, "Conteo"]
                fig = px.pie(vc, names=col, values="Conteo",
                             template=PLOTLY_THEME,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(paper_bgcolor=PLOTLY_BG)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas categoricas.")
