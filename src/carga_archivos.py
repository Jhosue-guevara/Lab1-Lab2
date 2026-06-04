"""
Analisis de Datos por Carga de Archivos.

Permite al usuario cargar un archivo (CSV, Excel o TSV) desde su computadora a
traves del navegador de archivos y explorar sus datos. Ofrece:

    - Vista previa y estadisticas del archivo cargado.
    - Analisis automatico: un motor que detecta el tipo de cada columna y elige
      y genera por si solo los graficos adecuados (sin que el usuario decida).
    - Generador manual de graficos, por si el usuario quiere construir uno especifico.

El "analisis automatico" no es una red neuronal entrenada, sino un motor de reglas
que decide el grafico correcto segun el tipo y las caracteristicas de cada columna,
igual que lo haria un analista de datos.

El punto de entrada es `run()`, invocado desde `Inicio.py`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
button[data-baseweb="tab"] { color:#475569 !important; font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:#1e40af !important; }
div[data-baseweb="tab-highlight"] { background-color:#1e40af !important; }
</style>
"""

PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"

# Color de acento y escala usados en los graficos.
ACCENT   = "#1e40af"
SECUENCE = px.colors.qualitative.Set2


def run():
    """Punto de entrada: carga el archivo y muestra preview, estadisticas y graficos."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis de Datos por Carga de Archivos</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-label">Cargue un archivo CSV, Excel o TSV desde su computadora '
        'y obtenga graficos automaticos de sus datos</div>',
        unsafe_allow_html=True,
    )

    # Navegador de archivos del sistema (boton "Browse files").
    uploaded = st.file_uploader(
        "Seleccione un archivo de datos",
        type=["csv", "xlsx", "xls", "tsv"],
        help="Formatos soportados: CSV, Excel (.xlsx/.xls) y TSV.",
    )

    # Si aun no se ha cargado nada, se muestra una indicacion y se detiene.
    if uploaded is None:
        st.info("Arrastre un archivo aqui o haga clic en 'Browse files' para cargar sus datos.")
        return

    # Lectura del archivo segun su extension.
    df = _leer_archivo(uploaded)
    if df is None:
        return

    st.success(f"Archivo cargado: **{uploaded.name}**")

    # Indicadores generales del archivo.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas",                f"{df.shape[0]:,}")
    c2.metric("Columnas",             df.shape[1])
    c3.metric("Campos numericos",     df.select_dtypes(include=np.number).shape[1])
    c4.metric("Valores nulos totales", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    # Cuatro pestanas: vista previa, estadisticas, analisis automatico y graficos manuales.
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vista previa", "Estadisticas", "Analisis automatico", "Graficos manuales"
    ])

    with tab1:
        _vista_previa(df)
    with tab2:
        _estadisticas(df)
    with tab3:
        _analisis_automatico(df)
    with tab4:
        _graficos_manuales(df)


def _leer_archivo(uploaded):
    """Lee el archivo cargado y devuelve un DataFrame, o None si ocurre un error.

    Soporta CSV/TSV (segun la extension) y Excel (con seleccion de hoja si tiene varias).
    """
    try:
        ext = uploaded.name.split(".")[-1].lower()
        if ext in ("csv", "tsv"):
            # El separador depende de la extension: tabulador para TSV, coma para CSV.
            sep = "\t" if ext == "tsv" else ","
            return pd.read_csv(uploaded, sep=sep, on_bad_lines="skip")
        # Excel: si el libro tiene varias hojas, se deja elegir cual leer.
        xls   = pd.ExcelFile(uploaded)
        sheet = (st.selectbox("Hoja de Excel", xls.sheet_names)
                 if len(xls.sheet_names) > 1 else xls.sheet_names[0])
        return pd.read_excel(uploaded, sheet_name=sheet)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DETECCION DE TIPOS DE COLUMNA (la "inteligencia" del analisis automatico)
# ──────────────────────────────────────────────────────────────────────────────
def _clasificar_columnas(df):
    """Clasifica las columnas en numericas, categoricas y de fecha.

    Reglas usadas:
      - Numericas: tipo numerico, descartando las que parecen identificadores
        (casi todos los valores distintos).
      - Fecha: columnas de tipo fecha, o de texto que se pueden interpretar como fecha.
      - Categoricas: texto con pocos valores distintos (entre 2 y 25), descartando
        identificadores y campos de texto libre.

    Devuelve tres listas: (numericas, categoricas, fechas).
    """
    n = len(df)
    numericas, categoricas, fechas = [], [], []

    for col in df.columns:
        serie = df[col]

        # ¿Es numerica? (descartando identificadores casi unicos)
        if pd.api.types.is_numeric_dtype(serie):
            if serie.nunique() < 0.95 * n:
                numericas.append(col)
            continue

        # ¿Es de tipo fecha ya reconocido por pandas?
        if pd.api.types.is_datetime64_any_dtype(serie):
            fechas.append(col)
            continue

        # ¿Es texto interpretable como fecha? Para evitar falsos positivos (como los
        # identificadores), solo se intenta si los valores parecen fechas (contienen
        # separadores - / :) o el nombre de la columna lo sugiere.
        muestra = serie.dropna().astype(str).head(50)
        nombre_fecha = any(k in col.lower()
                           for k in ["date", "time", "fecha", "timestamp", "hora"])
        parece_fecha = len(muestra) and (muestra.str.contains(r"[-/:]").mean() > 0.5
                                         or nombre_fecha)
        if parece_fecha:
            parsed = pd.to_datetime(muestra, errors="coerce")
            if parsed.notna().mean() > 0.8:
                fechas.append(col)
                continue

        # En otro caso es categorica si tiene pocos valores distintos (2 a 25).
        unicos = serie.nunique()
        if 2 <= unicos <= 25:
            categoricas.append(col)

    return numericas, categoricas, fechas


# ──────────────────────────────────────────────────────────────────────────────
# 3. ANALISIS AUTOMATICO
# ──────────────────────────────────────────────────────────────────────────────
def _analisis_automatico(df):
    """Detecta el tipo de cada columna y genera por si solo los graficos adecuados."""
    st.subheader("Analisis automatico")
    st.caption(
        "Este motor examina cada columna, identifica su tipo (numerica, categorica o "
        "de fecha) y elige el grafico mas adecuado de forma automatica, sin que usted "
        "tenga que configurarlo. Es un sistema de reglas, no una IA entrenada."
    )

    numericas, categoricas, fechas = _clasificar_columnas(df)

    # Resumen de lo que detecto el motor.
    st.markdown(
        f"**Deteccion automatica:** {len(numericas)} columna(s) numerica(s), "
        f"{len(categoricas)} categorica(s) y {len(fechas)} de fecha."
    )

    if not (numericas or categoricas):
        st.warning("No se encontraron columnas adecuadas para graficar automaticamente.")
        return

    graficos = 0   # contador de graficos generados

    # 1) Serie temporal: si hay fecha + numerica, se grafica la evolucion en el tiempo.
    if fechas and numericas:
        fcol, ycol = fechas[0], numericas[0]
        st.markdown(f"**Serie temporal** — se detecto la fecha '{fcol}'; "
                    f"se grafica la evolucion de '{ycol}' en el tiempo.")
        datos = df[[fcol, ycol]].dropna().copy()
        datos[fcol] = pd.to_datetime(datos[fcol], errors="coerce")
        datos = datos.dropna().sort_values(fcol)
        fig = px.line(datos, x=fcol, y=ycol, template=PLOTLY_THEME,
                      color_discrete_sequence=[ACCENT])
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
        graficos += 1

    # 2) Histogramas: para las primeras columnas numericas (distribucion).
    for col in numericas[:3]:
        st.markdown(f"**Histograma de '{col}'** — variable numerica; se muestra como "
                    "se distribuyen sus valores.")
        fig = px.histogram(df, x=col, nbins=40, template=PLOTLY_THEME,
                           color_discrete_sequence=[ACCENT], marginal="box")
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
        graficos += 1

    # 3) Barras: para las primeras columnas categoricas (frecuencia por categoria).
    for col in categoricas[:2]:
        st.markdown(f"**Barras de '{col}'** — variable categorica; se muestran las "
                    "categorias mas frecuentes.")
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "Frecuencia"]
        fig = px.bar(vc, x=col, y="Frecuencia", color="Frecuencia",
                     color_continuous_scale="Blues", template=PLOTLY_THEME)
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        graficos += 1

    # 4) Relaciones entre numericas: mapa de correlaciones + dispersion del par mas correlacionado.
    if len(numericas) >= 2:
        st.markdown("**Mapa de correlaciones** — se mide que tan relacionadas estan "
                    "las variables numericas entre si.")
        corr = df[numericas].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        template=PLOTLY_THEME, title="Correlaciones entre variables numericas")
        fig.update_layout(paper_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
        graficos += 1

        # Par de variables con mayor correlacion (en valor absoluto), excluyendo la
        # diagonal. Se usa una copia escribible del array para poder anular la diagonal.
        corr_abs = corr.abs().to_numpy(copy=True)
        np.fill_diagonal(corr_abs, 0)
        i, j = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
        x_col, y_col = corr.columns[i], corr.columns[j]
        st.markdown(f"**Dispersion '{x_col}' vs '{y_col}'** — es el par de variables "
                    "con mayor correlacion; se grafica su relacion.")
        muestra = df[[x_col, y_col]].dropna().sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(muestra, x=x_col, y=y_col, opacity=0.6, trendline="ols",
                         template=PLOTLY_THEME, color_discrete_sequence=[ACCENT])
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
        graficos += 1

    st.success(f"Analisis automatico completado: se generaron {graficos} grafico(s).")


# ──────────────────────────────────────────────────────────────────────────────
# 1. VISTA PREVIA
# ──────────────────────────────────────────────────────────────────────────────
def _vista_previa(df):
    """Muestra las primeras filas del archivo, con un control para cuantas ver."""
    n = st.slider("Filas a mostrar", 5, min(200, len(df)), 20)
    st.dataframe(df.head(n), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. ESTADISTICAS
# ──────────────────────────────────────────────────────────────────────────────
def _estadisticas(df):
    """Muestra los tipos de dato, nulos y las estadisticas descriptivas."""
    st.markdown("**Tipos de datos y nulos**")
    dtype_df = pd.DataFrame({
        "Columna": df.dtypes.index,
        "Tipo":    df.dtypes.values.astype(str),
        "Nulos":   df.isnull().sum().values,
        "% Nulos": (df.isnull().mean() * 100).round(2).values,
        "Unicos":  df.nunique().values,
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # describe() solo aplica a las columnas numericas.
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        st.markdown("**Estadisticas descriptivas**")
        st.dataframe(
            num_df.describe().T.reset_index().rename(columns={"index": "Campo"}),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. GRAFICOS MANUALES
# ──────────────────────────────────────────────────────────────────────────────
def _graficos_manuales(df):
    """Permite al usuario construir un grafico especifico eligiendo tipo y columnas."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()

    # Columnas disponibles segun el tipo, usadas para poblar los selectores.
    st.subheader("Generador de graficos")
    chart_type = st.selectbox("Tipo de grafico", [
        "Histograma", "Box Plot", "Scatter Plot", "Barras (categorico)",
        "Linea de tiempo", "Mapa de calor de correlaciones", "Pie Chart",
    ])

    if chart_type == "Histograma":
        # Histograma: muestra como se distribuyen los valores de una columna numerica.
        if numeric_cols:
            col  = st.selectbox("Campo numerico", numeric_cols)
            bins = st.slider("Numero de barras", 10, 100, 30)   # cantidad de intervalos
            # marginal="box" agrega un box plot encima para ver cuartiles y atipicos.
            fig  = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=[ACCENT],
                                template=PLOTLY_THEME, marginal="box")
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas numericas.")

    elif chart_type == "Box Plot":
        # Box plot: resume mediana, cuartiles y valores atipicos de una variable numerica.
        if numeric_cols:
            col   = st.selectbox("Campo numerico", numeric_cols)
            # Opcionalmente se separa la caja por categorias de otra columna.
            group = st.selectbox("Agrupar por (opcional)", ["Ninguno"] + cat_cols)
            grp   = None if group == "Ninguno" else group
            fig   = px.box(df, y=col, x=grp, color=grp, template=PLOTLY_THEME,
                           color_discrete_sequence=SECUENCE)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas numericas.")

    elif chart_type == "Scatter Plot":
        # Dispersion: relacion entre dos variables numericas (eje X y eje Y).
        if len(numeric_cols) >= 2:
            x_col     = st.selectbox("Eje X", numeric_cols, index=0)
            y_col     = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols) - 1))
            # Color opcional por una tercera columna (categorica o numerica).
            color_col = st.selectbox("Color por", ["Ninguno"] + cat_cols + numeric_cols)
            clr       = None if color_col == "Ninguno" else color_col
            # Se toma una muestra para no saturar el grafico; trendline="ols" agrega la
            # linea de tendencia, solo cuando no se colorea por una tercera variable.
            fig = px.scatter(
                df.sample(min(2000, len(df)), random_state=42),
                x=x_col, y=y_col, color=clr, opacity=0.6, template=PLOTLY_THEME,
                trendline="ols" if clr is None else None,
            )
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 2 columnas numericas.")

    elif chart_type == "Barras (categorico)":
        # Barras: frecuencia de las categorias mas comunes de una columna de texto.
        if cat_cols:
            col   = st.selectbox("Campo categorico", cat_cols)
            top_n = st.slider("Top N categorias", 5, 30, 15)
            # value_counts cuenta cuantas veces aparece cada categoria.
            vc    = df[col].value_counts().head(top_n).reset_index()
            vc.columns = [col, "Frecuencia"]
            fig = px.bar(vc, x=col, y="Frecuencia", color="Frecuencia",
                         color_continuous_scale="Blues", template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas categoricas.")

    elif chart_type == "Linea de tiempo":
        # Linea de tiempo: evolucion de un valor numerico a lo largo de un eje temporal.
        if numeric_cols:
            # Se sugieren como eje X las columnas que parezcan de fecha por su nombre.
            date_cols = [c for c in df.columns
                         if any(k in c.lower() for k in ["date", "time", "year", "month", "fecha"])]
            x_col = st.selectbox("Eje X (fecha/tiempo)",
                                 date_cols if date_cols else df.columns.tolist())
            y_col = st.selectbox("Eje Y (valor)", numeric_cols)
            # Se ordena por el eje X para que la linea siga el orden temporal correcto.
            fig   = px.line(df.sort_values(x_col), x=x_col, y=y_col, template=PLOTLY_THEME,
                            color_discrete_sequence=[ACCENT])
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Mapa de calor de correlaciones":
        # Mapa de calor: que tan relacionadas estan las variables numericas entre si.
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()   # matriz de correlaciones (-1 a 1)
            # text_auto=".2f" escribe el valor en cada celda; RdBu_r colorea de rojo
            # (correlacion negativa) a azul (positiva).
            fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Mapa de correlaciones", template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 2 columnas numericas.")

    elif chart_type == "Pie Chart":
        # Grafico de pastel: proporcion de cada categoria sobre el total.
        if cat_cols:
            col   = st.selectbox("Campo categorico", cat_cols)
            top_n = st.slider("Top N categorias", 3, 20, 8)
            vc    = df[col].value_counts().head(top_n).reset_index()
            vc.columns = [col, "Conteo"]
            fig = px.pie(vc, names=col, values="Conteo", template=PLOTLY_THEME,
                         color_discrete_sequence=SECUENCE)
            fig.update_layout(paper_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas categoricas.")
