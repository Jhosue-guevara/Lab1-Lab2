"""
Modulo de Analisis Exploratorio de Datos (EDA).

Construye la seccion "Analisis Exploratorio" de la aplicacion Streamlit sobre el
dataset de deteccion de fraude en comercio minorista. La interfaz se organiza en
seis pestanas:

    1. Descripcion del dataset    -> dimensiones, tipos, nulos y resumen estadistico.
    2. Descripcion de los campos  -> descripcion textual + medidas de cada campo.
    3. Navegador del dataset      -> tabla navegable con filtros y descarga CSV.
    4. Buscador de registros      -> busqueda por codigo (funcionalidad bonus).
    5. Graficador exploratorio    -> grafico adecuado segun el tipo de campo.
    6. Hipotesis                  -> dos hipotesis con su validacion y conclusion.

El punto de entrada es la funcion `run()`, invocada desde `Inicio.py`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# ESTILOS
# Hoja de estilos local en modo claro. Se inyecta una sola vez al inicio de run()
# para que esta seccion mantenga su apariencia aunque se navegue desde otra pagina.
# Paleta sobria: azul #1e40af como unico acento sobre grises neutros (slate).
# ──────────────────────────────────────────────────────────────────────────────
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Lienzo general: fondo blanco, texto slate-900 y tipografia Inter */
.stApp { background-color: #ffffff !important; color: #0f172a !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Barra lateral con un tono gris muy claro y borde sutil de separacion */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Titulo de la seccion con una pequena barra-acento debajo */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.section-title::after {
    content: "";
    display: block;
    width: 42px;
    height: 3px;
    background: #1e40af;
    border-radius: 2px;
    margin-top: 8px;
}

/* Subtitulo descriptivo bajo el titulo */
.sub-label {
    color: #64748b;
    font-size: 0.9rem;
    margin: 10px 0 22px;
}

/* Encabezados internos */
h2, h3, h4 { color: #0f172a !important; font-weight: 650; }

/* Tarjetas de metricas (st.metric): borde fino, esquinas suaves y hover discreto */
div[data-testid="stMetric"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .06);
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: .03em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Pestanas: texto gris y acento azul en la pestana activa */
button[data-baseweb="tab"] {
    color: #475569 !important;
    background: transparent !important;
    font-weight: 600;
    font-size: .92rem;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #1e40af !important; }
div[data-baseweb="tab-highlight"] { background-color: #1e40af !important; }

/* Caja de conclusion de las hipotesis: borde-acento a la izquierda */
.conc-box {
    background: #f1f5f9;
    border-left: 4px solid #1e40af;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 14px;
    color: #334155;
    font-size: .92rem;
    line-height: 1.6;
}

/* Tablas con borde fino y esquinas suaves */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Etiquetas de los controles de entrada */
.stSelectbox label, .stRadio label, .stSlider label,
.stMultiSelect label, .stTextInput label {
    color: #334155 !important;
    font-weight: 500;
}

/* Boton de descarga en el color de acento */
.stDownloadButton button {
    background: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover { background: #1d3a9e !important; }
</style>
"""

# Nombre legible del dataset, usado en titulos y en el nombre del archivo de descarga.
DATASET_NAME = "Retail Fraud Detection 100K"


@st.cache_data
def load_fraud():
    """Carga el CSV de transacciones de fraude.

    El decorador `@st.cache_data` evita releer el archivo de disco en cada
    interaccion: Streamlit reutiliza el DataFrame ya cargado en memoria.
    """
    return pd.read_csv("data/retail_fraud_detection_100k.csv")


# Descripcion textual de cada campo del dataset. Se muestra en la pestana
# "Descripcion de los campos" para que el usuario entienda el significado de cada
# columna ademas de ver sus medidas estadisticas.
FIELD_DESCRIPTIONS = {
    "transaction_id":                   "Identificador unico de la transaccion.",
    "customer_id":                      "Identificador unico del cliente.",
    "transaction_timestamp":            "Fecha y hora exacta en que se realizo la transaccion.",
    "transaction_amount":               "Monto de la transaccion.",
    "payment_method":                   "Metodo de pago utilizado.",
    "device_type":                      "Tipo de dispositivo desde el que se realizo la transaccion.",
    "location":                         "Pais o ubicacion donde se origino la transaccion.",
    "merchant_category":                "Categoria del comercio o producto.",
    "is_international":                  "Indica si la transaccion fue internacional (1) o nacional (0).",
    "transaction_frequency_24h":        "Numero de transacciones del cliente en las ultimas 24 horas.",
    "avg_transaction_amount_7d":        "Monto promedio de transacciones del cliente en los ultimos 7 dias.",
    "failed_transaction_count_24h":     "Numero de transacciones fallidas en las ultimas 24 horas.",
    "account_age_days":                 "Antiguedad de la cuenta del cliente en dias.",
    "previous_fraud_flag":              "Indica si el cliente tuvo fraudes previos (1) o no (0).",
    "unusual_amount_flag":              "Indica si el monto es inusual para el cliente (1) o no (0).",
    "unusual_location_flag":            "Indica si la ubicacion es inusual para el cliente (1) o no (0).",
    "multiple_transactions_short_time": "Indica multiples transacciones en poco tiempo (1) o no (0).",
    "high_risk_device_flag":            "Indica si el dispositivo es de alto riesgo (1) o no (0).",
    "velocity_flag":                    "Indica una velocidad anomala de transacciones (1) o no (0).",
    "fraud_flag":                       "Variable objetivo: indica si la transaccion es fraudulenta (1) o legitima (0).",
    "fraud_risk":                       "Nivel de riesgo de fraude estimado (Low, Medium, High).",
}

# Nombre en espanol de cada columna. Se usa SOLO para mostrar (selectores, tablas
# y graficos); internamente se siguen usando los nombres reales del CSV para los
# calculos. Asi la interfaz es legible sin romper el procesamiento de datos.
COLUMN_ES = {
    "transaction_id":                   "ID de transaccion",
    "customer_id":                      "ID de cliente",
    "transaction_timestamp":            "Fecha y hora",
    "transaction_amount":               "Monto de transaccion",
    "payment_method":                   "Metodo de pago",
    "device_type":                      "Tipo de dispositivo",
    "location":                         "Ubicacion",
    "merchant_category":                "Categoria del comercio",
    "is_international":                  "Es internacional",
    "transaction_frequency_24h":        "Frecuencia de transacciones (24h)",
    "avg_transaction_amount_7d":        "Monto promedio (7 dias)",
    "failed_transaction_count_24h":     "Transacciones fallidas (24h)",
    "account_age_days":                 "Antiguedad de cuenta (dias)",
    "previous_fraud_flag":              "Fraude previo",
    "unusual_amount_flag":              "Monto inusual",
    "unusual_location_flag":            "Ubicacion inusual",
    "multiple_transactions_short_time": "Multiples transacciones en poco tiempo",
    "high_risk_device_flag":            "Dispositivo de alto riesgo",
    "velocity_flag":                    "Velocidad anomala",
    "fraud_flag":                       "Fraude",
    "fraud_risk":                       "Riesgo de fraude",
}


def es(col):
    """Devuelve el nombre en espanol de una columna (o el original si no esta mapeado)."""
    return COLUMN_ES.get(col, col)


def df_es(data):
    """Devuelve una copia del DataFrame con las columnas renombradas al espanol.

    Se usa unicamente para mostrar tablas; no altera el DataFrame original.
    """
    return data.rename(columns=COLUMN_ES)


# Constantes de estilo para los graficos de Plotly: plantilla limpia y fondo
# blanco para que las figuras combinen con el resto de la interfaz.
PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"


def run():
    """Punto de entrada de la seccion. Dibuja el encabezado y las seis pestanas.

    Es la unica funcion publica del modulo; `Inicio.py` la importa y la llama
    cuando el usuario selecciona "Analisis Exploratorio".
    """
    # Inyecta los estilos y el encabezado de la seccion.
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Exploratorio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-label">Exploracion interactiva del dataset {DATASET_NAME}</div>',
        unsafe_allow_html=True,
    )

    # Carga (cacheada) del dataset que alimenta todas las pestanas.
    df = load_fraud()

    # Submenus como pestanas horizontales en la parte superior del contenido.
    tab_desc, tab_campos, tab_nav, tab_busc, tab_graf, tab_hip = st.tabs([
        "Descripcion del dataset",
        "Descripcion de los campos",
        "Navegador del dataset",
        "Buscador de registros",
        "Graficador exploratorio",
        "Hipotesis",
    ])

    # Cada pestana delega en una funcion especializada que recibe el DataFrame.
    with tab_desc:
        _descripcion_dataset(df)
    with tab_campos:
        _descripcion_campos(df)
    with tab_nav:
        _navegador(df)
    with tab_busc:
        _buscador(df)
    with tab_graf:
        _graficador(df)
    with tab_hip:
        st.subheader("Validacion de Hipotesis")
        _hipotesis_fraud(df)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DESCRIPCION DEL DATASET
# ──────────────────────────────────────────────────────────────────────────────
def _descripcion_dataset(df):
    """Muestra una vision general del dataset: dimensiones, tipos y estadisticas.

    Parametros
    ----------
    df : pandas.DataFrame
        Dataset completo de transacciones.
    """
    st.subheader("Descripcion del dataset")

    # Fila de cuatro metricas resumen: filas, columnas y conteo por tipo de dato.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")                                  # numero de registros
    c2.metric("Columnas", df.shape[1])                                      # numero de campos
    c3.metric("Campos numericos", df.select_dtypes(include=np.number).shape[1])   # solo numericos
    c4.metric("Campos categoricos", df.select_dtypes(exclude=np.number).shape[1]) # resto (texto/categoria)

    st.markdown("---")
    st.markdown("**Tipos de datos por columna**")
    # Tabla resumen por columna: nombre en espanol, tipo, nulos y unicos.
    dtype_df = pd.DataFrame({
        "Campo":    [es(c) for c in df.columns],              # nombre del campo en espanol
        "Tipo":     df.dtypes.values.astype(str),             # dtype como texto (int64, object, ...)
        "Nulos":    df.isnull().sum().values,                 # conteo de valores faltantes
        "% Nulos":  (df.isnull().mean() * 100).round(2).values,  # proporcion de nulos en porcentaje
        "Unicos":   df.nunique().values,                      # cardinalidad (valores distintos)
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Vista previa (primeras 5 filas)**")
    st.dataframe(df_es(df.head()), use_container_width=True)   # muestra con encabezados en espanol

    st.markdown("**Resumen estadistico general**")
    # describe(include="all") cubre numericos y categoricos; se transpone (.T) para
    # tener un campo por fila. Se traduce tanto el nombre del campo (indice) como
    # los encabezados de las estadisticas (count, mean, std, ...) al espanol.
    stats_es = {
        "index": "Campo",
        "count": "Conteo",      "unique": "Unicos",   "top": "Mas frecuente",
        "freq":  "Frecuencia",  "mean":   "Media",    "std": "Desv. Est.",
        "min":   "Minimo",      "25%":    "Q1 (25%)", "50%": "Mediana (50%)",
        "75%":   "Q3 (75%)",    "max":    "Maximo",
    }
    resumen = df.describe(include="all").T.reset_index().rename(columns=stats_es)
    resumen["Campo"] = resumen["Campo"].map(es)   # traduce el nombre de cada campo
    st.dataframe(resumen, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPCION DE LOS CAMPOS
# ──────────────────────────────────────────────────────────────────────────────
def _descripcion_campos(df):
    """Permite seleccionar un campo y ver su descripcion y sus medidas.

    - Campos cuantitativos: medidas de `describe` (media, desv., cuartiles, ...) e histograma.
    - Campos categoricos: lista de posibles valores con su frecuencia.
    """
    st.subheader("Descripcion de los campos")
    # Selector del campo a inspeccionar. Se muestran los nombres en espanol
    # (format_func) pero el valor devuelto sigue siendo el nombre real de la columna.
    campo = st.selectbox("Seleccione un campo", df.columns.tolist(), format_func=es)

    # Descripcion textual del campo (si existe en el diccionario), en una caja resaltada.
    descripcion = FIELD_DESCRIPTIONS.get(campo)
    if descripcion:
        st.markdown(
            f'<div class="conc-box"><b>Descripcion:</b> {descripcion}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("")  # pequeno espacio vertical

    # Rama segun el tipo de dato del campo seleccionado.
    if pd.api.types.is_numeric_dtype(df[campo]):
        # ----- Campo CUANTITATIVO -----
        st.markdown("**Tipo:** Cuantitativo numerico")
        desc = df[campo].describe()   # Serie con count, mean, std, min, cuartiles y max

        # Primera fila de medidas.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Conteo",     f"{desc['count']:.0f}")
        c2.metric("Media",      f"{desc['mean']:.4f}")
        c3.metric("Desv. Est.", f"{desc['std']:.4f}")
        c4.metric("Minimo",     f"{desc['min']:.4f}")
        # Segunda fila de medidas (cuartiles y maximo).
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Q1 (25%)", f"{desc['25%']:.4f}")
        c6.metric("Mediana",  f"{desc['50%']:.4f}")
        c7.metric("Q3 (75%)", f"{desc['75%']:.4f}")
        c8.metric("Maximo",   f"{desc['max']:.4f}")

        # Informacion complementaria sobre nulos y valores unicos.
        st.markdown(f"**Valores nulos:** {df[campo].isnull().sum()} ({df[campo].isnull().mean()*100:.1f}%)")
        st.markdown(f"**Valores unicos:** {df[campo].nunique()}")

        # Histograma para visualizar la distribucion del campo (eje en espanol).
        fig = px.histogram(df, x=campo, nbins=50,
                           title=f"Distribucion de {es(campo)}",
                           labels={campo: es(campo)},
                           color_discrete_sequence=["#1e40af"],
                           template=PLOTLY_THEME)
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # ----- Campo CATEGORICO / TEXTO -----
        st.markdown("**Tipo:** Categorico / Texto")
        st.markdown(f"**Valores nulos:** {df[campo].isnull().sum()} ({df[campo].isnull().mean()*100:.1f}%)")
        st.markdown(f"**Valores unicos:** {df[campo].nunique()}")

        # Tabla de frecuencias: cada valor posible con su conteo y porcentaje.
        vc = df[campo].value_counts().reset_index()
        vc.columns = [es(campo), "Frecuencia"]                # encabezado del campo en espanol
        vc["% del Total"] = (vc["Frecuencia"] / len(df) * 100).round(2)
        st.markdown("**Posibles valores (top 30):**")
        st.dataframe(vc.head(30), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NAVEGADOR DEL DATASET
# ──────────────────────────────────────────────────────────────────────────────
def _navegador(df):
    """Tabla navegable del dataset con seleccion de columnas, limite de filas y descarga."""
    st.subheader("Navegador del dataset completo")
    st.caption(f"Total de registros: {len(df):,}")

    # Controles de filtrado dentro de un expansor para no saturar la vista.
    with st.expander("Filtros rapidos"):
        # Columnas a mostrar (por defecto, las primeras 8). Las etiquetas salen en
        # espanol pero el valor seleccionado es el nombre real de la columna.
        cols_filter = st.multiselect("Columnas a mostrar", df.columns.tolist(),
                                     default=df.columns[:8].tolist(), format_func=es)
        # Numero de filas a mostrar.
        num_rows = st.slider("Filas a mostrar", 10, 500, 50, step=10)

    # Si el usuario no eligio columnas, se muestran todas.
    cols_to_show = cols_filter if cols_filter else df.columns.tolist()
    # Se renombran los encabezados al espanol solo para la vista.
    st.dataframe(df_es(df[cols_to_show].head(num_rows)), use_container_width=True)

    # Boton para descargar el dataset completo como CSV (codificado en UTF-8).
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV completo", csv_bytes,
                       file_name=f"{DATASET_NAME}.csv", mime="text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# 4. BUSCADOR DE REGISTROS (BONUS)
# ──────────────────────────────────────────────────────────────────────────────
def _buscador(df):
    """Busca registros por un codigo o identificador (funcionalidad bonus).

    El usuario elige un campo identificador y escribe un valor; se devuelven todas
    las filas cuyo campo contenga ese texto (busqueda insensible a mayusculas).
    """
    st.subheader("Buscador de registros por codigo (Bonus)")

    # Se proponen como campos de busqueda aquellos que parezcan identificadores.
    id_cols = [c for c in df.columns
               if any(k in c.lower() for k in ["id", "transaction", "customer"])]
    if not id_cols:                      # si no hay candidatos, usar la primera columna
        id_cols = [df.columns[0]]

    search_col = st.selectbox("Campo de busqueda", id_cols, format_func=es)
    search_val = st.text_input(f"Ingrese el valor a buscar en '{es(search_col)}'")

    if search_val:
        # Coincidencia parcial sobre el campo convertido a texto; na=False ignora nulos.
        mask   = df[search_col].astype(str).str.contains(search_val, case=False, na=False)
        result = df[mask]
        st.success(f"{len(result)} registro(s) encontrado(s).")
        if not result.empty:
            st.dataframe(df_es(result), use_container_width=True)   # encabezados en espanol
    else:
        # Mensaje guia cuando aun no se ha escrito nada.
        st.info("Ingrese un valor para buscar registros.")


# ──────────────────────────────────────────────────────────────────────────────
# 5. GRAFICADOR EXPLORATORIO
# ──────────────────────────────────────────────────────────────────────────────
def _graficador(df):
    """Genera el grafico adecuado al tipo del campo seleccionado.

    - Cuantitativo: histograma, box plot o violin plot (a eleccion del usuario).
    - Categorico: grafico de barras con las categorias mas frecuentes.
    """
    st.subheader("Graficador Exploratorio")
    campo = st.selectbox("Seleccione un campo para graficar",
                         df.columns.tolist(), format_func=es)
    etiqueta = es(campo)   # nombre del campo en espanol para titulos y ejes

    if pd.api.types.is_numeric_dtype(df[campo]):
        # ----- Campo CUANTITATIVO: el usuario elige el tipo de grafico -----
        tipo_grafico = st.radio("Tipo de grafico",
                                ["Histograma", "Box Plot", "Violin Plot"],
                                horizontal=True)

        if tipo_grafico == "Histograma":
            # Histograma con un box plot marginal para ver distribucion y outliers.
            fig = px.histogram(df, x=campo, nbins=60,
                               title=f"Histograma de {etiqueta}",
                               labels={campo: etiqueta},
                               color_discrete_sequence=["#1e40af"],
                               template=PLOTLY_THEME, marginal="box")
        elif tipo_grafico == "Box Plot":
            # Box plot: resume mediana, cuartiles y valores atipicos.
            fig = px.box(df, y=campo, title=f"Box Plot de {etiqueta}",
                         labels={campo: etiqueta},
                         color_discrete_sequence=["#3b82f6"],
                         template=PLOTLY_THEME)
        else:
            # Violin plot: combina densidad y box plot.
            fig = px.violin(df, y=campo, box=True,
                            title=f"Violin Plot de {etiqueta}",
                            labels={campo: etiqueta},
                            color_discrete_sequence=["#0369a1"],
                            template=PLOTLY_THEME)

        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # ----- Campo CATEGORICO: grafico de barras de las categorias mas comunes -----
        top_n = st.slider("Top N categorias", 5, 30, 15)
        vc = df[campo].value_counts().head(top_n).reset_index()
        vc.columns = [etiqueta, "Frecuencia"]               # encabezado en espanol
        fig = px.bar(vc, x=etiqueta, y="Frecuencia",
                     title=f"Top {top_n} valores de '{etiqueta}'",
                     color="Frecuencia",
                     color_continuous_scale="Blues",
                     template=PLOTLY_THEME)
        # Inclina las etiquetas del eje X para que no se solapen.
        fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
                          xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6. HIPOTESIS
# ──────────────────────────────────────────────────────────────────────────────
def _hipotesis_fraud(df):
    """Valida dos hipotesis sobre el fraude mediante pruebas estadisticas.

    H1: las transacciones fraudulentas tienen montos mayores (Mann-Whitney U).
    H2: la categoria de producto se asocia con el fraude (chi-cuadrado).
    En ambos casos se muestran metricas, un grafico y una conclusion automatica.
    """
    # Selector de hipotesis.
    hip = st.selectbox("Seleccione una hipotesis", [
        "H1: Las transacciones fraudulentas tienen montos significativamente mayores",
        "H2: Existe diferencia en la tasa de fraude entre categorias de productos",
    ])

    # Localiza de forma robusta las columnas relevantes (los nombres podrian variar).
    fraud_col  = _find_col(df, ["fraud_flag", "is_fraud", "fraud_indicator", "fraudulent", "fraud"])
    amount_col = _find_col(df, ["transaction_amount", "amount", "total", "price"])
    cat_col    = _find_col(df, ["merchant_category", "product_category", "category"])

    if hip.startswith("H1"):
        # ===== HIPOTESIS 1: monto vs. fraude =====
        st.markdown("**H1: Las transacciones fraudulentas tienen montos significativamente mayores**")
        if fraud_col and amount_col:
            # Subconjunto con las dos columnas necesarias, sin nulos.
            sub = df[[fraud_col, amount_col]].dropna()
            sub[fraud_col] = pd.to_numeric(sub[fraud_col], errors="coerce")
            # Separa los montos segun sean transacciones fraudulentas (1) o legitimas (0).
            fraudulent = sub.loc[sub[fraud_col] == 1, amount_col]
            legitimate = sub.loc[sub[fraud_col] == 0, amount_col]

            # Metricas: medias de cada grupo y p-value de la prueba.
            c1, c2, c3 = st.columns(3)
            c1.metric("Media Fraude",   f"${fraudulent.mean():,.2f}")
            c2.metric("Media Legitima", f"${legitimate.mean():,.2f}")
            # Mann-Whitney U (no parametrica) probando si el fraude tiende a montos mayores.
            _, pval = stats.mannwhitneyu(fraudulent, legitimate, alternative="greater")
            c3.metric("P-value (Mann-Whitney)", f"{pval:.4e}")

            # Box plot comparando la distribucion de montos por tipo de transaccion.
            sub["Tipo"] = sub[fraud_col].map({0: "Legitima", 1: "Fraudulenta"})
            fig = px.box(sub, x="Tipo", y=amount_col,
                         title="Monto por tipo de transaccion",
                         labels={amount_col: es(amount_col)},
                         color="Tipo",
                         color_discrete_sequence=["#1e40af", "#dc2626"],
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig, use_container_width=True)

            # Conclusion: validada si el fraude tiene mayor media y el p-value es significativo.
            conclusion = "VALIDADA" if fraudulent.mean() > legitimate.mean() and pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> H1 queda <b>{conclusion}</b>. '
                f'El monto promedio de transacciones fraudulentas (${fraudulent.mean():,.2f}) '
                f'es mayor al de legitimas (${legitimate.mean():,.2f}) con p={pval:.2e}. '
                + ("Existe una diferencia estadisticamente significativa en los montos."
                   if conclusion == "VALIDADA"
                   else "No hay evidencia suficiente de una diferencia significativa.")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron columnas necesarias.")

    else:
        # ===== HIPOTESIS 2: categoria vs. fraude =====
        st.markdown("**H2: Existe diferencia en la tasa de fraude entre categorias de productos**")
        if fraud_col and cat_col:
            # Subconjunto con categoria y fraude, sin nulos.
            sub = df[[fraud_col, cat_col]].dropna()
            sub[fraud_col] = pd.to_numeric(sub[fraud_col], errors="coerce")
            # Tasa de fraude (media de la bandera 0/1) por categoria, ordenada de mayor a menor.
            rates = sub.groupby(cat_col)[fraud_col].mean().sort_values(ascending=False).reset_index()
            rates.columns = [cat_col, "Tasa de Fraude"]
            rates["Tasa (%)"] = (rates["Tasa de Fraude"] * 100).round(2)

            # Grafico de barras de la tasa de fraude por categoria.
            fig = px.bar(rates.head(20), x=cat_col, y="Tasa (%)",
                         title="Tasa de fraude por categoria",
                         labels={cat_col: es(cat_col)},
                         color="Tasa (%)",
                         color_continuous_scale="Blues",
                         template=PLOTLY_THEME)
            fig.update_layout(paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
                              xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)
            # Tabla de tasas con el encabezado de categoria en espanol.
            st.dataframe(rates.rename(columns={cat_col: es(cat_col)}),
                         use_container_width=True, hide_index=True)

            # Prueba chi-cuadrado de independencia entre categoria y fraude.
            # La tabla de contingencia cruza cada categoria contra la bandera de fraude.
            contingency = pd.crosstab(sub[cat_col], sub[fraud_col])
            chi2, pval, _, _ = stats.chi2_contingency(contingency)
            c1, c2 = st.columns(2)
            c1.metric("Chi-cuadrado", f"{chi2:.4f}")
            c2.metric("P-value", f"{pval:.4e}")

            # Conclusion: validada si el p-value indica asociacion significativa.
            conclusion = "VALIDADA" if pval < 0.05 else "NO VALIDADA"
            st.markdown(
                f'<div class="conc-box"><b>Conclusion:</b> H2 queda <b>{conclusion}</b>. '
                f'La prueba chi-cuadrado de independencia arroja un estadistico de <b>{chi2:.2f}</b> '
                f'con un p-value de <b>{pval:.2e}</b>. '
                + ("Existe una asociacion estadisticamente significativa entre la categoria de producto y el fraude."
                   if conclusion == "VALIDADA"
                   else "No hay evidencia suficiente de que la categoria influya en la tasa de fraude.")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se encontraron columnas necesarias.")


def _find_col(df, candidates):
    """Devuelve el nombre real de una columna a partir de una lista de candidatos.

    Hace la busqueda insensible a mayusculas y en dos pasadas para evitar falsos
    positivos: primero exige una coincidencia exacta del nombre y, solo si no la
    hay, acepta una coincidencia parcial (la columna contiene el candidato).

    Parametros
    ----------
    df : pandas.DataFrame
    candidates : list[str]
        Nombres posibles, en orden de preferencia.

    Retorna
    -------
    str | None
        El nombre original de la columna encontrada, o None si no hay coincidencia.
    """
    # Mapa { nombre_en_minusculas : nombre_original } de todas las columnas.
    lower_cols = {c.lower(): c for c in df.columns}
    # Prioridad 1: coincidencia exacta con el nombre de la columna.
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    # Prioridad 2: coincidencia parcial (la columna contiene el candidato).
    for cand in candidates:
        for col_lower, col_orig in lower_cols.items():
            if cand in col_lower:
                return col_orig
    return None
