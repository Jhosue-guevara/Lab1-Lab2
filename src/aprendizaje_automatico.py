"""
Modulo de Aprendizaje Automatico.

Interfaz para entrenar y evaluar modelos de machine learning sobre el dataset de
deteccion de fraude. Permite al usuario:

    - Elegir el tipo de tarea (Regresion o Clasificacion).
    - Seleccionar el algoritmo (al menos dos por tarea).
    - Seleccionar la variable a analizar / objetivo (Y).
    - Seleccionar una o varias variables independientes (X).
    - Definir el porcentaje de datos de entrenamiento y de prueba.
    - Ver una grafica con los datos de entrenamiento y las predicciones de prueba.
    - Ver las metricas del modelo y los p-values de significancia de las variables.

El punto de entrada es `run()`, invocado desde `Inicio.py`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import statsmodels.api as sm
import warnings

# Reutiliza el mapa de nombres en espanol y las constantes de estilo del modulo EDA
# para mantener una unica fuente de verdad en las traducciones y la apariencia.
from src.analisis_exploratorio import COLUMN_ES, es, LIGHT_CSS, PLOTLY_THEME, PLOTLY_BG

warnings.filterwarnings("ignore")


@st.cache_data
def load_fraud():
    """Carga (cacheada) el CSV de transacciones de fraude."""
    return pd.read_csv("data/retail_fraud_detection_100k.csv")


@st.cache_data
def encode_dataset():
    """Devuelve el dataset con las columnas de texto codificadas a numeros.

    Los algoritmos de scikit-learn solo operan con valores numericos, por lo que
    cada columna categorica se transforma con LabelEncoder. El resultado se cachea
    para no repetir la codificacion en cada interaccion.
    """
    df = load_fraud().copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


# Catalogo de algoritmos disponibles para cada tipo de tarea (al menos dos por tarea).
ALGOS_REG = ["Regresion Lineal", "Arbol de Decision (Reg.)", "Random Forest (Reg.)"]
ALGOS_CLS = ["Regresion Logistica", "Arbol de Decision (Cls.)", "Random Forest (Cls.)"]

# Columnas que son identificadores o marcas de tiempo: no aportan informacion
# predictiva (son etiquetas unicas), por lo que se excluyen de los selectores de
# variables para evitar resultados sin sentido (como R2 = 0).
EXCLUDE_COLS = {"transaction_id", "customer_id", "transaction_timestamp"}

# Limite de filas para ajustar los modelos de statsmodels (calculo de p-values).
# Con muchas filas el resultado no cambia de forma relevante y se gana velocidad.
PVALUE_SAMPLE = 20000


def run():
    """Punto de entrada: dibuja el panel de configuracion y, al entrenar, los resultados."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aprendizaje Automatico</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-label">Entrena y evalua modelos de regresion y clasificacion '
        'sobre el dataset Retail Fraud Detection</div>',
        unsafe_allow_html=True,
    )

    df = encode_dataset()
    # Columnas numericas utilizables, descartando identificadores y marcas de tiempo.
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in EXCLUDE_COLS]

    # Guia breve para que el usuario entienda el flujo de la seccion.
    st.info(
        "Como usar esta seccion: 1) elija si quiere predecir un numero (Regresion) "
        "o una categoria (Clasificacion); 2) elija el algoritmo; 3) elija que predecir "
        "(Y) y con que variables predecirlo (X); 4) defina el porcentaje de "
        "entrenamiento y pulse Entrenar. Consejo: elija como X variables que de "
        "verdad puedan influir en Y."
    )

    # ── Panel de configuracion (arriba, en el area principal) ──────────────────
    st.subheader("Configuracion del modelo")

    # Fila 1: tipo de tarea y algoritmo. El tipo de tarea va fuera de cualquier
    # formulario para que la lista de algoritmos se actualice de inmediato.
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.radio(
            "Tipo de tarea", ["Regresion", "Clasificacion"], horizontal=True,
            help="Regresion predice un valor numerico (por ejemplo, un monto). "
                 "Clasificacion predice una categoria (por ejemplo, si una "
                 "transaccion es fraude o no).",
        )
    with col2:
        algos     = ALGOS_REG if task_type == "Regresion" else ALGOS_CLS
        algorithm = st.selectbox(
            "Algoritmo", algos,
            help="Metodo con el que el modelo aprende la relacion entre las "
                 "variables. Puede entrenar con varios y comparar sus resultados.",
        )

    # Variables objetivo sugeridas segun la tarea:
    #  - Clasificacion: columnas de baja cardinalidad (<= 10 valores distintos).
    #  - Regresion: columnas con muchos valores distintos (continuas).
    if task_type == "Clasificacion":
        target_cols = [c for c in numeric_cols if df[c].nunique() <= 10]
    else:
        target_cols = [c for c in numeric_cols if df[c].nunique() > 15]
    if not target_cols:                 # respaldo por si ninguna cumpliera el criterio
        target_cols = numeric_cols

    # Fila 2: variable a analizar (Y) y variables independientes (X).
    col3, col4 = st.columns(2)
    with col3:
        var_dep = st.selectbox(
            "Variable a analizar (Y)", target_cols, format_func=es,
            help="Es lo que el modelo intentara predecir (la variable objetivo).",
        )
    with col4:
        avail_x = [c for c in numeric_cols if c != var_dep]
        # Multiselect: el usuario puede elegir una o varias variables independientes.
        var_indep = st.multiselect(
            "Variables independientes (X)", avail_x,
            default=avail_x[:2], format_func=es,
            help="Son los datos que el modelo usa para predecir Y. Elija variables "
                 "que puedan influir en Y; puede seleccionar varias.",
        )

    # Fila 3: porcentaje de entrenamiento (el de prueba es el complemento).
    train_pct = st.slider(
        "Porcentaje de datos de entrenamiento (%)", 50, 90, 80, step=5,
        help="Parte de los datos que se usa para entrenar el modelo. El resto se "
             "reserva para probar que tan bien predice con datos que no ha visto.",
    )
    test_pct  = 100 - train_pct

    # Si hay mas de una X, se elige cual usar como eje del grafico exploratorio.
    eje_x = var_indep[0] if var_indep else None
    if len(var_indep) > 1:
        eje_x = st.selectbox(
            "Variable para el eje X del grafico", var_indep, format_func=es,
            help="Variable que se mostrara en el eje horizontal del grafico de resultados.",
        )

    # Resumen legible de la configuracion elegida.
    x_legible = ", ".join(es(c) for c in var_indep) if var_indep else "(ninguna)"
    st.markdown(
        f"**Configuracion:** {algorithm} · Y = {es(var_dep)} · X = {x_legible} · "
        f"Entrenamiento {train_pct}% / Prueba {test_pct}%"
    )

    # ── Entrenamiento ───────────────────────────────────────────────────────────
    if st.button("Entrenar Modelo", type="primary"):
        if not var_indep:
            st.warning("Seleccione al menos una variable independiente (X).")
            return
        _train_and_display(df, var_dep, var_indep, eje_x, algorithm,
                           train_pct / 100, task_type)


def _build_model(algorithm):
    """Devuelve la instancia del modelo de scikit-learn segun el algoritmo elegido."""
    if algorithm == "Regresion Lineal":
        return LinearRegression()
    if algorithm == "Arbol de Decision (Reg.)":
        return DecisionTreeRegressor(max_depth=6, random_state=42)
    if algorithm == "Random Forest (Reg.)":
        return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    if algorithm == "Regresion Logistica":
        return LogisticRegression(max_iter=1000, random_state=42)
    if algorithm == "Arbol de Decision (Cls.)":
        return DecisionTreeClassifier(max_depth=6, random_state=42)
    return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)


def _train_and_display(df, y_col, x_cols, eje_x, algorithm, train_ratio, task_type):
    """Entrena el modelo y muestra metricas, grafica y p-values.

    Parametros
    ----------
    df : DataFrame ya codificado a numerico.
    y_col : str            -> columna objetivo.
    x_cols : list[str]     -> columnas independientes.
    eje_x : str            -> columna usada como eje X del grafico.
    algorithm : str        -> algoritmo seleccionado.
    train_ratio : float    -> proporcion de entrenamiento (0-1).
    task_type : str        -> "Regresion" o "Clasificacion".
    """
    # Subconjunto con las columnas necesarias y sin filas nulas.
    sub  = df[x_cols + [y_col]].dropna()
    X_df = sub[x_cols]
    y    = sub[y_col].values

    # Particion en entrenamiento y prueba segun el porcentaje elegido.
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, train_size=train_ratio, random_state=42
    )

    es_clasificacion = task_type == "Clasificacion"

    # En clasificacion se estandarizan las variables (ayuda a la convergencia de
    # la regresion logistica). En regresion se usan los valores tal cual.
    if es_clasificacion:
        scaler = StandardScaler().fit(X_train_df)
        X_train = scaler.transform(X_train_df)
        X_test  = scaler.transform(X_test_df)
    else:
        X_train = X_train_df.values
        X_test  = X_test_df.values

    # Construccion y ajuste del modelo.
    model = _build_model(algorithm)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    st.markdown("---")
    st.subheader("Metricas del modelo")

    # Se guardan los valores clave para construir despues una conclusion adaptativa.
    if es_clasificacion:
        acc, baseline = _metricas_clasificacion(y_train, y_test, y_pred_train, y_pred_test)
    else:
        r2_test = _metricas_regresion(model, y_train, y_test, y_pred_train, y_pred_test)

    # ── Grafica: datos de entrenamiento (reales) + predicciones de prueba ───────
    st.subheader("Efecto del modelo")
    st.caption(
        "Cada punto es una transaccion. En azul los datos de entrenamiento (reales), "
        "en gris los de prueba (reales) y en rojo (x) las predicciones del modelo "
        "sobre los datos de prueba. Cuanto mas cerca esten las x rojas de los puntos "
        "grises, mejor predice el modelo."
    )
    _grafica_efecto(model, X_train_df, X_test_df, y_train, y_test, y_pred_test,
                    eje_x, y_col, x_cols, algorithm, es_clasificacion)

    # En clasificacion se anade la matriz de confusion si hay pocas clases.
    if es_clasificacion and len(np.unique(y_test)) <= 10:
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm = px.imshow(cm, text_auto=True, title="Matriz de Confusion",
                           color_continuous_scale="Blues", template=PLOTLY_THEME)
        fig_cm.update_layout(paper_bgcolor=PLOTLY_BG)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Significancia de las variables (p-values) ───────────────────────────────
    st.subheader("Significancia de las variables (p-values)")
    st.caption(
        "El p-value indica si cada variable X realmente influye en Y. Si es menor a "
        "0.05, su efecto se considera estadisticamente significativo; si es mayor, no "
        "hay evidencia de que esa variable ayude a predecir Y."
    )
    _mostrar_pvalues(X_train_df, y_train, x_cols, es_clasificacion)

    # ── Conclusion adaptativa: cambia segun que tan bien predijo el modelo ──────
    st.subheader("Conclusion")
    if es_clasificacion:
        _conclusion_clasificacion(acc, baseline)
    else:
        _conclusion_regresion(r2_test)


def _metricas_regresion(model, y_train, y_test, y_pred_train, y_pred_test):
    """Calcula y muestra las metricas de un modelo de regresion."""
    mse   = mean_squared_error(y_test, y_pred_test)        # error cuadratico medio
    rmse  = np.sqrt(mse)                                    # raiz del MSE (mismas unidades que Y)
    mae   = mean_absolute_error(y_test, y_pred_test)        # error absoluto medio
    r2    = r2_score(y_test, y_pred_test)                   # bondad de ajuste en prueba
    r2_tr = r2_score(y_train, y_pred_train)                 # bondad de ajuste en entrenamiento

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R2 (Prueba)", f"{r2:.4f}",
              help="Que tan bien predice el modelo con datos nuevos. Va de 0 a 1: "
                   "1 es perfecto y un valor cercano a 0 o negativo significa que casi "
                   "no acierta.")
    c2.metric("R2 (Entrenamiento)", f"{r2_tr:.4f}",
              help="Lo mismo que el R2 de prueba pero con los datos que el modelo ya "
                   "vio al entrenar. Si es mucho mayor que el de prueba, el modelo se "
                   "esta sobreajustando (memoriza en vez de aprender).")
    c3.metric("RMSE", f"{rmse:.4f}",
              help="Error tipico del modelo, en las mismas unidades que Y. Mientras "
                   "mas bajo, mejor. Penaliza mas los errores grandes.")
    c4.metric("MAE", f"{mae:.4f}",
              help="Error promedio del modelo, en las unidades de Y. Es el promedio "
                   "de cuanto se equivoca en cada prediccion. Mientras mas bajo, mejor.")
    c5.metric("MSE", f"{mse:.4f}",
              help="Error cuadratico medio: el promedio de los errores al cuadrado. "
                   "Mientras mas bajo, mejor. Es la base del RMSE.")

    # Explicacion en lenguaje sencillo de cada metrica.
    st.caption(
        "R2 mide que tan bien el modelo explica los datos: 1 es perfecto, 0 significa "
        "que no explica nada (un R2 cercano a 0 o negativo indica que las variables X "
        "elegidas no sirven para predecir Y). RMSE, MAE y MSE son medidas del error: "
        "mientras mas bajas, mejor (estan en las unidades de Y)."
    )

    # Para la regresion lineal se muestran tambien los coeficientes e intercepto.
    if hasattr(model, "coef_") and np.ndim(model.coef_) == 1:
        intval = model.intercept_ if np.isscalar(model.intercept_) else float(model.intercept_)
        st.caption(f"Intercepto: {intval:.4f}")

    return r2   # R2 de prueba, usado para la conclusion adaptativa


def _metricas_clasificacion(y_train, y_test, y_pred_train, y_pred_test):
    """Calcula y muestra las metricas de un modelo de clasificacion."""
    acc    = accuracy_score(y_test, y_pred_test)            # exactitud en prueba
    acc_tr = accuracy_score(y_train, y_pred_train)          # exactitud en entrenamiento
    prec   = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    rec    = recall_score(y_test,  y_pred_test, average="weighted", zero_division=0)
    f1     = f1_score(y_test,      y_pred_test, average="weighted", zero_division=0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exactitud (Prueba)", f"{acc:.4f}",
              help="Porcentaje de aciertos del modelo con datos nuevos. Va de 0 a 1: "
                   "0.85 significa que acierta el 85% de las veces.")
    c2.metric("Exactitud (Entrenamiento)", f"{acc_tr:.4f}",
              help="Porcentaje de aciertos sobre los datos con los que se entreno. Si "
                   "es mucho mayor que la de prueba, el modelo se esta sobreajustando.")
    c3.metric("Precision", f"{prec:.4f}",
              help="De todo lo que el modelo marco como positivo (por ejemplo, fraude), "
                   "que porcentaje realmente lo era. Mientras mas alta, menos falsas "
                   "alarmas.")
    c4.metric("Recall", f"{rec:.4f}",
              help="De todos los positivos reales (los fraudes que de verdad ocurrieron), "
                   "cuantos logro detectar el modelo. Mientras mas alto, menos casos se "
                   "le escapan.")
    c5.metric("F1-Score", f"{f1:.4f}",
              help="Equilibrio entre Precision y Recall en un solo numero. Util cuando "
                   "interesa tanto no dar falsas alarmas como no dejar pasar casos.")

    # Explicacion en lenguaje sencillo de cada metrica (todas van de 0 a 1).
    st.caption(
        "Exactitud: porcentaje de predicciones correctas. Precision: de lo que el "
        "modelo marco como positivo, cuanto acerto. Recall: de los positivos reales, "
        "cuantos detecto. F1-Score: equilibrio entre precision y recall. En todas, "
        "mientras mas cerca de 1, mejor."
    )

    # baseline = proporcion de la clase mas frecuente; es el acierto que se lograria
    # "adivinando" siempre la clase mayoritaria. Sirve para juzgar si el modelo aporta.
    valores, conteos = np.unique(y_test, return_counts=True)
    baseline = conteos.max() / conteos.sum()
    return acc, baseline


def _grafica_efecto(model, X_train_df, X_test_df, y_train, y_test, y_pred_test,
                    eje_x, y_col, x_cols, algorithm, es_clasificacion):
    """Dibuja, sobre el eje X elegido, los datos de entrenamiento y las predicciones de prueba.

    Siempre incluye:
      - puntos de entrenamiento (valores reales),
      - puntos de prueba (valores reales),
      - puntos de prueba (predicciones del modelo).
    Si la regresion usa una sola variable, anade la linea de prediccion continua.
    """
    # Muestras para no saturar el grafico cuando hay muchos puntos.
    n_tr = min(500, len(X_train_df))
    n_te = min(300, len(X_test_df))
    idx_tr = np.random.RandomState(42).choice(len(X_train_df), n_tr, replace=False)
    idx_te = np.random.RandomState(7).choice(len(X_test_df),  n_te, replace=False)

    x_tr = X_train_df[eje_x].values
    x_te = X_test_df[eje_x].values
    etiqueta_x = es(eje_x)
    etiqueta_y = es(y_col)

    fig = go.Figure()
    # Datos de entrenamiento (reales).
    fig.add_trace(go.Scatter(
        x=x_tr[idx_tr], y=y_train[idx_tr], mode="markers",
        name="Entrenamiento (real)",
        marker=dict(color="#1e40af", opacity=0.45, size=6),
    ))
    # Datos de prueba (reales).
    fig.add_trace(go.Scatter(
        x=x_te[idx_te], y=y_test[idx_te], mode="markers",
        name="Prueba (real)",
        marker=dict(color="#64748b", opacity=0.5, size=6),
    ))
    # Datos de prueba (predicciones del modelo).
    fig.add_trace(go.Scatter(
        x=x_te[idx_te], y=y_pred_test[idx_te], mode="markers",
        name="Prueba (prediccion)",
        marker=dict(color="#dc2626", opacity=0.6, size=7, symbol="x"),
    ))

    # Linea de prediccion continua solo si hay exactamente una variable independiente.
    if not es_clasificacion and len(x_cols) == 1:
        xr = np.linspace(x_tr.min(), x_tr.max(), 300).reshape(-1, 1)
        fig.add_trace(go.Scatter(
            x=xr[:, 0], y=model.predict(xr), mode="lines",
            name="Linea de prediccion",
            line=dict(color="#0369a1", width=2),
        ))

    fig.update_layout(
        title=f"{algorithm}: {etiqueta_x} vs {etiqueta_y}",
        xaxis_title=etiqueta_x, yaxis_title=etiqueta_y,
        template=PLOTLY_THEME, paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


def _mostrar_pvalues(X_train_df, y_train, x_cols, es_clasificacion):
    """Calcula los p-values de cada variable con statsmodels y los muestra en tabla.

    - Regresion: modelo OLS (minimos cuadrados ordinarios).
    - Clasificacion: modelo Logit (regresion logistica), con variables estandarizadas.
    El p-value indica si el efecto de cada variable es estadisticamente significativo
    (convencionalmente, significativo cuando p < 0.05).
    """
    # Se trabaja sobre una muestra si el conjunto es muy grande (mas rapido).
    if len(X_train_df) > PVALUE_SAMPLE:
        muestra = X_train_df.sample(PVALUE_SAMPLE, random_state=42)
        Xp = muestra
        yp = pd.Series(y_train, index=X_train_df.index).loc[muestra.index].values
    else:
        Xp = X_train_df
        yp = y_train

    try:
        if es_clasificacion:
            # Estandariza para que el Logit converja de forma estable.
            Xs = StandardScaler().fit_transform(Xp)
            Xs = pd.DataFrame(Xs, columns=Xp.columns)
            modelo = sm.Logit(yp, sm.add_constant(Xs)).fit(disp=0)
        else:
            modelo = sm.OLS(yp, sm.add_constant(Xp)).fit()

        # Tabla con el coeficiente y el p-value de cada variable (se omite la constante).
        tabla = pd.DataFrame({
            "Variable":    [es(c) for c in Xp.columns],
            "Coeficiente": modelo.params.drop("const").values,
            "P-value":     modelo.pvalues.drop("const").values,
        })
        tabla["Significativa (p<0.05)"] = np.where(tabla["P-value"] < 0.05, "Si", "No")
        tabla["Coeficiente"] = tabla["Coeficiente"].round(5)
        tabla["P-value"]     = tabla["P-value"].apply(lambda p: f"{p:.4e}")
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        # Conclusion resumida sobre las variables significativas.
        signif = tabla.loc[tabla["Significativa (p<0.05)"] == "Si", "Variable"].tolist()
        if signif:
            st.markdown(
                f'<div class="conc-box"><b>Interpretacion:</b> con un nivel de '
                f'significancia del 5%, las variables con efecto estadisticamente '
                f'significativo son: <b>{", ".join(signif)}</b>.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="conc-box"><b>Interpretacion:</b> ninguna variable resulta '
                'significativa al 5% en este modelo.</div>',
                unsafe_allow_html=True,
            )
    except Exception as err:
        # Si el modelo estadistico no converge, se informa sin interrumpir la app.
        st.info(f"No se pudieron calcular los p-values para esta configuracion ({err}).")


def _conclusion_regresion(r2):
    """Muestra una conclusion en lenguaje claro segun el R2 obtenido en regresion.

    El mensaje cambia segun la calidad del ajuste, en lugar de ser siempre el mismo.
    """
    if r2 >= 0.5:
        st.success(
            f"Buen resultado: el modelo explica cerca del {r2*100:.0f}% de la variacion "
            f"de Y (R2 = {r2:.3f}). Las variables elegidas si ayudan a predecirla."
        )
    elif r2 >= 0.1:
        st.warning(
            f"Resultado debil: el modelo solo explica alrededor del {r2*100:.0f}% de Y "
            f"(R2 = {r2:.3f}). Las variables aportan poco; pruebe con otras variables X."
        )
    else:
        st.error(
            f"El modelo practicamente no predice Y (R2 = {r2:.3f}, casi 0). Las variables "
            "elegidas no tienen una relacion clara con Y en este dataset. Sugerencias: "
            "pruebe otras variables X, o cambie a Clasificacion para predecir 'Fraude', "
            "que es donde este conjunto de datos da mejores resultados. "
            "Nota: aunque alguna variable aparezca como 'significativa' en los p-values, "
            "con muchos datos eso puede ocurrir sin que el modelo prediga bien; guiese por el R2."
        )


def _conclusion_clasificacion(acc, baseline):
    """Muestra una conclusion en lenguaje claro comparando la exactitud con el baseline.

    El baseline es el acierto de adivinar siempre la clase mas frecuente; el modelo
    solo es util si supera ese valor de forma clara.
    """
    mejora = acc - baseline
    if mejora >= 0.05:
        st.success(
            f"Buen resultado: el modelo acierta el {acc*100:.1f}% de las veces, por "
            f"encima del {baseline*100:.1f}% que se lograria adivinando siempre la clase "
            "mas comun. Las variables elegidas aportan informacion util."
        )
    elif mejora >= 0.01:
        st.warning(
            f"Resultado modesto: el modelo acierta el {acc*100:.1f}%, apenas por encima "
            f"del {baseline*100:.1f}% de adivinar la clase mas comun. Pruebe con otras "
            "variables X o con otro algoritmo."
        )
    else:
        st.error(
            f"El modelo no mejora a adivinar: acierta el {acc*100:.1f}% frente al "
            f"{baseline*100:.1f}% de la clase mas comun. Las variables elegidas no estan "
            "ayudando. Pruebe otras variables X (por ejemplo, las banderas de riesgo) o "
            "cambie de algoritmo."
        )
