import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

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

PLOTLY_THEME = "simple_white"
PLOTLY_BG    = "#ffffff"

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


def _safe_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def run():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aprendizaje Automatico</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Entrena y evalua modelos de machine learning sobre los datasets</div>',
                unsafe_allow_html=True)

    ds_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
    df_raw  = DATASETS[ds_name]()

    df = df_raw.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_cols = _safe_numeric_cols(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuracion del modelo")

    task_type = st.sidebar.radio("Tipo de tarea", ["Regresion", "Clasificacion"])

    algos_reg = ["Regresion Lineal", "Arbol de Decision (Reg.)", "Random Forest (Reg.)"]
    algos_cls = ["Regresion Logistica", "Arbol de Decision (Cls.)", "Random Forest (Cls.)"]
    algos     = algos_reg if task_type == "Regresion" else algos_cls
    algorithm = st.sidebar.selectbox("Algoritmo", algos)

    var_dep   = st.sidebar.selectbox("Variable dependiente (Y)", numeric_cols, index=0)
    avail_x   = [c for c in numeric_cols if c != var_dep]
    var_indep = st.sidebar.selectbox("Variable independiente (X)", avail_x,
                                     index=min(1, len(avail_x) - 1))

    train_pct = st.sidebar.slider("Porcentaje de entrenamiento (%)", 50, 90, 80, step=5)
    test_pct  = 100 - train_pct

    st.markdown(
        f"**Configuracion:** `{algorithm}` · Y = `{var_dep}` · X = `{var_indep}` · "
        f"Train {train_pct}% / Test {test_pct}%"
    )

    if st.button("Entrenar Modelo", type="primary"):
        _train_and_display(df, var_dep, var_indep, algorithm, train_pct / 100, task_type)


def _train_and_display(df, y_col, x_col, algorithm, train_ratio, task_type):
    sub = df[[x_col, y_col]].dropna()
    X   = sub[[x_col]].values
    y   = sub[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )

    if algorithm == "Regresion Lineal":
        model = LinearRegression()
    elif algorithm == "Arbol de Decision (Reg.)":
        model = DecisionTreeRegressor(max_depth=6, random_state=42)
    elif algorithm == "Random Forest (Reg.)":
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    elif algorithm == "Regresion Logistica":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "Arbol de Decision (Cls.)":
        model = DecisionTreeClassifier(max_depth=6, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    st.markdown("---")
    st.subheader("Metricas del modelo")

    if task_type == "Regresion":
        mse   = mean_squared_error(y_test, y_pred_test)
        rmse  = np.sqrt(mse)
        mae   = mean_absolute_error(y_test, y_pred_test)
        r2    = r2_score(y_test, y_pred_test)
        r2_tr = r2_score(y_train, y_pred_train)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("R2 (Test)",   f"{r2:.4f}")
        c2.metric("R2 (Train)",  f"{r2_tr:.4f}")
        c3.metric("RMSE",        f"{rmse:.4f}")
        c4.metric("MAE",         f"{mae:.4f}")
        c5.metric("MSE",         f"{mse:.4f}")

        if hasattr(model, "coef_"):
            st.metric("Coeficiente", f"{model.coef_[0]:.4f}")
        if hasattr(model, "intercept_"):
            intval = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_[0]
            st.metric("Intercepto", f"{intval:.4f}")

        fig = go.Figure()
        idx_tr = np.random.choice(len(X_train), min(500, len(X_train)), replace=False)
        idx_te = np.random.choice(len(X_test),  min(200, len(X_test)),  replace=False)

        fig.add_trace(go.Scatter(
            x=X_train[idx_tr, 0], y=y_train[idx_tr],
            mode="markers", name="Train (real)",
            marker=dict(color="#4338ca", opacity=0.5, size=5),
        ))
        fig.add_trace(go.Scatter(
            x=X_test[idx_te, 0], y=y_test[idx_te],
            mode="markers", name="Test (real)",
            marker=dict(color="#dc2626", opacity=0.5, size=5),
        ))
        x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        fig.add_trace(go.Scatter(
            x=x_range[:, 0], y=model.predict(x_range),
            mode="lines", name="Prediccion",
            line=dict(color="#0369a1", width=2),
        ))
        fig.update_layout(
            title=f"{algorithm}: {x_col} vs {y_col}",
            xaxis_title=x_col, yaxis_title=y_col,
            template=PLOTLY_THEME,
            paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        acc   = accuracy_score(y_test, y_pred_test)
        acc_tr= accuracy_score(y_train, y_pred_train)
        prec  = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec   = recall_score(y_test,    y_pred_test, average="weighted", zero_division=0)
        f1    = f1_score(y_test,        y_pred_test, average="weighted", zero_division=0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy (Test)",  f"{acc:.4f}")
        c2.metric("Accuracy (Train)", f"{acc_tr:.4f}")
        c3.metric("Precision",        f"{prec:.4f}")
        c4.metric("Recall",           f"{rec:.4f}")
        c5.metric("F1-Score",         f"{f1:.4f}")

        idx_te = np.random.choice(len(X_test), min(300, len(X_test)), replace=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_test[idx_te, 0], y=y_test[idx_te],
            mode="markers", name="Real",
            marker=dict(color="#4338ca", opacity=0.6, size=6),
        ))
        fig.add_trace(go.Scatter(
            x=X_test[idx_te, 0], y=y_pred_test[idx_te],
            mode="markers", name="Prediccion",
            marker=dict(color="#dc2626", opacity=0.6, size=6, symbol="x"),
        ))
        fig.update_layout(
            title=f"Predicciones vs Reales ({algorithm})",
            xaxis_title=x_col, yaxis_title=y_col,
            template=PLOTLY_THEME,
            paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(np.unique(y_test)) <= 10:
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm = px.imshow(
                cm, text_auto=True,
                title="Matriz de Confusion",
                color_continuous_scale="Blues",
                template=PLOTLY_THEME,
            )
            fig_cm.update_layout(paper_bgcolor=PLOTLY_BG)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(
        '<div class="conc-box">Modelo entrenado exitosamente. '
        'Ajuste los parametros en la barra lateral para comparar diferentes configuraciones.</div>',
        unsafe_allow_html=True,
    )
