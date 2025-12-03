import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc

# ============================================================
# CONFIG
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

BACKGROUND_COLOR = "#f4f5fb"
CARD_BG = "#ffffff"
CARD_SHADOW = "0 4px 12px rgba(0, 0, 0, 0.06)"


# ============================================================
# HELPERS
# ============================================================
def load_model(name):
    path = os.path.join(MODELS, name)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[ERROR] Falló cargar {name}: {e}")
            return None
    print(f"[WARN] Archivo no encontrado: {name}")
    return None


def construct_features(values_dict, columns):
    if not columns:
        return pd.DataFrame()

    base = {c: 0 for c in columns}
    for k, v in values_dict.items():
        if k in base:
            base[k] = 0 if v is None else v

    return pd.DataFrame([base])[columns]


def dynamic_inputs(columns, type_id):
    if not columns:
        return html.Div("Modelo sin columnas cargadas.", className="text-danger")

    layout = []
    for col in columns:
        layout.append(
            dbc.Row([
                dbc.Label(col, className="text-muted mt-2"),
                dcc.Input(
                    id={"type": type_id, "column": col},
                    type="number",
                    value=0,
                    step=1,
                    className="form-control mb-2"
                )
            ])
        )
    return html.Div(layout)


def pretty_card(children, extra=None):
    style = {
        "backgroundColor": CARD_BG,
        "padding": "20px",
        "borderRadius": "14px",
        "boxShadow": CARD_SHADOW
    }
    if extra:
        style.update(extra)
    return html.Div(children, style=style)


# ============================================================
# CARGA DE MODELOS
# ============================================================
modelo_reg_nn      = load_model("modelo_regresion_final.pkl")
escalador_reg_nn   = load_model("escalador_regresion.pkl")
cols_reg_nn        = load_model("columnas_regresion.pkl") or []

modelo_reg_tec     = load_model("modelo_regresion_tec.pkl")
cols_reg_tec       = load_model("columnas_regresion_tec.pkl") or []

modelo_clf_nn      = load_model("modelo_clasificacion_final.pkl")
escalador_clf_nn   = load_model("escalador_clasificacion.pkl")
cols_clf_nn        = load_model("columnas_clasificacion.pkl") or []

modelo_clf_tec     = load_model("modelo_clasificacion_tec.pkl")
cols_clf_tec       = load_model("columnas_clf_tec.pkl") or []

print("[INFO] Columnas cargadas correctamente.")


# ============================================================
# DATASET PARA GRÁFICOS
# ============================================================
df = pd.read_csv(os.path.join(DATA, "airbnb_limpio.csv"))

df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("$", "")
    .str.replace(",", "")
    .str.replace("USD", "")
    .str.strip()
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

max_price = df["price"].quantile(0.99)
fig_hist = px.histogram(df[df["price"] <= max_price], x="price", nbins=40)
fig_box = px.box(df[df["price"] <= max_price], x="room_type", y="price")
fig_scatter = px.scatter(
    df[df["price"] <= max_price],
    x="accommodates",
    y="price",
    color="number_of_reviews"
)


# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


# ============================================================
# LAYOUT — ESTILO INSPIRADO EN TU AMIGA
# ============================================================
app.layout = html.Div(
    [
        html.H1(
            "Airbnb ML Dashboard",
            style={
                "textAlign": "center",
                "marginTop": "20px",
                "fontFamily": "Helvetica Neue"
            },
        ),

        html.Div([
            html.Div([
                pretty_card(
                    [
                        html.H4("1. Selección de modelo"),
                        dcc.Dropdown(
                            id="selector-modelo",
                            options=[
                                {"label": "Regresión (NN)", "value": "reg_nn"},
                                {"label": "Regresión (TEC)", "value": "reg_tec"},
                                {"label": "Clasificación (NN)", "value": "clf_nn"},
                                {"label": "Clasificación (TEC)", "value": "clf_tec"},
                            ],
                            value="reg_nn",
                            clearable=False,
                            className="mb-3"
                        ),
                        html.Hr(),
                        html.Div(id="inputs-dinamicos"),
                        dbc.Button(
                            "Calcular",
                            id="btn-calcular",
                            color="primary",
                            className="mt-3 w-100"
                        ),
                        html.H3(id="output-principal", className="text-center mt-4"),
                        html.P(id="output-secundario", className="text-center text-muted")
                    ]
                )
            ], style={"flex": "0 0 32%", "minWidth": "280px"}),

            html.Div([
                pretty_card(
                    [
                        dbc.Tabs([
                            dbc.Tab(dcc.Graph(figure=fig_hist), label="Histograma"),
                            dbc.Tab(dcc.Graph(figure=fig_box), label="Boxplot"),
                            dbc.Tab(dcc.Graph(figure=fig_scatter), label="Dispersión"),
                        ])
                    ],
                    extra={"height": "100%"}
                )
            ], style={"flex": "0 0 65%", "minWidth": "350px"})
        ],
        style={
            "display": "flex",
            "justifyContent": "center",
            "padding": "20px",
            "gap": "20px",
            "flexWrap": "wrap",
        })
    ],
    style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh"}
)


# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output("inputs-dinamicos", "children"),
    Input("selector-modelo", "value")
)
def render_inputs(modelo):
    if modelo == "reg_nn":
        return dynamic_inputs(cols_reg_nn, "reg_nn")
    if modelo == "reg_tec":
        return dynamic_inputs(cols_reg_tec, "reg_tec")
    if modelo == "clf_nn":
        return dynamic_inputs(cols_clf_nn, "clf_nn")
    if modelo == "clf_tec":
        return dynamic_inputs(cols_clf_tec, "clf_tec")
    return "Modelo no válido."


def get_values(values, ids, model_tag):
    out = {}
    for v, id_ in zip(values, ids):
        if id_["type"] == model_tag:
            out[id_["column"]] = 0 if v is None else v
    return out


@app.callback(
    Output("output-principal", "children"),
    Output("output-secundario", "children"),
    Input("btn-calcular", "n_clicks"),
    State("selector-modelo", "value"),
    State({"type": ALL, "column": ALL}, "value"),
    State({"type": ALL, "column": ALL}, "id"),
)
def calcular(n, modelo, values, ids):
    if not n:
        return "", ""

    # -----------------------------
    # REGRESIÓN NN
    # -----------------------------
    if modelo == "reg_nn":
        dic = get_values(values, ids, "reg_nn")
        X = construct_features(dic, cols_reg_nn)
        if escalador_reg_nn:
            X = escalador_reg_nn.transform(X)
        pred = float(modelo_reg_nn.predict(X)[0])
        return f"${pred:,.2f}", "Predicción de precio (NN)"

    # -----------------------------
    # REGRESIÓN TEC
    # -----------------------------
    if modelo == "reg_tec":
        dic = get_values(values, ids, "reg_tec")
        X = construct_features(dic, cols_reg_tec)
        pred = float(modelo_reg_tec.predict(X)[0])
        return f"${pred:,.2f}", "Predicción de precio (TEC)"

    # -----------------------------
    # CLASIFICACIÓN NN
    # -----------------------------
    if modelo == "clf_nn":
        dic = get_values(values, ids, "clf_nn")
        X = construct_features(dic, cols_clf_nn)
        if escalador_clf_nn:
            X = escalador_clf_nn.transform(X)

        if hasattr(modelo_clf_nn, "predict_proba"):
            prob = modelo_clf_nn.predict_proba(X)[0, 1]
        else:
            prob = modelo_clf_nn.predict(X)[0]

        cls = "RECOMENDADO" if prob >= 0.5 else "NO RECOMENDADO"
        return cls, f"Probabilidad {prob:.2%}"

    # -----------------------------
    # CLASIFICACIÓN TEC
    # -----------------------------
    if modelo == "clf_tec":
        dic = get_values(values, ids, "clf_tec")
        X = construct_features(dic, cols_clf_tec)

        if hasattr(modelo_clf_tec, "predict_proba"):
            prob = modelo_clf_tec.predict_proba(X)[0, 1]
        else:
            prob = modelo_clf_tec.predict(X)[0]

        cls = "RECOMENDADO" if prob >= 0.5 else "NO RECOMENDADO"
        return cls, f"Probabilidad {prob:.2%}"

    return "", ""


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)