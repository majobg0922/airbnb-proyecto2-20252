import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "Copia de airbnb_limpio.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    df = pd.DataFrame({
        "price": [100, 200, 150],
        "room_type": ["Private", "Entire", "Private"],
        "accommodates": [2, 4, 2],
        "number_of_reviews": [10, 5, 20]
    })

def cargar_modelo(nombre):
    path = os.path.join(BASE_DIR, nombre)
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

modelo_reg_nn = cargar_modelo("modelo_regresion_final.pkl")
escalador_reg_nn = cargar_modelo("escalador_regresion.pkl")
columnas_reg_nn = cargar_modelo("columnas_regresion.pkl") or []

modelo_clf_nn = cargar_modelo("modelo_clasificacion_final.pkl")
escalador_clf_nn = cargar_modelo("escalador_clasificacion.pkl")
columnas_clf = cargar_modelo("columnas_clasificacion.pkl") or []

modelo_reg_tec = cargar_modelo("modelo_regresion_tec.pkl")
columnas_reg_tec = cargar_modelo("columnas_regresion_tec.pkl") or []

modelo_clf_tec = cargar_modelo("modelo_clasificacion_tec.pkl")
columnas_clf_tec = cargar_modelo("columnas_clf_tec.pkl") or []


def construir_features_tec(valores_usuario: dict, columnas_totales):
    valores_completos = {col: 0 for col in columnas_totales}
    valores_completos.update(valores_usuario)
    df_user = pd.DataFrame([valores_completos])
    return df_user[columnas_totales]


def detectar_columnas_binarias(columnas):
    bin_keywords = ['is_', 'has_', 'binary', 'room_', 'pets_', 'wifi', 'parking', 'private', 'shared']
    return [col for col in columnas if any(k in col.lower() for k in bin_keywords)]


def detectar_columnas_numericas(columnas, binarias):
    return [col for col in columnas if col not in binarias]


def generar_inputs_dinamicos(columnas, tipo_id="input-tec"):
    if not columnas:
        return html.Div("No se cargaron columnas.")

    columnas_binarias = detectar_columnas_binarias(columnas)
    columnas_numericas = detectar_columnas_numericas(columnas, columnas_binarias)

    inputs = []

    for col in columnas_numericas:
        inputs.append(
            dbc.Col([
                dbc.Label(col.replace("_", " ").capitalize()),
                dcc.Input(
                    id={"type": tipo_id, "column": col},
                    type="number",
                    value=0,
                    step=1,
                    style={"width": "100%"},
                ),
            ], md=4, className="mb-3")
        )

    for col in columnas_binarias:
        inputs.append(
            dbc.Col([
                dbc.Label(col.replace("_", " ").capitalize()),
                dbc.Checklist(
                    options=[{"label": "Sí", "value": 1}],
                    value=[],
                    id={"type": tipo_id, "column": col},
                    switch=True,
                ),
            ], md=4, className="mb-3")
        )

    return dbc.Row(inputs)


def numeric_input(id_, label, value, min_=0, step=1):
    return dbc.Col(
        [
            dbc.Label(label),
            dcc.Input(
                id=id_,
                type="number",
                value=value,
                min=min_,
                step=step,
                style={"width": "100%"}
            )
        ],
        md=4,
        className="mb-3",
    )


inputs_layout_simple = dbc.Row([
    numeric_input("in_acc", "Accommodates", 2, min_=1),
    numeric_input("in_bed", "Bedrooms", 1, min_=0),
    numeric_input("in_bath", "Bathrooms", 1, min_=0, step=0.5),
    numeric_input("in_min", "Minimum nights", 2, min_=1),
    numeric_input("in_max", "Maximum nights", 30, min_=1),
    numeric_input("in_avail", "Availability 365", 180, min_=0),
])


# ===============================
# FIGURAS (HISTOGRAMA ARREGLADO)
# ===============================
# LIMPIEZA DE PRICE (FIX CRÍTICO)
if "price" in df.columns:
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("USD", "", regex=False)
        .str.strip()
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
if not df.empty:

    max_price = df["price"].quantile(0.99)

    fig_price_hist = px.histogram(
        df[df["price"] <= max_price],
        x="price",
        nbins=30,
        title="Distribución de precios"
    )
    fig_price_hist.update_layout(
        template="simple_white",
        bargap=0.05,
        xaxis_title="Precio (USD)",
        yaxis_title="Número de anuncios"
    )

    col_room = "room_type" if "room_type" in df.columns else df.columns[0]

    fig_price_room = px.box(
        df[df["price"] <= max_price],
        x=col_room,
        y="price",
        title="Precio por tipo"
    )

    if {"accommodates", "number_of_reviews"}.issubset(df.columns):
        fig_price_accom = px.scatter(
            df[df["price"] <= max_price],
            x="accommodates",
            y="price",
            color="number_of_reviews",
            title="Precio vs Capacidad"
        )
    else:
        fig_price_accom = px.scatter(title="Faltan columnas")
else:
    fig_price_hist = fig_price_room = fig_price_accom = px.scatter(title="Sin datos")


# ===============================
# APP LAYOUT
# ===============================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "Airbnb Dashboard ML"

app.layout = dbc.Container([
    html.H1("TABLERO AIRBNB — PREDICCIÓN DE PRECIOS", className="text-center my-4"),

    dcc.Tabs(
        id="tabs-principal",
        value="tab-visualizaciones",
        children=[
            dcc.Tab(label=" Visualizaciones", value="tab-visualizaciones"),
            dcc.Tab(label=" Regresión NN", value="tab-reg-nn"),
            dcc.Tab(label="Regresión Estadística", value="tab-reg-tec"),
            dcc.Tab(label=" Clasificación NN", value="tab-clf-nn"),
            dcc.Tab(label=" Clasificación Estadística", value="tab-clf-tec"),
        ]
    ),
    html.Br(),
    html.Div(id="contenido-tabs"),
], fluid=True)


# ===============================
# CALLBACKS DE RENDERIZADO
# ===============================

@app.callback(
    Output("contenido-tabs", "children"),
    Input("tabs-principal", "value")
)
def render_tab(tab_value):

    if tab_value == "tab-visualizaciones":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_price_hist), md=6),
                dbc.Col(dcc.Graph(figure=fig_price_room), md=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_price_accom), md=12)
            ]),
        ], fluid=True)

    elif tab_value == "tab-reg-nn":
        return dbc.Container([
            html.H3("Regresión Neural Network (Precio)", className="mb-3"),
            inputs_layout_simple,
            dbc.Button("Calcular Precio (NN)", id="btn_reg_nn",
                       color="primary", className="mt-2"),
            html.H4(id="out_reg_nn", className="mt-4 text-primary"),
        ])

    elif tab_value == "tab-reg-tec":
        return dbc.Container([
            html.H3("Regresión Modelo Estadístico (TEC)", className="mb-3"),
            generar_inputs_dinamicos(columnas_reg_tec, tipo_id="input-reg-tec"),
            dbc.Button("Calcular Precio (TEC)", id="btn_reg_tec",
                       color="secondary", className="mt-2"),
            html.H4(id="out_reg_tec", className="mt-4 text-secondary"),
        ])

    elif tab_value == "tab-clf-nn":
        return dbc.Container([
            html.H3("Clasificación NN (Recomendación)", className="mb-3"),
            inputs_layout_simple,
            dbc.Button("Clasificar (NN)", id="btn_clf_nn",
                       color="success", className="mt-2"),
            html.H4(id="out_clf_nn", className="mt-4"),
            html.P(id="out_clf_nn_prob", className="text-muted"),
        ])

    elif tab_value == "tab-clf-tec":
        return dbc.Container([
            html.H3("Clasificación Estadística (TEC)", className="mb-3"),
            generar_inputs_dinamicos(columnas_clf_tec, tipo_id="input-clf-tec"),
            dbc.Button("Clasificar (TEC)", id="btn_clf_tec",
                       color="info", className="mt-2"),
            html.H4(id="out_clf_tec", className="mt-4"),
            html.P(id="out_clf_tec_prob", className="text-muted"),
        ])


# ===============================
# CALLBACKS DE LÓGICA
# ===============================

@app.callback(
    Output("out_reg_nn", "children"),
    Input("btn_reg_nn", "n_clicks"),
    [
        State("in_acc", "value"),
        State("in_bed", "value"),
        State("in_bath", "value"),
        State("in_min", "value"),
        State("in_max", "value"),
        State("in_avail", "value"),
    ]
)
def predict_reg_nn(n_clicks, acc, bed, bath, min_n, max_n, avail):
    if not n_clicks or modelo_reg_nn is None:
        return ""

    valores = {
        "accommodates": acc,
        "bedrooms": bed,
        "bathrooms": bath,
        "minimum_nights": min_n,
        "maximum_nights": max_n,
        "availability_365": avail,
    }

    X = construir_features_tec(valores, columnas_reg_nn)

    if escalador_reg_nn is not None:
        X_scaled = escalador_reg_nn.transform(X)
    else:
        X_scaled = X

    pred_raw = float(modelo_reg_nn.predict(X_scaled).flatten()[0])

    if pred_raw < 0:
        pred_raw = 0

    return f"Precio estimado (NN): ${pred_raw:,.2f} USD"


@app.callback(
    Output("out_clf_nn", "children"),
    Output("out_clf_nn_prob", "children"),
    Input("btn_clf_nn", "n_clicks"),
    [
        State("in_acc", "value"),
        State("in_bed", "value"),
        State("in_bath", "value"),
        State("in_min", "value"),
        State("in_max", "value"),
        State("in_avail", "value"),
    ]
)
def predict_clf_nn(n_clicks, acc, bed, bath, min_n, max_n, avail):
    if not n_clicks or modelo_clf_nn is None:
        return "", ""

    valores = {
        "accommodates": acc,
        "bedrooms": bed,
        "bathrooms": bath,
        "minimum_nights": min_n,
        "maximum_nights": max_n,
        "availability_365": avail
    }

    X = construir_features_tec(valores, columnas_clf)

    if escalador_clf_nn is not None:
        X_final = escalador_clf_nn.transform(X)
    else:
        X_final = X

    if hasattr(modelo_clf_nn, "predict_proba"):
        proba = float(modelo_clf_nn.predict_proba(X_final)[0, 1])
    else:
        proba = float(modelo_clf_nn.predict(X_final).ravel()[0])

    proba = max(0.0, min(1.0, proba))

    clase = "Recomendado" if proba > 0.5 else "No recomendado"
    style = {"color": "green" if proba > 0.5 else "red"}

    return html.Span(clase, style=style), f"Probabilidad: {proba:.2%}"


@app.callback(
    Output("out_clf_tec", "children"),
    Output("out_clf_tec_prob", "children"),
    Input("btn_clf_tec", "n_clicks"),
    State({"type": "input-clf-tec", "column": ALL}, "value"),
    State({"type": "input-clf-tec", "column": ALL}, "id"),
)
def predict_clf_tec(n_clicks, values, ids):
    if not n_clicks or modelo_clf_tec is None:
        return "", ""

    valores = {}
    for val, id_dict in zip(values, ids):
        col = id_dict["column"]
        if isinstance(val, list):
            valores[col] = 1 if val else 0
        else:
            valores[col] = val or 0

    X_df = pd.DataFrame([valores])
    X_df = X_df.reindex(columns=columnas_clf_tec, fill_value=0)

    if hasattr(modelo_clf_tec, "predict_proba"):
        proba = float(modelo_clf_tec.predict_proba(X_df)[0, 1])
    else:
        proba = float(modelo_clf_tec.predict(X_df).ravel()[0])

    proba = max(0.0, min(1.0, proba))

    clase = "Recomendado" if proba > 0.5 else "No recomendado"
    style = {"color": "green" if proba > 0.5 else "red"}

    return html.Span(clase, style=style), f"Probabilidad: {proba:.2%}"


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)