from dash import Dash, html

app = Dash(__name__)
app.layout = html.H1("Hola desde Dash")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)