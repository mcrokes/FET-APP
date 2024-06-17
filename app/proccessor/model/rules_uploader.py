from dash import html


def graph_rules(rules):
    div = html.Div(
        [
            html.P(f"{r}") for r in rules
        ],
        style={"padding": 20, "color": "white", "font-weight": "bold", "background-color": "black"}
    )

    return div
