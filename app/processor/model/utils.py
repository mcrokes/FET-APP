from dash import html, dcc


def graph_card(figura):
    div = html.Div(
        [
            dcc.Graph(figure=figura)
        ]
    )
    return div
