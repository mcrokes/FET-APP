from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

from app.proccessor.models import ModelForProccess

datasetLayout = html.Div(
    [
        dcc.Location(id='path'),
        dcc.Loading(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.H3(
                                            ["DATASET ", html.Span(id="dataset-title")],
                                            style={"text-align": "center"},
                                        ),
                                        html.Div(id="dataset-view"),
                                    ]
                                )
                            ],
                            style={"margin": "auto"},
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("DESCRIPCIÓN PREDICTORES"),
                                html.Div(id="features-description"),
                            ],
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        html.H3("METRICAS DEL DATA SET"),
                        html.H4("Variables Numericas"),
                        html.Div(id="numeric-plot"),
                        html.H4("Variables Objeto"),
                        html.Div(id="object-plot"),
                    ],
                    style={"padding-top": "20px"},
                ),
            ]
        )
    ],
    className="section-content",
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)

def datasetCallbacks(app, furl:Function):    
    @app.callback(
        Output("dataset-title", "children"),
        Output("dataset-view", "children"),
        Input("path", "href"),
    )
    def graph_explainers(cl):
        f = furl(cl)
        param1 = f.args["model_id"]
        try:
            model_x: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == param1
            ).first()
            df = model_x.to_dict()["dataset"]
            dtt = "Title"
            return (
                dtt,
                html.Div(
                    [
                        dash_table.DataTable(
                            data=df.to_dict("records"),
                            columns=[{"name": i, "id": i} for i in df.columns],
                            page_size=10,
                            # style_header={
                            #     "backgroundColor": "rgb(30, 30, 30)",
                            #     "color": "white",
                            # },
                            # style_data={
                            #     "backgroundColor": "rgb(50, 50, 50)",
                            #     "color": "white",
                            # },
                        ),
                    ]
                ),
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
