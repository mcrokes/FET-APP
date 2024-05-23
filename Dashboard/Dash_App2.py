# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: jimmybow
"""
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, State, Output

from app.proccessor.models import ModelForProccess
from .Dash_fun import apply_layout_with_auth, load_object, save_object
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from furl import furl

url_base = "/dash/app2/"

layout = html.Div(
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


def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)

    # @app.callback(Output("target", "children"), [Input("input_text", "value")])
    # def callback_fun(value):
    #     return "your input is {}".format(value)

    print("APP CREATED")
    @app.callback(
        Output("dataset-title", "children"), Output("dataset-view", "children"),
        # Input("dataset-title", "children"),
        Input('path', 'href')
    )
    def graph_explainers(cl):
        f = furl(cl)
        param1= f.args['user_id']
        try:
            model_x: ModelForProccess = ModelForProccess.query.filter(ModelForProccess.id == param1).first()
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

    return app.server
