# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: jimmybow
"""
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, State, Output
from .Dash_fun import apply_layout_with_auth, load_object, save_object
from app.base.models import User

url_base = "/dash/app1/"

layout = html.Div(
    [
        dcc.Graph(id="example"),
        dcc.Slider(
            id="slider",
            min=0,
            max=100,
            value=50,
            marks={i: str(i) for i in range(0, 101, 10)},
        ),
        html.A(
            html.Button(
                "Volver a la página de inicio",
                style={"margin-top": "20px"},
            ),
            id="back_button",
            href="/",
        ),
    ]
)


def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=url_base)
    apply_layout_with_auth(app, layout)

    @app.callback(
        Output("example", "figure"),
        [Input("slider", "value")],
        [State("back_button", "n_clicks")],
    )
    def update_graph(value, n_clicks):
        if n_clicks is not None and n_clicks > 0:
            return no_update
        users = User.query.all()

        data = {"users": [{"name": u.username, "email": u.email} for u in users]}

        return {
            "data": [
                {
                    "x": [d["name"] for d in data["users"]],
                    "y": [d["email"] for d in data["users"]],
                    "type": "bar",
                }
            ]
        }

    return app.server
