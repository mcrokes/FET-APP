from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.proccessor.model.explainers.decision_tree_surrogate import SurrogateTree
from app.proccessor.model.rules_uploader import graph_rules
from app.proccessor.models import ModelForProccess

surrogateLayout = html.Div(
    [
        dcc.Loading(
            [
                dbc.Row(
                    [
                        dbc.Label(
                            "Profundidad máxima del árbol subrogado",
                            style={"text-align": "center", "font-weight": "bold"},
                            html_for="max-depth-input-row",
                            width=5,
                        ),
                        dbc.Col(
                            dbc.Input(
                                type="number",
                                id="max-depth-input-row",
                                min=3,
                                max=20,
                                value=5,
                                style={"width": "55%", "margin": "auto"},
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Reconstruir Árbol",
                                id="surrogate-tree-reconstruction-btn",
                                n_clicks=0,
                                className="btn-secondary",
                            ),
                        ),
                    ],
                    style={"margin-left": "6rem"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.H3("REGLAS DEL MODELO"),
                                        html.Div(id="rules-output-upload"),
                                    ]
                                )
                            ]
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        html.H3("VISUALIZACION DEL ARBOL"),
                        html.Img(id="tree-visual-output-upload"),
                    ],
                    style={"padding-top": "20px"},
                ),
            ]
        )
    ],
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def surrogateCallbacks(app, furl: Function):
    @app.callback(
        Output("rules-output-upload", "children"),
        Output("tree-visual-output-upload", "src"),
        Input("surrogate-tree-reconstruction-btn", "n_clicks"),
        State("max-depth-input-row", "value"),
        Input("path", "href"),
    )
    def refresh_surrogate_layout(n, max_depht, cl):
        f = furl(cl)
        param1 = f.args["model_id"]
        model_x: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == param1
            ).first()

        classifier_model: RandomForestClassifier = model_x.to_dict()["model"]
        classifier_dataset: pd.DataFrame = model_x.to_dict()["dataset"]

        target_description = {
                "column_name": "Sobreviviente",
                "variables": [
                    {"old_value": 0, "new_value": "Muere"},
                    {"old_value": 1, "new_value": "Vive"},
                ],
            }
            
        surrogate_model = SurrogateTree(
                class_names=list(set(classifier_dataset[model_x.target_row])),
                df=classifier_dataset,
                max_depth=max_depht,
                model=classifier_model,
                target=model_x.target_row,
                x_train=classifier_dataset.drop(columns=model_x.target_row),
            )
        rg = surrogate_model.get_rules()
        tg = surrogate_model.graph_tree()

        return graph_rules(rg), tg

