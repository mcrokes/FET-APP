from pyclbr import Function
from turtle import width
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.model.explainers.decision_tree_surrogate import (
    ExplainSingleTree,
    SurrogateTree,
)
from app.proccessor.model.rules_uploader import graph_rules
from app.proccessor.models import (
    ModelForProccess,
    SurrogateTreeClassifierData,
    Tree,
    TreeClassifierRule,
    TreeClassifierRuleCause,
)

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
        Input("surrogate-tree-reconstruction-btn", "n_clicks"),
        State("max-depth-input-row", "value"),
        Input("path", "href"),
    )
    def refresh_surrogate_layout(n, max_depht, cl):
        f = furl(cl)
        param1 = f.args["model_id"]

        surrogate_model: SurrogateTreeClassifierData = (
            SurrogateTreeClassifierData.query.filter(
                SurrogateTreeClassifierData.explained_classifier_model_id == 6
            ).first()
        )

        tree: Tree = surrogate_model.tree
        rules = []
        for index, r in enumerate(tree.rules):
            rule: TreeClassifierRule = r
            causes = []
            for c in rule.causes:
                cause: TreeClassifierRuleCause = c
                causes.append(
                    html.Tr(
                        [
                            html.Td(
                                cause.getElement("predictor"), style={"width": "40%"}
                            ),
                            html.Td(
                                cause.getElement("relation_sign"),
                                style={"width": "20%"},
                            ),
                            html.Td(cause.getElement("value"), style={"width": "40%"}),
                        ]
                    )
                )
            causes_body = [html.Tbody(causes)]
            causes_table = dbc.Table(causes_body, bordered=True, style={"margin": "0"})

            rules.append(
                html.Tr(
                    [
                        html.Td(index+1),
                        html.Td(causes_table, style={"padding": "0"}),
                        html.Td(rule.getElement("target_value")),
                        html.Td(rule.getElement("probability")),
                        html.Td(rule.getElement("samples_amount")),
                    ]
                )
            )

        sub_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("PREDICTOR", style={"width": "40%"}),
                        html.Th("SIGN", style={"width": "20%"}),
                        html.Th("VALUE", style={"width": "40%"}),
                    ]
                )
            )
        ]
        sub_header_table = dbc.Table(sub_header, bordered=True, style={"margin": "0"})

        table_header = [
            html.Thead(
                [
                    html.Tr(
                        [                            
                            html.Th("RULE", rowSpan=2),
                            html.Th("CAUSES"),
                            html.Th("RESULTS", colSpan=3),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Th(sub_header_table, style={"padding": "0"}),
                            html.Th("TARGET VALUE"),
                            html.Th("PROBABILITY"),
                            html.Th("SAMPLES"),
                        ]
                    ),
                ]
            )
        ]

        table_body = [html.Tbody(rules)]

        rg = dbc.Table(table_header + table_body, bordered=True)

        # model: DecisionTreeClassifier = surrogate_model.getElement("tree_model")
        # dataset: pd.DataFrame = (
        #     surrogate_model.explained_classifier_model.data_set_data.getElement(
        #         "dataset"
        #     )
        # )
        # target_row: str = surrogate_model.explained_classifier_model.getElement(
        #     "target_row"
        # )
        # tg = ExplainSingleTree.graph_tree(
        #     x_train=dataset.drop(columns=target_row),
        #     y_train=dataset[target_row],
        #     tree=model,
        #     class_names=["muere", "vive"],
        #     feature_names=model.feature_names_in_,
        # )
        return rg
