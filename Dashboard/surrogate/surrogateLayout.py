import json
from pyclbr import Function
from dash import dcc, html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.proccessor.model.explainers.decision_tree_surrogate import (
    ExplainSingleTree,
)
from app.proccessor.models import (
    ExplainedModel,
    SurrogateTreeData,
    Tree,
)

id_sufix = ["surrogate"]
surrogateLayout = html.Div(
    [
        dcc.Loading(
            [
                html.Div(
                    [
                        html.Plaintext(
                            "Profundidad",
                            className="tree-creator-label",
                        ),
                        dbc.Input(
                            type="number",
                            id="max-depth-input-row",
                            min=3,
                            value=3,
                            className="tree-creator-input",
                        ),
                        html.Button(
                            "Reconstruir Árbol",
                            id="surrogate-tree-reconstruction-btn",
                            n_clicks=0,
                            className="tree-btn tree-creator-btn",
                        ),
                    ],
                    className="tree-creator",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [

                                        html.Div(
                                            [
                                                html.Plaintext(
                                                    "REGLAS DEL MODELO", className="rules-title"
                                                ),
                                                html.I(
                                                    id=f"{id_sufix[0]}-info",
                                                    className="fa fa-info-circle info-icon",
                                                ),
                                                dbc.Tooltip(
                                                    [
                                                        html.Plaintext(
                                                            [
                                                                """
                                                                Árbol Subrogado: Un árbol de decisión que se entrena 
                                                                con las predicciones del modelo de Random Forest original, 
                                                                permitiendo simplificar el bosque para interpretar la 
                                                                salida del modelo de manera más fácil y comprensible.
                                                                """,
                                                            ]
                                                        ),
                                                    ],
                                                    className="personalized-tooltip",
                                                    target=f"{id_sufix[0]}-info",
                                                ),
                                            ],
                                            className="title-hint-container",
                                        ),
                                        html.Div(
                                            id="rules-output-upload",
                                            className="rules-table-container",
                                        ),
                                    ],
                                    className="container"
                                )
                            ]
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                html.Div(
                    [
                        html.Plaintext(
                            "VISUALIZACION DEL ARBOL", className="rules-title"
                        ),
                        html.Button(
                            "MOSTRAR",
                            id="build-tree-btn",
                            n_clicks=0,
                            className="tree-btn tree-creator-btn",
                            style={"margin-left": "3rem"},
                        ),
                        html.Button(
                            "DESCARGAR",
                            id="download-tree-btn",
                            n_clicks=0,
                            hidden=True,
                            className="tree-btn tree-creator-btn",
                            style={"margin-left": "3rem"},
                        ),
                        dbc.Tooltip(
                            [
                                html.Plaintext(
                                    [
                                        "* Para visualizar el árbol deberá instalar el software ",
                                        html.Strong(
                                            html.A("Graphviz", href="https://graphviz.org/download/", target='_blank')),
                                        " en su ordenador.",
                                    ]
                                ),
                                html.Plaintext(
                                    [
                                        "* Las muestras con las que se hace la visualización ",
                                        html.Strong("no son las mismas"),
                                        " que las de las reglas de modelo.",
                                    ]
                                ),
                            ],
                            autohide=False,
                            className="personalized-tooltip",
                            target="build-tree-btn",
                        ),
                    ],
                    className="tree-creator container",
                    style={"padding-top": "20px", "justify-content": "flex-start"},
                ),
                dcc.Store(id="svg-holder", data={}),
                html.Div(
                    [html.Img(id="tree-visual-output-upload")],
                    className="tree-img-container",
                ),
                dcc.Download(id='download-svg')
            ]
        )
    ],
    style={"margin": "auto"},
)


def surrogateCallbacks(app, furl: Function, isRegressor: bool = False):
    @app.callback(
        Output("rules-output-upload", "children"),
        Output("max-depth-input-row", "max"),
        Output("tree-visual-output-upload", "src"),
        Output("download-tree-btn", "hidden"),
        State("max-depth-input-row", "value"),
        Input("surrogate-tree-reconstruction-btn", "n_clicks"),
        Input("path", "href"),
    )
    def refresh_surrogate_layout(max_depht, n, cl):
        f = furl(cl)
        model_id = f.args["model_id"]

        surrogate_model: SurrogateTreeData = (
            SurrogateTreeData.query.filter(
                SurrogateTreeData.explained_model_id == model_id
            )
            .join(SurrogateTreeData.tree)
            .filter(Tree.depth == max_depht)
            .first()
        )

        lenght = len(
            SurrogateTreeData.query.filter(
                SurrogateTreeData.explained_model_id == model_id
            ).all()
        )

        model_x: ExplainedModel = surrogate_model.explained_model
        if not isRegressor:
            class_names = [
                var["new_value"]
                for var in model_x.explainer_classifier.getElement("target_names_dict")["variables"]
            ]
            rules = ExplainSingleTree.get_rules(
                tree_model=surrogate_model.getElement("tree_model").tree_,
                q_variables=[
                    var["column_name"] for var in model_x.getElement("q_variables_dict")
                ],
                q_variables_values=model_x.getElement("q_variables_dict"),
                features=surrogate_model.getElement("tree_model").feature_names_in_,
                class_names=class_names,
                model_type="Classifier"
            )
        else:
            rules = ExplainSingleTree.get_rules(
                tree_model=surrogate_model.getElement("tree_model").tree_,
                q_variables=[
                    var["column_name"] for var in model_x.getElement("q_variables_dict")
                ],
                q_variables_values=model_x.getElement("q_variables_dict"),
                features=surrogate_model.getElement("tree_model").feature_names_in_,
                class_names=None,
                model_type="Regressor"
            )

        rules_table = []
        for index, rule in enumerate(rules):
            causes = []
            for cause in rule["causes"]:
                value_cell = ""
                for jindex, value in enumerate(cause["value"]):
                    if jindex > 0:
                        value_cell += f" o {value}"
                    else:
                        value_cell += value
                causes.append(
                    html.Tr(
                        [
                            html.Td(cause["item"], style={"width": "40%"}),
                            html.Td(
                                cause["sign"],
                                style={"width": "20%"},
                            ),
                            html.Td(f"{value_cell}", style={"width": "40%"}),
                        ]
                    )
                )
            causes_body = [html.Tbody(causes)]
            causes_table = dbc.Table(
                causes_body,
                style={"margin": "0", "border": "solid #2A3F54 1pt"},
                className="rules-table",
            )

            rules_table.append(
                html.Tr(
                    [
                        html.Td(index + 1),
                        html.Td(causes_table, style={"padding": "0"}),
                        html.Td(rule["target_value"]),
                        html.Td(rule["probability"]),
                        html.Td(rule["samples_amount"]),
                    ]
                )
            )

        sub_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("PREDICTOR", style={"width": "40%"}),
                        html.Th("CONDICIÓN", style={"width": "20%"}),
                        html.Th("VALOR", style={"width": "40%"}),
                    ]
                )
            )
        ]
        sub_header_table = dbc.Table(
            sub_header, style={"margin": "0"}, className="rules-table"
        )

        table_header = [
            html.Thead(
                [
                    html.Tr(
                        [
                            html.Th("REGLA", rowSpan=2),
                            html.Th("CAUSAS"),
                            html.Th("REULTADO", colSpan=3),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Th(sub_header_table, style={"padding": "0"}),
                            html.Th("VALOR OBJETIVO"),
                            html.Th("PROBABILIDAD"),
                            html.Th("MUESTRAS"),
                        ]
                    ),
                ], style={"position": "sticky", "top": "0"}
            )
        ]

        table_body = [html.Tbody(rules_table)]

        rg = dbc.Table(
            table_header + table_body, bordered=True, className="rules-table"
        )

        return rg, lenght + 2, "", True

    @app.callback(
        Output("tree-visual-output-upload", "src", allow_duplicate=True),
        Output("svg-holder", "data"),
        Output("download-tree-btn", "hidden", allow_duplicate=True),
        State("max-depth-input-row", "value"),
        State("path", "href"),
        Input("build-tree-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def build_img_tree(max_depht, cl, build):
        f = furl(cl)
        model_id = f.args["model_id"]

        surrogate_model: SurrogateTreeData = (
            SurrogateTreeData.query.filter(
                SurrogateTreeData.explained_model_id == model_id
            )
            .join(SurrogateTreeData.tree)
            .filter(Tree.depth == max_depht)
            .first()
        )

        model_x: ExplainedModel = surrogate_model.explained_model

        model: DecisionTreeClassifier | DecisionTreeRegressor = surrogate_model.getElement("tree_model")

        dataset: pd.DataFrame = (
            surrogate_model.explained_model.data_set_data.getElement(
                "dataset"
            )
        )
        target_row: str = surrogate_model.explained_model.getElement(
            "target_row"
        )
        try:
            target_description = model_x.explainer_classifier.getElement("target_names_dict")
            class_names = [
                element["new_value"] for element in target_description["variables"]
            ]
        except:
            class_names = None

        tg, viz = ExplainSingleTree.graph_tree(
            x_train=dataset.drop(columns=target_row),
            y_train=model_x.getElement("model").predict(dataset.drop(columns=target_row)),
            tree=model,
            class_names=class_names,
            feature_names=model.feature_names_in_,
        )
        # Convertir el árbol en un archivo SVG

        svg_str = viz.view().svg()

        # Devolver el archivo SVG como respuesta
        return tg, json.dumps(svg_str), False

    @app.callback(
        Output("download-svg", "data"),
        State("svg-holder", "data"),
        Input("download-tree-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_img_tree(data, download):
        if download:
            svg_str = json.loads(data)
            return dict(content=svg_str, filename='arbol.svg', type='text/svg+xml')
        raise PreventUpdate
