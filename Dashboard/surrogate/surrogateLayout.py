from pyclbr import Function
from dash import dcc, html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.model.explainers.decision_tree_surrogate import (
    ExplainSingleTree,
)
from app.proccessor.models import (
    ExplainedClassifierModel,
    SurrogateTreeClassifierData,
    Tree,
)

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
                                dbc.Row(
                                    [
                                        html.Plaintext(
                                            "REGLAS DEL MODELO", className="rules-title"
                                        ),
                                        html.Div(
                                            id="rules-output-upload",
                                            className="rules-table-container",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
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
                        dbc.Tooltip(
                            [
                                html.Plaintext(
                                    [
                                        "* Para visualizar el árbol deberá instalar el software ",
                                        html.Strong("Graphviz"),
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
                            className="personalized-tooltip",
                            target="build-tree-btn",
                        ),
                    ],
                    className="tree-creator",
                    style={"padding-top": "20px", "justify-content": "flex-start"},
                ),
                html.Div(
                    [html.Img(id="tree-visual-output-upload")],
                    className="tree-img-container",
                ),
            ]
        )
    ],
    style={"margin": "auto"},
)


def surrogateCallbacks(app, furl: Function):
    @app.callback(
        Output("rules-output-upload", "children"),
        Output("max-depth-input-row", "max"),
        Output("tree-visual-output-upload", "src"),
        State("max-depth-input-row", "value"),
        Input("surrogate-tree-reconstruction-btn", "n_clicks"),
        Input("path", "href"),
    )
    def refresh_surrogate_layout(max_depht, n, cl):
        f = furl(cl)
        model_id = f.args["model_id"]

        surrogate_model: SurrogateTreeClassifierData = (
            SurrogateTreeClassifierData.query.filter(
                SurrogateTreeClassifierData.explained_classifier_model_id == model_id
            )
            .join(SurrogateTreeClassifierData.tree)
            .filter(Tree.depth == max_depht)
            .first()
        )

        lenght = len(
            SurrogateTreeClassifierData.query.filter(
                SurrogateTreeClassifierData.explained_classifier_model_id == model_id
            ).all()
        )

        model_x: ExplainedClassifierModel = surrogate_model.explained_classifier_model
        rules = ExplainSingleTree.get_rules(
            model=surrogate_model.getElement("tree_model").tree_,
            q_variables=[
                var["column_name"] for var in model_x.getElement("q_variables_dict")
            ],
            q_variables_values=model_x.getElement("q_variables_dict"),
            features=surrogate_model.getElement("tree_model").feature_names_in_,
            class_names=[
                var["new_value"]
                for var in model_x.getElement("target_names_dict")["variables"]
            ],
            target=model_x.getElement("target_row"),
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
                        html.Th("SIGN", style={"width": "20%"}),
                        html.Th("VALUE", style={"width": "40%"}),
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

        table_body = [html.Tbody(rules_table)]

        rg = dbc.Table(
            table_header + table_body, bordered=True, className="rules-table"
        )

        return rg, lenght + 2, ""

    @app.callback(
        Output("tree-visual-output-upload", "src", allow_duplicate=True),
        State("max-depth-input-row", "value"),
        State("path", "href"),
        Input("build-tree-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def build_img_tree(max_depht, cl, build):
        f = furl(cl)
        model_id = f.args["model_id"]

        surrogate_model: SurrogateTreeClassifierData = (
            SurrogateTreeClassifierData.query.filter(
                SurrogateTreeClassifierData.explained_classifier_model_id == model_id
            )
            .join(SurrogateTreeClassifierData.tree)
            .filter(Tree.depth == max_depht)
            .first()
        )

        model_x: ExplainedClassifierModel = surrogate_model.explained_classifier_model

        model: DecisionTreeClassifier = surrogate_model.getElement("tree_model")
        dataset: pd.DataFrame = (
            surrogate_model.explained_classifier_model.data_set_data.getElement(
                "dataset"
            )
        )
        target_row: str = surrogate_model.explained_classifier_model.getElement(
            "target_row"
        )
        target_description = model_x.getElement("target_names_dict")
        class_names = [
            element["new_value"] for element in target_description["variables"]
        ]
        tg = ExplainSingleTree.graph_tree(
            x_train=dataset.drop(columns=target_row)[: int(len(dataset) / 2)],
            y_train=dataset[target_row][: int(len(dataset) / 2)],
            tree=model,
            class_names=class_names,
            feature_names=model.feature_names_in_,
        )
        return tg
