import json
from dash import dcc, html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from flask_login import current_user
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from app.API.utils import findTranslationsParent, setText, getDashboardTranslations
from app.processor.model.explainers.decision_tree_surrogate import (
    ExplainSingleTree,
)
from app.processor.models import ExplainedModel

id_sufix = ["trees"]


def specificTreesLayout(specificTreesTranslations):
    commonTranslations = findTranslationsParent(specificTreesTranslations, 'common')
    tableTranslations = findTranslationsParent(commonTranslations, 'table')
    tableTooltipTranslations = findTranslationsParent(tableTranslations, 'tooltip')
    treeTranslations = findTranslationsParent(commonTranslations, 'tree')
    treeTooltipTranslations = findTranslationsParent(treeTranslations, 'tooltip')

    layout = html.Div(
        [
            dcc.Loading(
                [
                    html.Div(
                        [
                            html.Plaintext(
                                setText(commonTranslations, 'depth', 'dashboard.trees.common'),
                                className="tree-creator-label",
                            ),
                            dbc.Input(
                                type="number",
                                id="s-max-depth-input-row",
                                min=0,
                                value=0,
                                className="tree-creator-input",
                            ),
                            html.Button(
                                setText(commonTranslations, 'build-btn', 'dashboard.trees.common'),
                                id="s-surrogate-tree-reconstruction-btn",
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
                                                        setText(tableTranslations, 'title',
                                                                'dashboard.trees.common.table'),
                                                        className="rules-title"
                                                    ),
                                                    html.I(
                                                        id=f"{id_sufix[0]}-info",
                                                        className="fa fa-info-circle info-icon",
                                                    ),
                                                    dbc.Tooltip(
                                                        [
                                                            html.Plaintext(
                                                                [
                                                                    setText(tableTooltipTranslations, 'text-1',
                                                                            'dashboard.trees.common.table.tooltip'),
                                                                    html.Strong(
                                                                        setText(tableTooltipTranslations, 'text-2',
                                                                                'dashboard.trees.common.table.tooltip')
                                                                    ),
                                                                    setText(tableTooltipTranslations, 'text-3',
                                                                            'dashboard.trees.common.table.tooltip'),
                                                                    html.Strong(
                                                                        setText(tableTooltipTranslations, 'text-4',
                                                                                'dashboard.trees.common.table.tooltip')
                                                                    ),
                                                                    setText(tableTooltipTranslations, 'text-5',
                                                                            'dashboard.trees.common.table.tooltip'),
                                                                    html.Strong(
                                                                        setText(tableTooltipTranslations, 'text-6',
                                                                                'dashboard.trees.common.table.tooltip')
                                                                    ),
                                                                    setText(tableTooltipTranslations, 'text-7',
                                                                            'dashboard.trees.common.table.tooltip'),
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
                                                id="s-rules-output-upload",
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
                                setText(treeTranslations, 'title',
                                        'dashboard.trees.common.tree'), className="rules-title"
                            ),
                            html.Button(
                                setText(treeTranslations, 'show-btn',
                                        'dashboard.trees.common.tree'),
                                id="s-build-tree-btn",
                                n_clicks=0,
                                className="tree-btn tree-creator-btn",
                                style={"margin-left": "3rem"},
                            ),
                            html.Button(
                                setText(treeTranslations, 'download-btn',
                                        'dashboard.trees.common.tree'),
                                id="s-download-tree-btn",
                                n_clicks=0,
                                hidden=True,
                                className="tree-btn tree-creator-btn",
                                style={"margin-left": "3rem"},
                            ),
                            dbc.Tooltip(
                                [
                                    html.Plaintext(
                                        [
                                            setText(treeTooltipTranslations, 'text-1',
                                                    'dashboard.trees.common.tree.tooltip'),
                                            html.Strong(
                                                html.A(
                                                    setText(treeTooltipTranslations, 'text-2',
                                                            'dashboard.trees.common.tree.tooltip'),
                                                    href=setText(treeTooltipTranslations, 'text-3',
                                                                 'dashboard.trees.common.tree.tooltip'),
                                                    target='_blank')),
                                            setText(treeTooltipTranslations, 'text-4',
                                                    'dashboard.trees.common.tree.tooltip'),
                                        ]
                                    ),
                                    html.Plaintext(
                                        [
                                            setText(treeTooltipTranslations, 'text-5',
                                                    'dashboard.trees.common.tree.tooltip'),
                                            html.Strong(
                                                setText(treeTooltipTranslations, 'text-6',
                                                        'dashboard.trees.common.tree.tooltip')
                                            ),
                                            setText(treeTooltipTranslations, 'text-7',
                                                    'dashboard.trees.common.tree.tooltip'),
                                        ]
                                    ),
                                ],
                                autohide=False,
                                className="personalized-tooltip",
                                target="s-build-tree-btn",
                            ),
                        ],
                        className="tree-creator container",
                        style={"padding-top": "20px", "justify-content": "flex-start"},
                    ),
                    dcc.Store(id="s-svg-holder", data={}),
                    html.Div(
                        [html.Img(id="s-tree-visual-output-upload")],
                        className="tree-img-container",
                    ),
                    dcc.Download(id='s-download-svg')
                ]
            )
        ],
        style={"margin": "auto"},
    )
    return layout


def specificTreesCallbacks(app, furl, isRegressor: bool = False):
    @app.callback(
        Output("s-rules-output-upload", "children"),
        Output("s-max-depth-input-row", "max"),
        Output("s-tree-visual-output-upload", "src"),
        Output("s-download-tree-btn", "hidden"),
        Input("s-surrogate-tree-reconstruction-btn", "n_clicks"),
        State("s-max-depth-input-row", "value"),
        Input("path", "href"),
    )
    def refresh_specificTrees_layout(_, tree_number, cl):
        f = furl(cl)
        model_id = f.args["model_id"]

        # TRANSLATIONS
        commonTranslations = getDashboardTranslations(current_user.langSelection, 'trees', 'common')
        tableTranslations = findTranslationsParent(commonTranslations, 'table')
        tableHeadersTranslations = findTranslationsParent(tableTranslations, 'headers')

        # NORMAL FLOW

        model_x: ExplainedModel = ExplainedModel.query.filter(
            ExplainedModel.id == model_id
        ).first()

        model: RandomForestClassifier | RandomForestRegressor = model_x.getElement("model")

        length = len(model.estimators_)

        rules = ExplainSingleTree.get_rules(
            tree_model=model.estimators_[tree_number].tree_,
            q_variables=[
                var["column_name"] for var in model_x.getElement("q_variables_dict")
            ],
            q_variables_values=model_x.getElement("q_variables_dict"),
            features=model.feature_names_in_,
            class_names=[
                var["new_value"]
                for var in model_x.explainer_classifier.getElement("target_names_dict")["variables"]
            ] if not isRegressor else None,
            model_type="Classifier" if not isRegressor else "Regressor"
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
                        html.Td([rule["target_value"],
                                 f' {model_x.explainer_regressor.getElement("unit")}' if isRegressor else '']),
                        html.Td(rule["probability"]),
                        html.Td(rule["samples_amount"]),
                    ]
                )
            )

        sub_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            setText(tableHeadersTranslations, 'predictor', 'dashboard.trees.common.table.headers'),
                            style={"width": "40%"}),
                        html.Th(
                            setText(tableHeadersTranslations, 'condition', 'dashboard.trees.common.table.headers'),
                            style={"width": "20%"}),
                        html.Th(
                            setText(tableHeadersTranslations, 'value', 'dashboard.trees.common.table.headers'),
                            style={"width": "40%"}),
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
                            html.Th(
                                setText(tableHeadersTranslations, 'rule', 'dashboard.trees.common.table.headers'),
                                rowSpan=2),
                            html.Th(setText(tableHeadersTranslations, 'causes',
                                            'dashboard.trees.common.table.headers')),
                            html.Th(
                                setText(tableHeadersTranslations, 'result', 'dashboard.trees.common.table.headers'),
                                colSpan=3),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Th(sub_header_table, style={"padding": "0"}),
                            html.Th(setText(tableHeadersTranslations, 'target',
                                            'dashboard.trees.common.table.headers')),
                            html.Th(
                                setText(tableHeadersTranslations, 'proba', 'dashboard.trees.common.table.headers')),
                            html.Th(setText(tableHeadersTranslations, 'samples',
                                            'dashboard.trees.common.table.headers')),
                        ]
                    ),
                ], style={"position": "sticky", "top": "0"}
            )
        ]

        table_body = [html.Tbody(rules_table)]

        rg = dbc.Table(
            table_header + table_body, bordered=True, className="rules-table"
        )
        return rg, length, "", True

    @app.callback(
        Output("s-tree-visual-output-upload", "src", allow_duplicate=True),
        Output("s-svg-holder", "data"),
        Output("s-download-tree-btn", "hidden", allow_duplicate=True),
        State("s-max-depth-input-row", "value"),
        State("path", "href"),
        Input("s-build-tree-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def build_img_tree(tree_number, cl, _):
        f = furl(cl)
        model_id = f.args["model_id"]

        model_x: ExplainedModel = ExplainedModel.query.filter(
            ExplainedModel.id == model_id
        ).first()

        model: RandomForestClassifier = model_x.getElement("model")

        model: DecisionTreeClassifier = model.estimators_[tree_number]
        dataset: pd.DataFrame = model_x.data_set_data.getElement("dataset")
        target_row: str = model_x.getElement("target_row")
        if not isRegressor:
            target_description = model_x.explainer_classifier.getElement("target_names_dict")
            class_names = [
                element["new_value"] for element in target_description["variables"]
            ]
        else:
            class_names = None
        x_train = dataset.drop(columns=target_row)
        tg, viz = ExplainSingleTree.graph_tree(
            x_train=x_train,
            y_train=dataset[target_row],
            tree=model,
            class_names=class_names,
            feature_names=x_train.columns,
        )

        # Convertir el árbol en un archivo SVG

        svg_str = viz.view().svg()
        # Devolver el archivo SVG como respuesta
        return tg, json.dumps(svg_str), False

    @app.callback(
        Output("s-download-svg", "data"),
        State("s-svg-holder", "data"),
        Input("s-download-tree-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_img_tree(data, download):
        if download:
            svg_str = json.loads(data)
            return dict(content=svg_str, filename='arbol.svg', type='text/svg+xml')
        raise PreventUpdate
