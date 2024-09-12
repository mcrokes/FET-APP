from logging import exception
import multiprocessing
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
from flask_login import current_user
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from app.API.utils import findTranslationsParent, setText, getDashboardTranslations
from app.proccessor.model.values import get_target_dropdown
from app.proccessor.models import ExplainedModel

id_sufix = ["importances", "permutation-importances"]


def importancesLayout(importanceTranslations):
    commonTranslations = findTranslationsParent(importanceTranslations, 'common')
    classifierTranslations = findTranslationsParent(importanceTranslations, 'classifier')
    giniTranslations = findTranslationsParent(commonTranslations, 'gini')
    giniTooltipTranslations = findTranslationsParent(giniTranslations, 'tooltip')
    permutationTranslations = findTranslationsParent(commonTranslations, 'permutation')
    permutationTooltipTranslations = findTranslationsParent(permutationTranslations, 'tooltip')

    layout = html.Div(
        dcc.Loading(
            [
                html.Div([
                    html.Div(
                        [
                            html.I(
                                id=f"{id_sufix[0]}-info",
                                className="fa fa-info-circle info-icon",
                            ),
                            dbc.Tooltip(
                                [
                                    html.Plaintext(
                                        [
                                            setText(giniTooltipTranslations, 'text-1',
                                                    'dashboard.importance.common.gini.tooltip'),
                                            html.Strong(setText(giniTooltipTranslations, 'text-2',
                                                                'dashboard.importance.common.gini.tooltip')),
                                            setText(giniTooltipTranslations, 'text-3',
                                                    'dashboard.importance.common.gini.tooltip'),
                                        ]
                                    ),
                                ],
                                className="personalized-tooltip",
                                target=f"{id_sufix[0]}-info",
                            ),
                        ],
                        style={"display": "flex", "justify-content": "end"},
                    ),
                    dbc.Row(
                        html.Div(id="importance-output-upload"),
                        style={"padding-top": "20px"},
                    ),
                    html.Div(
                        [
                            html.I(
                                id=f"{id_sufix[1]}-info",
                                className="fa fa-info-circle info-icon",
                            ),
                            dbc.Tooltip(
                                [
                                    html.Plaintext(
                                        [
                                            setText(permutationTooltipTranslations, 'text-1',
                                                    'dashboard.importance.common.permutation.tooltip'),
                                            setText(permutationTooltipTranslations, 'text-2',
                                                    'dashboard.importance.common.permutation.tooltip'),
                                            html.Strong(
                                                setText(permutationTooltipTranslations, 'text-3',
                                                        'dashboard.importance.common.permutation.tooltip')),
                                            setText(permutationTooltipTranslations, 'text-4',
                                                    'dashboard.importance.common.permutation.tooltip'),
                                        ]
                                    ),
                                    html.Plaintext(id="extra-hint"),
                                ],
                                className="personalized-tooltip",
                                target=f"{id_sufix[1]}-info",
                            ),
                        ],
                        style={"display": "flex", "justify-content": "end"},
                    ),
                    dbc.Row(
                        html.Div(id="permutation-importance-output-upload"),
                        style={"padding-top": "20px"},
                    ),
                    dbc.Row(html.Div([
                        dcc.Dropdown(
                            id="importances-permut-positive-class-selector",
                            placeholder=setText(classifierTranslations, 'selector-placeholder',
                                                'dashboard.importance.classifier'),
                        )
                    ], id='selector-container', hidden=True)),
                ], className="container")
            ],
        ),
        style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
    )

    return layout


def importancesCallbacks(app, furl, isRegressor: bool = False):
    @app.callback(
        Output("importance-output-upload", "children"),
        Output("permutation-importance-output-upload", "children"),
        Output("importances-permut-positive-class-selector", "options"),
        Output("selector-container", "hidden"),
        Output("extra-hint", "children"),
        Input("path", "href"),
        Input("importances-permut-positive-class-selector", "value"),
    )
    def graph_explainers(cl, positive_class):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            # TRANSLATIONS
            translationsCommon = getDashboardTranslations(current_user.langSelection, 'importance', 'common')
            translationsClassifier = getDashboardTranslations(current_user.langSelection, 'importance', 'classifier')
            translationsGini = findTranslationsParent(translationsCommon, 'gini')
            translationsPermutation = findTranslationsParent(translationsCommon, 'permutation')

            # NORMAL FLOW
            model_x: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == model_id
            ).first()

            # NORMAL FLOW
            model: RandomForestClassifier | RandomForestRegressor = model_x.getElement("model")
            dataset: pd.DataFrame = model_x.data_set_data.getElement("dataset")

            df_feature_importance: pd.DataFrame = pd.DataFrame(
                {
                    setText(translationsGini, 'label-y', 'dashboard.importance.common.gini'): model.feature_names_in_,
                    setText(translationsGini, 'label-x',
                            'dashboard.importance.common.gini'): model.feature_importances_,
                }
            )
            print(df_feature_importance)
            importances_fig = px.bar(
                data_frame=df_feature_importance.sort_values(
                    setText(translationsGini, 'label-x',
                            'dashboard.importance.common.gini'), ascending=False
                ),
                x=setText(translationsGini, 'label-x', 'dashboard.importance.common.gini'),
                y=setText(translationsGini, 'label-y', 'dashboard.importance.common.gini'),
                title=setText(translationsGini, 'title', 'dashboard.importance.common.gini'),
            )

            if not isRegressor:
                classifer_model = model_x.explainer_classifier
                target_description = classifer_model.getElement("target_names_dict")
                old_class_names = [
                    element["old_value"] for element in target_description["variables"]
                ]
                try:
                    positive_class = old_class_names[int(positive_class)]
                except Exception as e:
                    str(e)
                    if positive_class is None:
                        positive_class = dataset[model_x.target_row]

                y = dataset[
                    dataset[model_x.target_row] == positive_class
                    ][model_x.target_row]
            else:
                y = dataset[model_x.target_row]

            permutation_importance_model = permutation_importance(
                estimator=model,
                X=dataset.drop(columns=model_x.target_row)[
                    dataset[model_x.target_row] == positive_class
                    ] if not isRegressor else dataset.drop(columns=model_x.target_row),
                y=y,
                n_repeats=5,
                scoring="accuracy" if not isRegressor else "neg_root_mean_squared_error",
                n_jobs=multiprocessing.cpu_count() - 1,
                random_state=123,
            )
            df_permutation_importance = pd.DataFrame(
                {
                    k: permutation_importance_model[k]
                    for k in ["importances_mean", "importances_std"]
                }
            )
            df_permutation_importance[
                setText(translationsPermutation, 'label-y', 'dashboard.importance.common.permutation')
            ] = (
                model.feature_names_in_)
            df_ordered_importance = df_permutation_importance.sort_values(
                "importances_mean", ascending=True
            )
            permutation_fig = px.bar(
                data_frame=df_ordered_importance,
                error_x=df_ordered_importance["importances_std"],
                x="importances_mean",
                y=setText(translationsPermutation, 'label-y', 'dashboard.importance.common.permutation'),
                title=setText(translationsPermutation, 'title', 'dashboard.importance.common.permutation'),
                labels={
                    "importances_mean":
                        setText(translationsPermutation, 'label-x', 'dashboard.importance.common.permutation')
                },
            )
            classifier_hint = html.Strong(setText(translationsClassifier, 'hint', 'dashboard.importance.classifier'))
            return (
                dcc.Graph(figure=importances_fig),
                dcc.Graph(figure=permutation_fig),
                get_target_dropdown(target_description["variables"]) if not isRegressor else [],
                True if isRegressor else False,
                '' if isRegressor else classifier_hint,
            )
        except Exception as e:
            print(e)
            exception(e)
            raise PreventUpdate
