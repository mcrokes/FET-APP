import math
import multiprocessing
from pyclbr import Function
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from app.proccessor.model.values import get_target_dropdown
from app.proccessor.models import ExplainedClassifierModel, ModelForProccess


class_selector = dcc.Dropdown(
    id="importances-permut-positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
)

importancesLayout = html.Div(
    [
        dbc.Row(
            html.Div(id="importance-output-upload"),
            style={"padding-top": "20px"},
        ),
        dbc.Row(
            html.Div(id="permutation-importance-output-upload"),
            style={"padding-top": "20px"},
        ),
        dbc.Row([class_selector]),
    ],
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def importancesCallbacks(app, furl: Function):
    @app.callback(
        Output("importance-output-upload", "children"),
        Output("permutation-importance-output-upload", "children"),
        Output("importances-permut-positive-class-selector", "options"),
        Input("path", "href"),
        Input("importances-permut-positive-class-selector", "value"),
    )
    def graph_explainers(cl, positive_class):
        f = furl(cl)
        param1 = f.args["model_id"]
        try:
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == param1
            ).first()

            classifier_model: RandomForestClassifier = model_x.getElement("model")
            classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement(
                "dataset"
            )
            target_description = model_x.getElement("target_names_dict")
            df_feature_importance: pd.DataFrame = pd.DataFrame(
                {
                    "Predictor": classifier_model.feature_names_in_,
                    "Importancia": classifier_model.feature_importances_,
                }
            )
            importances_fig = px.bar(
                data_frame=df_feature_importance.sort_values(
                    "Importancia", ascending=False
                ),
                x="Importancia",
                y="Predictor",
                title="Importance GINI",
            )
            positive_class = (
                int(positive_class)
                if positive_class is not None
                else classifier_dataset[model_x.target_row]
            )

            permutation_importance_model = permutation_importance(
                estimator=classifier_model,
                X=classifier_dataset.drop(columns=model_x.target_row)[
                    classifier_dataset[model_x.target_row] == positive_class
                ],
                y=classifier_dataset[
                    classifier_dataset[model_x.target_row] == positive_class
                ][model_x.target_row],
                n_repeats=5,
                scoring="neg_root_mean_squared_error",
                n_jobs=multiprocessing.cpu_count() - 1,
                random_state=123,
            )
            df_permutation_importance = pd.DataFrame(
                {
                    k: permutation_importance_model[k]
                    for k in ["importances_mean", "importances_std"]
                }
            )
            df_permutation_importance["Predictor"] = classifier_model.feature_names_in_
            df_ordered_importance = df_permutation_importance.sort_values(
                "importances_mean", ascending=True
            )
            permutation_fig = px.bar(
                data_frame=df_ordered_importance,
                error_x=df_ordered_importance["importances_std"],
                x="importances_mean",
                y="Predictor",
                title="IMPORTANCIAS POR PERMUTACION",
                labels={"importances_mean": "Importancia +- error: "},
            )
            return (
                dcc.Graph(figure=importances_fig),
                dcc.Graph(figure=permutation_fig),
                get_target_dropdown(target_description["variables"]),
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
