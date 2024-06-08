import multiprocessing
from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from app.proccessor.model.dataset_interaction_methods import update_y_pred
from app.proccessor.models import ModelForProccess

from sklearn import metrics
import plotly.express as px


def get_target_dropdown(values_dict):
    return [
        {"label": value["new_value"], "value": value["old_value"]}
        for value in values_dict
    ]


def get_y_test_transformed(y_test):
    for value in y_test:
        if value is not int:
            y_test = LabelEncoder().fit_transform(y_test)
            break
    return y_test


def __create_matrix(y_test, y_pred, class_names):
    # Generate the confusion matrix
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig = px.imshow(
        img=cm,
        title="MATRIZ DE CONFUSION",
        labels=dict(x="PREDICCIONES", y="VALORES REALES", color="CANTIDAD"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
    )
    return fig


def initialize_matrix(
    dropdown_value,
    slider_value,
    y_test,
    x_test,
    classifier_model: RandomForestClassifier,
    class_names,
):
    y_pred_new = classifier_model.predict(x_test)

    if dropdown_value is not None:
        positive_class = int(dropdown_value)
        probability_predictions = classifier_model.predict_proba(x_test)

        try:
            y_pred_new = update_y_pred(
                prediction=y_pred_new,
                probability_predictions=probability_predictions,
                cut_off=slider_value,
                positive_class=positive_class,
            )
        except Exception as e:
            print(e)

    return __create_matrix(y_test=y_test, y_pred=y_pred_new, class_names=class_names)


cutoff = dbc.Switch(
    label="USAR COTOFF",
    value=False,
    id="check-cutoff",
)

class_selector = dcc.Dropdown(
    value=None,
    id="positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    disabled=True,
)

slider = dcc.Slider(0.01, 0.99, 0.1, value=0.5, id="cutoff-slider", disabled=True)

metrics_layout = html.Div(
    [
        dcc.Loading(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row([html.Div(id="matrix-output-upload")]),
                                dbc.Row(
                                    [html.Div([cutoff], style={"padding-left": "20px"})]
                                ),
                                dbc.Row([class_selector]),
                                dbc.Row([slider], style={"padding-top": "20px"}),
                            ],
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    ["TEXTO CON EXPLICACION DEL GRÁFICO"],
                                    id="matrix-explanation",
                                )
                            ],
                            xs=4,
                            sm=4,
                            md=4,
                            lg=4,
                            xl=4,
                            xxl=4,
                        ),
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.Div(id="roc-output-upload"),
                                    ]
                                )
                            ],
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    ["TEXTO CON EXPLICACION DEL GRÁFICO"],
                                    id="roc-explanation",
                                )
                            ],
                            xs=4,
                            sm=4,
                            md=4,
                            lg=4,
                            xl=4,
                            xxl=4,
                        ),
                    ]
                ),
            ]
        )
    ],
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def metricsCallbacks(app, furl: Function):
    @app.callback(
        Output("matrix-output-upload", "children"),
        Output("cutoff-slider", "disabled"),
        Output("positive-class-selector", "disabled"),
        Output("positive-class-selector", "options"),
        Input("check-cutoff", "value"),
        Input("positive-class-selector", "value"),
        Input("cutoff-slider", "value"),
        Input("path", "href"),
    )
    def graph_explainers(cutoff, positive_class, slider, cl):
        f = furl(cl)
        param1 = f.args["model_id"]
        try:
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

            if positive_class or slider or cutoff:
                if cutoff and positive_class is not None:
                    return (
                        dcc.Graph(
                            figure=initialize_matrix(
                                dropdown_value=positive_class,
                                slider_value=slider,
                                y_test=classifier_dataset[model_x.target_row],
                                x_test=classifier_dataset.drop(
                                    columns=model_x.target_row
                                ),
                                classifier_model=classifier_model,
                                class_names=list(
                                    set(classifier_dataset[model_x.target_row])
                                ),
                            )
                        ),
                        False,
                        False,
                        get_target_dropdown(target_description["variables"]),
                    )
                else:
                    div = dcc.Graph(
                        figure=initialize_matrix(
                            dropdown_value=None,
                            slider_value=None,
                            y_test=classifier_dataset[model_x.target_row],
                            x_test=classifier_dataset.drop(columns=model_x.target_row),
                            classifier_model=classifier_model,
                            class_names=list(
                                set(classifier_dataset[model_x.target_row])
                            ),
                        )
                    )
                    if cutoff and positive_class is None:
                        return div, True, False, get_target_dropdown(target_description["variables"])
                    else:
                        return div, True, True, get_target_dropdown(target_description["variables"])
            else:
                return None, None, None, get_target_dropdown(target_description["variables"])
        except Exception as e:
            print(e)
            raise PreventUpdate
