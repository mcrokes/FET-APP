import json
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask_login import current_user
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Dashboard.metrics.metricsLayout import get_target_dropdown
from Dashboard.utils import findTranslationsParent, getTranslations, setText
from app.proccessor.models import ExplainedModel

from treeinterpreter import treeinterpreter as ti


def setBottomLegend(fig):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.5, xanchor="right", x=1)
    )
    return fig


def getTreeInterpreterParamethersRegressor(
        instance,
        instanceModified,
        model: RandomForestRegressor | DecisionTreeRegressor,
        commonTranslations
):
    contributionTranslations = findTranslationsParent(commonTranslations, 'contribution')
    contributionHeadersTranslations = findTranslationsParent(contributionTranslations, 'headers')

    prediction_label = setText(commonTranslations, 'prediction', 'dashboard.predictions.common')

    finalPrediction = setText(contributionTranslations, 'final-prediction', 'dashboard.predictions.common.contribution')
    actual = setText(contributionTranslations, 'actual', 'dashboard.predictions.common.contribution')
    accumulated = setText(contributionTranslations, 'accumulated', 'dashboard.predictions.common.contribution')
    populationMedia = setText(contributionTranslations, 'population-media', 'dashboard.predictions.common.contribution')
    media = setText(contributionTranslations, 'media', 'dashboard.predictions.common.contribution')

    instance_header = setText(contributionHeadersTranslations, 'instance',
                              'dashboard.predictions.common.contribution.headers')
    contribution_header = setText(contributionHeadersTranslations, 'contribution',
                                  'dashboard.predictions.common.contribution.headers')
    predictor_header = setText(contributionHeadersTranslations, 'predictor',
                               'dashboard.predictions.common.contribution.headers')
    value_header = setText(contributionHeadersTranslations, 'value',
                           'dashboard.predictions.common.contribution.headers')

    general_dict = {
        (instance_header, predictor_header): [],
        (instance_header, value_header): [],
    }
    contribution_graph_data = []

    prediction, bias, contributions = ti.predict(model, instance)
    predictions_graph_data = {
        "labels": None,
        "values": prediction[0],
    }

    point = 0

    general_dict[(contribution_header, contribution_header)] = []
    bar_base = bias[0]

    media_array_x = [accumulated]
    media_array_y = [bias[0]]
    sorted_contributions = sorted(zip(list(contributions[0]), instance)
                                  )
    contribution_graph_data.append({"graph_data": []})
    point += bar_base

    for jIndex, (contribution, feature) in enumerate(sorted_contributions):
        if feature not in general_dict[(instance_header, predictor_header)]:
            general_dict[(instance_header, predictor_header)].append(feature)
            general_dict[(instance_header, value_header)].append(instanceModified[feature])
        general_dict[(contribution_header, contribution_header)].append(
            f"{round(contribution, 3)}"
        )
        bar_base += contribution
        x = [accumulated]
        y = [contribution]

        contribution_graph_data[0]["graph_data"].append(
            go.Bar(
                name=f"{feature}({round(y[0], 3)}) {round(y[0] / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100, 2)}%",
                x=x,
                y=y,
            )
        )
        point += contribution

    general_dict[(contribution_header, contribution_header)].append(
        f"{round(bias[0], 3)}"
    )

    contribution_graph_data[0]["graph_data"].insert(
        0,
        go.Bar(
            name=f"{media} ({round(media_array_y[0], 3)}) {round(media_array_y[0] / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100, 3)}%",
            x=media_array_x,
            y=media_array_y,
        ),
    )
    contribution_graph_data[0]["graph_data"].append(
        go.Bar(
            name=f"{prediction_label} ({round(prediction[0][0], 3)}) 100%",
            x=[finalPrediction],
            y=[prediction[0][0]],
        )
    )
    contribution_graph_data[0]["graph_data"].append(
        go.Scatter(
            x=[accumulated, finalPrediction],
            y=[point, point],
            mode="lines",
            name=f"{actual} ({round(point, 3)}) {round((point / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100), 2)} %",
            line=dict(dash="dash"),
            marker_color=["blue", "blue"],
        )
    )

    general_dict[(instance_header, predictor_header)].append(populationMedia)
    general_dict[(instance_header, value_header)].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


def getTreeInterpreterParamethersClassifier(
        instance,
        instanceModified,
        model: RandomForestClassifier | DecisionTreeClassifier,
        class_names,
        current_class,
        commonTranslations
):
    contributionTranslations = findTranslationsParent(commonTranslations, 'contribution')
    contributionHeadersTranslations = findTranslationsParent(contributionTranslations, 'headers')

    prediction_label = setText(commonTranslations, 'prediction', 'dashboard.predictions.common')

    finalPrediction = setText(contributionTranslations, 'final-prediction', 'dashboard.predictions.common.contribution')
    actual = setText(contributionTranslations, 'actual', 'dashboard.predictions.common.contribution')
    accumulated = setText(contributionTranslations, 'accumulated', 'dashboard.predictions.common.contribution')
    populationMedia = setText(contributionTranslations, 'population-media', 'dashboard.predictions.common.contribution')
    media = setText(contributionTranslations, 'media', 'dashboard.predictions.common.contribution')

    instance_header = setText(contributionHeadersTranslations, 'instance',
                              'dashboard.predictions.common.contribution.headers')
    contribution_header = setText(contributionHeadersTranslations, 'contribution',
                                  'dashboard.predictions.common.contribution.headers')
    predictor_header = setText(contributionHeadersTranslations, 'predictor',
                               'dashboard.predictions.common.contribution.headers')
    value_header = setText(contributionHeadersTranslations, 'value',
                           'dashboard.predictions.common.contribution.headers')
    general_dict = {
        (instance_header, predictor_header): [],
        (instance_header, value_header): [],
    }
    contribution_graph_data = []

    prediction, bias, contributions = ti.predict(model, instance)
    predictions_graph_data = {
        "labels": class_names,
        "values": prediction[0],
    }

    point = 0

    for index, class_name in enumerate(class_names):
        general_dict[(contribution_header, class_name)] = []
        bar_base = bias[0][index]
        media_array_x = [accumulated]
        media_array_y = [bias[0][index]]
        sorted_contributions = sorted(
            zip(contributions[0], instance),
            key=lambda x: -max(x[0]),
        )
        if index == current_class:
            contribution_graph_data.append({"class_name": class_name, "graph_data": []})
            point += bar_base

        for jIndex, (contribution, feature) in enumerate(sorted_contributions):
            if feature not in general_dict[(instance_header, predictor_header)]:
                general_dict[(instance_header, predictor_header)].append(feature)
                general_dict[(instance_header, value_header)].append(instanceModified[feature])
            general_dict[(contribution_header, class_name)].append(
                f"{round(contribution[index], 3)} ({round(contribution[index] * 100, 1)}%)"
            )
            bar_base += contribution[index]
            if index == current_class:
                x = [accumulated]
                y = [contribution[index]]

                contribution_graph_data[0]["graph_data"].append(
                    go.Bar(
                        name=f"{feature}({round(y[0], 3)}) {round(y[0] / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100, 2)}%",
                        x=x,
                        y=y,
                    )
                )
                point += contribution[index]

        general_dict[(contribution_header, class_name)].append(
            f"{round(bias[0][index], 3)} ({round(bias[0][index] * 100, 1)}%)"
        )
        if index == current_class:
            contribution_graph_data[0]["graph_data"].insert(
                0,
                go.Bar(
                    name=f"{media} ({round(media_array_y[0], 3)}) {round(media_array_y[0] / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100, 3)}%",
                    x=media_array_x,
                    y=media_array_y,
                ),
            )
            contribution_graph_data[0]["graph_data"].append(
                go.Bar(
                    name=f"{prediction_label} ({round(prediction[0][index], 3)}) 100%",
                    x=[finalPrediction],
                    y=[prediction[0][index]],
                )
            )
            contribution_graph_data[0]["graph_data"].append(
                go.Scatter(
                    x=[accumulated, finalPrediction],
                    y=[point, point],
                    mode="lines",
                    name=f"{actual} ({round(point, 3)}) {round((point / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100), 2)} %",
                    line=dict(dash="dash"),
                    marker_color=["blue", "blue"],
                )
            )

    general_dict[(instance_header, predictor_header)].append(populationMedia)
    general_dict[(instance_header, value_header)].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


def getIndividualPredictionsRegressor(model: RandomForestRegressor, instance, treesTranslations):
    labelsTranslations = findTranslationsParent(treesTranslations, 'labels')
    figures = []
    individual_predictions = [
        estimator.predict(instance)[0] for estimator in model.estimators_
    ]

    x = list(range(len(individual_predictions) + 1))
    data = [
        go.Bar(
            name=setText(treesTranslations, 'bar-title', 'dashboard.predictions.regressor.trees'),
            y=np.round(sorted(individual_predictions), 2),
            x=x[1:],
        ),
        go.Scatter(
            name=setText(treesTranslations, 'line-title', 'dashboard.predictions.regressor.trees'),
            x=[1, x[-1]],
            y=[model.predict(instance)[0], model.predict(instance)[0]],
            mode="lines",
            line=dict(dash="dash"),
        )
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        title=setText(treesTranslations, 'title', 'dashboard.predictions.regressor.trees'),
        xaxis_title=setText(labelsTranslations, 'x', 'dashboard.predictions.regressor.trees.labels'),
        yaxis_title=setText(labelsTranslations, 'y', 'dashboard.predictions.regressor.trees.labels'),
        bargap=0.1,
    )
    figures.append({"graph_data": fig})

    return figures


def getIndividualPredictionsClassifier(model, class_names, instance, cut_point, current_class, treesTranslations):
    labelsTranslations = findTranslationsParent(treesTranslations, 'labels')
    figures = []
    index = current_class
    class_name = class_names[index]
    individual_predictions = [
        estimator.predict_proba(instance)[0] for estimator in model.estimators_
    ]

    markers = []
    for val in np.array(individual_predictions)[:, index]:
        if val * 100 >= cut_point:
            markers.append("blue")
        else:
            markers.append("red")

    sorted_predictions = sorted(
        zip(np.array(individual_predictions)[:, index], markers),
        key=lambda x: (-x[0] if markers.count("blue") > markers.count("red") else x[0]),
    )
    predictions_for_actual_clase = np.array(sorted_predictions)[:, 0]
    prev_x = list(range(len(predictions_for_actual_clase) + 1))
    x = (
        prev_x[1:]
        if markers.count("blue") > markers.count("red")
        else prev_x[::-1][:-1]
    )
    y = np.round(predictions_for_actual_clase.astype(np.float64) * 100, 2)

    data = [
        go.Bar(
            name=setText(treesTranslations, 'bar-title', 'dashboard.predictions.classifier.trees'),
            y=y,
            x=x,
            marker_color=np.array(sorted_predictions)[:, 1],
        ), go.Scatter(
            name=setText(treesTranslations, 'lines-title', 'dashboard.predictions.classifier.trees'),
            x=[1, x[0]] if markers.count("blue") < markers.count("red") else [x[-1], 1],
            y=[cut_point, cut_point],
            mode="lines",
            line=dict(dash="dash"),
        )
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"{setText(treesTranslations, 'title', 'dashboard.predictions.classifier.trees')}"
              f" {class_name}",
        xaxis_title=setText(labelsTranslations, 'x', 'dashboard.predictions.classifier.trees.labels'),
        yaxis_title=setText(labelsTranslations, 'y', 'dashboard.predictions.classifier.trees.labels'),
        bargap=0.1,
        xaxis=dict(
            autorange=(
                "reversed" if markers.count("blue") > markers.count("red") else True
            )
        ),
    )
    figures.append({"class_name": class_name, "graph_data": fig})

    return figures


class_selector = dcc.Dropdown(
    id="prediction-positive-class-selector",
    clearable=False,
    style={"cursor": "pointer"},
)

marks = {}
for n in range(11):
    if n == 0:
        marks["1"] = 1
    else:
        marks[f"{n * 10}"] = f"{n * 10}%"
slider = dcc.Slider(
    min=1, max=100, step=1, marks=marks, value=50, id="trees-cutoff-slider"
)

id_sufix = ["contributions", "trees-graph", "prototypes"]


def predictionsLayout(predictionsTranslations):
    commonTranslations = findTranslationsParent(predictionsTranslations, 'common')
    contributionTranslations = findTranslationsParent(commonTranslations, 'contribution')
    contributionTooltipTranslations = findTranslationsParent(contributionTranslations, 'tooltip')

    classifierTranslations = findTranslationsParent(predictionsTranslations, 'classifier')
    classifierSelectorTranslations = findTranslationsParent(classifierTranslations, 'class-selector')

    layout = html.Div(
        [
            dcc.Store(id="current-class-data"),
            dbc.Row(
                [
                    dbc.Col(
                        id="instances-dropdown",
                        xs=12,
                        sm=12,
                        md=5,
                        lg=5,
                        xl=5,
                        xxl=5,
                        style={'margin-bottom': '1rem'}
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="prototypes-dropdown",
                                placeholder=setText(classifierSelectorTranslations, 'placeholder',
                                                    'dashboard.predictions.classifier.class-selector'),
                                clearable=True,
                                style={"cursor": "pointer", "width": "250px", "margin-right": "1rem"},
                            ),
                            html.Div(
                                [
                                    html.I(
                                        id=f"{id_sufix[2]}-info",
                                        className="fa fa-info-circle info-icon",
                                    ),
                                    dbc.Tooltip(
                                        [
                                            html.Plaintext(
                                                [
                                                    setText(classifierSelectorTranslations, 'tooltip',
                                                            'dashboard.predictions.classifier.class-selector'),
                                                ]
                                            ),
                                        ],
                                        className="personalized-tooltip",
                                        target=f"{id_sufix[2]}-info",
                                        placement='right',
                                    ),
                                ],
                                id=f"{id_sufix[2]}-container",
                                style={"display": "flex", "justify-content": "end"},
                            )
                        ],
                        xs=12,
                        sm=12,
                        md=5,
                        lg=5,
                        xl=5,
                        xxl=5,
                        style={'display': 'flex'}

                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    html.H3(
                                        [
                                            html.Plaintext(
                                                id="predictions-title",
                                                style={
                                                    "color": "black",
                                                    "font-size": "18px",
                                                },
                                            ),
                                        ],
                                        style={"text-align": "center"},
                                    ),
                                    html.Div(id="predictions-view"),
                                ]
                            )
                        ],
                        style={"margin": "auto"},
                        xs=12,
                        sm=12,
                        md=7,
                        lg=7,
                        xl=7,
                        xxl=7,
                    ),
                    dbc.Col(
                        html.Div(id="predictions-output-upload"),
                        style={"margin": "auto"},
                        xs=12,
                        sm=12,
                        md=5,
                        lg=5,
                        xl=5,
                        xxl=5,
                    ),
                ],
                style={"padding-top": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Plaintext(
                                id="predictions-class_selector-title",
                                children=[setText(classifierSelectorTranslations, 'title',
                                                  'dashboard.predictions.classifier.class-selector')],
                                style={"color": "black"},
                            ),
                        ],
                        xs=12,
                        sm=12,
                        md=12,
                        lg=8,
                        xl=8,
                        xxl=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                id="predictions-class_selector-container",
                                children=[class_selector],
                            ),
                        ],
                        xs=12,
                        sm=12,
                        md=12,
                        lg=8,
                        xl=8,
                        xxl=8,
                    ),
                ],
                style={"padding-top": "20px"},
            ),
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
                                    setText(contributionTooltipTranslations, 'text-1',
                                            'dashboard.predictions.common.contribution.tooltip'),
                                    html.Strong(
                                        setText(contributionTooltipTranslations, 'text-2',
                                                'dashboard.predictions.common.contribution.tooltip')
                                    ),
                                    setText(contributionTooltipTranslations, 'text-3',
                                            'dashboard.predictions.common.contribution.tooltip'),
                                ]
                            ),
                        ],
                        className="personalized-tooltip",
                        target=f"{id_sufix[0]}-info",
                    ),
                ],
                id=f"{id_sufix[0]}-container",
                style={"display": "flex", "justify-content": "end"},
            ),
            html.Div(id="contributions-output-upload"),
            html.Div(id="test"),
            html.Div(
                [
                    html.I(
                        id=f"{id_sufix[1]}-info",
                        className="fa fa-info-circle info-icon",
                    ),
                    dbc.Tooltip(
                        id="trees-tooltip",
                        className="personalized-tooltip",
                        target=f"{id_sufix[1]}-info",

                    ),
                ],
                id=f"{id_sufix[1]}-container",
                style={"display": "flex", "justify-content": "end"},
            ),
            html.Div(id="trees-output-upload"),
            html.Div(slider, id="trees-slider-container", hidden=True),
        ],
        className="section-content container",
        style={"margin": "auto"},
    )
    return layout


def predictionsCallbacks(app, furl, isRegressor: bool = False):
    @app.callback(
        Output("test_graph", "figure"),
        State("current-class-data", "data"),
        State("test_graph", "figure"),
        Input("test_graph", "restyleData"),
        prevent_initial_call=True,
    )
    def update_point(data, figure, restyleData):
        if restyleData and (
                restyleData[1][-1] != len(figure["data"]) - 1
                or restyleData[0]["visible"][restyleData[1][-1]]
        ):

            # TRANSLATIONS
            commonTranslations = getTranslations(current_user.langSelection, 'predictions', 'common')
            contributionTranslations = findTranslationsParent(commonTranslations, 'contribution')

            # NORMAL FLOW
            prediction_upper = contributionTranslations.get('prediction-upper') if contributionTranslations.get(
                'prediction-upper') else f"dashboard.predictions.common.prediction "

            actual_upper = contributionTranslations.get('actual-upper') if contributionTranslations.get(
                'actual-upper') else f"dashboard.predictions.common.contribution.actual "

            def isDisabled(inner_data):
                print('inner-data: ', inner_data['name'])
                print('prediction_upper: ', prediction_upper)
                print('actual_upper: ', actual_upper)
                if inner_data["name"].upper().find(prediction_upper.upper()) == 0:
                    return True
                elif inner_data["name"].upper().find(actual_upper.upper(), 0) == 0:
                    figure["data"].remove(inner_data)
                    return True
                elif inner_data.get("visible") == "legendonly":
                    return True
                for data in figure['data']:
                    print()
                    print("data: ", data['name'])

                return False

            point = sum(
                data["y"][0] if not isDisabled(data) else 0 for data in figure["data"]
            )
            data = json.loads(data)
            trace = go.Scatter(
                x=[
                    setText(contributionTranslations, 'accumulated',
                            'dashboard.predictions.common.contribution'),
                    setText(contributionTranslations, 'final-prediction',
                            'dashboard.predictions.common.contribution')
                ],
                y=[point, point],
                mode="lines",
                name=f"{setText(contributionTranslations, 'actual', 'dashboard.predictions.common.contribution')}"
                     f" ({round(point, 3)}) {round(point / (data['prediction'] if data['prediction'] > 0 or isRegressor else 1) * 100, 2)}%",
                line=dict(dash="dash"),
            )
            figure["data"].append(trace)
            return figure
        else:
            raise PreventUpdate

    def getIndexesDropOptions(model_x, x_test):
        options = []
        indexes_list = model_x.getElement('indexesList') if model_x.indexesList else None
        print('indexes_list: ', indexes_list is not None)
        for index, _ in x_test.iterrows():
            options.append({"label": index if indexes_list is None else indexes_list[index], "value": index})

        return options

    @app.callback(
        Output("instances-dropdown", "children", allow_duplicate=True),
        Output("trees-cutoff-slider", "value"),
        Output("prediction-positive-class-selector", "options"),
        Output("prediction-positive-class-selector", "value"),
        Output("trees-tooltip", "children"),
        Input("path", "href"),
        prevent_initial_call=True
    )
    def load_init_data(cl):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            # TRANSLATIONS
            commonTranslations = getTranslations(current_user.langSelection, 'predictions', 'common')

            classifierTranslations = getTranslations(current_user.langSelection, 'predictions', 'classifier')
            classifierTreesTranslations = findTranslationsParent(classifierTranslations, 'trees')
            classifierTreesTooltipTranslations = findTranslationsParent(classifierTreesTranslations, 'tooltip')

            regressorTranslations = getTranslations(current_user.langSelection, 'predictions', 'regressor')
            regressorTreesTranslations = findTranslationsParent(regressorTranslations, 'trees')
            regressorTreesTooltipTranslations = findTranslationsParent(regressorTreesTranslations, 'tooltip')

            # NORMAL FLOW

            model_x: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == model_id
            ).first()

            ds = model_x.data_set_data.getElement("dataset")
            x_test = ds.drop(columns=model_x.getElement("target_row"))
            if isRegressor:
                class_names = None
                target_dropdown = [{'value': 0}]
            else:
                target_description = model_x.explainer_classifier.getElement("target_names_dict")
                class_names = [var["new_value"] for var in target_description["variables"]]
                target_dropdown = get_target_dropdown(target_description["variables"])

            drop_down = dcc.Dropdown(
                id="select",
                placeholder=setText(commonTranslations, 'instance-selector-placeholder',
                                    'dashboard.predictions.common'),
                options=getIndexesDropOptions(model_x, x_test),
            )

            slider_initial_value = 100 / len(class_names) + 1 if not isRegressor else 50

            if isRegressor:
                trees_tooltip = [
                    html.Plaintext(
                        [
                            html.Strong(
                                setText(regressorTreesTooltipTranslations, 'text-1',
                                        'dashboard.predictions.regressor.trees.tooltip')
                            ),
                            setText(regressorTreesTooltipTranslations, 'text-2',
                                    'dashboard.predictions.regressor.trees.tooltip'),
                            html.Strong(
                                setText(regressorTreesTooltipTranslations, 'text-3',
                                        'dashboard.predictions.regressor.trees.tooltip')
                            ),
                            setText(regressorTreesTooltipTranslations, 'text-4',
                                    'dashboard.predictions.regressor.trees.tooltip'),
                        ]
                    ),
                    html.Plaintext(
                        [
                            setText(regressorTreesTooltipTranslations, 'text-5',
                                    'dashboard.predictions.regressor.trees.tooltip'),
                        ]
                    ),
                ]
            else:
                trees_tooltip = [
                    html.Plaintext(
                        [
                            html.Strong(
                                setText(classifierTreesTooltipTranslations, 'text-1',
                                        'dashboard.predictions.classifier.trees.tooltip')
                            ),
                            setText(classifierTreesTooltipTranslations, 'text-2',
                                    'dashboard.predictions.classifier.trees.tooltip'),
                            html.Strong(
                                setText(classifierTreesTooltipTranslations, 'text-3',
                                        'dashboard.predictions.classifier.trees.tooltip')
                            ),
                            setText(classifierTreesTooltipTranslations, 'text-4',
                                    'dashboard.predictions.classifier.trees.tooltip'),
                        ]
                    ),
                    html.Plaintext(
                        [
                            html.Strong(
                                setText(classifierTreesTooltipTranslations, 'text-5',
                                        'dashboard.predictions.classifier.trees.tooltip')
                            ),
                            setText(classifierTreesTooltipTranslations, 'text-6',
                                    'dashboard.predictions.classifier.trees.tooltip'),
                        ]
                    ),
                    html.Plaintext(
                        [
                            html.Strong(
                                setText(classifierTreesTooltipTranslations, 'text-7',
                                        'dashboard.predictions.classifier.trees.tooltip')
                            ),
                            setText(classifierTreesTooltipTranslations, 'text-8',
                                    'dashboard.predictions.classifier.trees.tooltip'),
                        ]
                    ),
                ]

            return (
                drop_down,
                slider_initial_value,
                target_dropdown,
                target_dropdown[0]["value"],
                trees_tooltip,
            )
        except Exception as e:
            print(e)
            raise PreventUpdate

    @app.callback(
        Output("trees-output-upload", "children"),
        State("path", "href"),
        Input("select", "value"),
        Input("trees-cutoff-slider", "value"),
        Input("prediction-positive-class-selector", "value"),
    )
    def graph_trees_predictions(cl, n, cut_point, positive_class):
        if n or n == 0:
            # TRANSLATIONS
            classifierTranslations = getTranslations(current_user.langSelection, 'predictions', 'classifier')
            classifierTreesTranslations = findTranslationsParent(classifierTranslations, 'trees')

            regressorTranslations = getTranslations(current_user.langSelection, 'predictions', 'regressor')
            regressorTreesTranslations = findTranslationsParent(regressorTranslations, 'trees')

            # NORMAL FLOW

            n = int(n) + 1
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                if isRegressor:
                    class_names = None
                else:
                    target_description = model_x.explainer_classifier.getElement("target_names_dict")
                    class_names = [var["new_value"] for var in target_description["variables"]]

                ds = model_x.data_set_data.getElement("dataset")
                x_test = ds.drop(columns=model_x.getElement("target_row"))
                instance: pd.DataFrame = (
                    x_test[n - 1: n] if 0 <= n <= len(x_test) else x_test[-1:]
                )

                if not isRegressor:
                    individual_predictions_graph = getIndividualPredictionsClassifier(
                        model=model_x.getElement("model"),
                        class_names=class_names,
                        instance=instance,
                        cut_point=cut_point,
                        current_class=positive_class,
                        treesTranslations=classifierTreesTranslations
                    )
                else:
                    individual_predictions_graph = getIndividualPredictionsRegressor(
                        model=model_x.getElement("model"),
                        instance=instance,
                        treesTranslations=regressorTreesTranslations
                    )

                return [
                    html.Div(
                        id=f"contribution_graph_{data['class_name'] if class_names is not None else 'regressor'}",
                        children=dcc.Graph(
                            figure=setBottomLegend(
                                go.Figure(
                                    data["graph_data"], layout=dict(barmode="stack")
                                )
                            )
                        ),
                    )
                    for data in individual_predictions_graph
                ]

            except Exception as e:
                print(e)
                raise PreventUpdate
        else:
            return []

    @app.callback(
        Output("predictions-title", "children"),
        Output("predictions-view", "children"),
        Output("contributions-output-upload", "children"),
        Output("predictions-output-upload", "children"),
        Output("trees-slider-container", "hidden"),
        Output("predictions-class_selector-container", "hidden"),
        Output("predictions-class_selector-title", "hidden"),
        Output(f"{id_sufix[0]}-container", "hidden"),
        Output(f"{id_sufix[1]}-container", "hidden"),
        Output("current-class-data", "data"),
        State("path", "href"),
        Input("select", "value"),
        Input("prediction-positive-class-selector", "value"),
    )
    def graph_explainers(cl, n, positive_class):
        if n or n == 0:
            # TRANSLATIONS
            commonTranslations = getTranslations(current_user.langSelection, 'predictions', 'common')
            contributionTranslations = findTranslationsParent(commonTranslations, 'contribution')
            contributionHeadersTranslations = findTranslationsParent(contributionTranslations, 'headers')

            classifierTranslations = getTranslations(current_user.langSelection, 'predictions', 'classifier')
            classifierContributionTranslations = findTranslationsParent(classifierTranslations, 'contribution')

            regressorTranslations = getTranslations(current_user.langSelection, 'predictions', 'regressor')
            regressorContributionTranslations = findTranslationsParent(regressorTranslations, 'contribution')

            # NORMAL FLOW
            n = int(n) + 1
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                if isRegressor:
                    class_names = None
                else:
                    target_description = model_x.explainer_classifier.getElement("target_names_dict")
                    class_names = [var["new_value"] for var in target_description["variables"]]

                ds = model_x.data_set_data.getElement("dataset")
                dsModified = model_x.data_set_data.getElement("dataset_modified")
                x_test = ds.drop(columns=model_x.getElement("target_row"))
                x_testModified = dsModified.drop(
                    columns=model_x.getElement("target_row")
                )

                instance: pd.DataFrame = (
                    x_test[n - 1: n] if 1 <= n <= len(x_test) else x_test[-1:]
                )

                instanceModified: pd.DataFrame = (
                    x_testModified[n - 1: n]
                    if 0 <= n <= len(x_test)
                    else x_testModified[-1:]
                )

                if isRegressor:
                    contribution_graph_data, general_dict, predictions_graph_data = (
                        getTreeInterpreterParamethersRegressor(
                            model=model_x.getElement("model"),
                            instance=instance,
                            instanceModified=instanceModified,
                            commonTranslations=commonTranslations
                        ))
                else:
                    contribution_graph_data, general_dict, predictions_graph_data = (
                        getTreeInterpreterParamethersClassifier(
                            current_class=positive_class,
                            instance=instance,
                            instanceModified=instanceModified,
                            class_names=class_names,
                            model=model_x.getElement("model"),
                            commonTranslations=commonTranslations
                        )
                    )
                df = pd.DataFrame(general_dict)
                dtt = (
                        setText(contributionTranslations, 'title', 'dashboard.predictions.common.contribution')
                        + ("" if isRegressor else setText(classifierContributionTranslations, 'sub-title',
                                                          'dashboard.predictions.classifier.contribution'))
                )
                pie_chart = go.Figure(
                    data=[
                        go.Pie(
                            labels=predictions_graph_data["labels"],
                            values=predictions_graph_data["values"],
                            hole=0.3,
                        )
                    ]
                )

                individual_title = setText(contributionTranslations, 'individual-title',
                                           'dashboard.predictions.common.contribution')
                individual_sub_title = setText(classifierContributionTranslations, "individual-sub-title",
                                               "dashboard.predictions.classifier.contribution")

                def getFigure(fig):
                    fig.update_layout(
                        title=f"{individual_title}{f'{individual_sub_title}{class_names[positive_class]}' if not isRegressor else ''}",
                        xaxis_title=setText(contributionTranslations, 'label-x',
                                            'dashboard.predictions.common.contribution'),
                        yaxis_title=(
                            setText(regressorContributionTranslations, 'label-y',
                                    'dashboard.predictions.regressor.contribution') if isRegressor else setText(
                                classifierContributionTranslations, 'label-y',
                                'dashboard.predictions.classifier.contribution')
                        )
                    )
                    return setBottomLegend(fig)

                contribution_header = setText(commonTranslations, 'contribution',
                                              'dashboard.predictions.common.contribution.headers')

                data_table = html.Div(
                    [
                        dash_table.DataTable(
                            data=[
                                {
                                    **{f"{x1}_{x2}": y for (x1, x2), y in data},
                                }
                                for (n, data) in [
                                    *enumerate(
                                        [
                                            list(x.items())
                                            for x in df.T.to_dict().values()
                                        ]
                                    )
                                ]
                            ],
                            columns=[
                                {"name": [i, j], "id": f"{i}_{j}"} if not isRegressor or i != contribution_header else {
                                    "name": [i], "id": f"{i}_{j}", "rowspan": 2}
                                for i, j in df.columns
                            ],
                            page_size=10,
                            merge_duplicate_headers=True,
                            style_header={
                                "font-size": "16px",
                                "font-weight": "bold",
                                "text-align": "center",
                                "color": "black",
                            },
                            style_header_conditional=[
                                {
                                    'if': {
                                        'header_index': 1,
                                        'column_id': f'{contribution_header}_{contribution_header}'
                                    },
                                    'backgroundColor': '#0010ff4d'
                                },
                            ],
                            style_data={
                                "whiteSpace": "normal",
                                "font-size": "14px",
                                "text-align": "center",
                                "color": "black",
                            },
                            fill_width=True,
                            style_table={"overflow": "scroll"},
                        ),
                    ]
                ),
                contributions_graph = [
                    html.Div(
                        id=f"contribution_graph_{data['class_name'] if class_names is not None else 'regressor'}",
                        children=dcc.Graph(
                            id="test_graph",
                            figure=getFigure(
                                go.Figure(
                                    data["graph_data"], layout=dict(barmode="stack")
                                )
                            ),
                            config={"scrollZoom": True},
                        ),
                    )
                    for data in contribution_graph_data
                ]

                return (
                    dtt,
                    data_table,
                    contributions_graph,
                    dcc.Graph(figure=pie_chart) if not isRegressor else
                    html.Plaintext(
                        f"{setText(commonTranslations, 'prediction', 'dashboard.predictions.common')}"
                        f": {predictions_graph_data['values'][0]}",
                        style={
                            "color": "black",
                            "font-size": "18px",
                        },
                    ),
                    False if not isRegressor else True,
                    False if not isRegressor else True,
                    False if not isRegressor else True,
                    False,
                    False,
                    json.dumps(
                        {
                            "prediction": predictions_graph_data["values"][positive_class] if not isRegressor else
                            predictions_graph_data['values'][0]}
                    ),
                )
            except Exception as e:
                print('error in graph_explainers: ', e)
                raise PreventUpdate

        else:
            return [], [], [], [], True, True, True, True, True, None

    if not isRegressor:
        @app.callback(
            Output("prototypes-dropdown", "options"),
            Output("prototypes-dropdown", "value"),
            Input("path", "href"),
        )
        def prototypes(cl):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:

                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                ds = model_x.data_set_data.getElement("dataset")
                iterrows = list(set(ds[model_x.getElement("target_row")]))
                options = []
                for index in iterrows:
                    print()
                    options.append({"label": index, "value": index})

                return (
                    options,
                    None,
                )

            except Exception as e:
                print(e)
                raise PreventUpdate

        @app.callback(
            Output("instances-dropdown", "children", allow_duplicate=True),
            State("path", "href"),
            Input("prototypes-dropdown", "value"),
            prevent_initial_call=True,
        )
        def load_data(cl, class_value):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                commonTranslations = getTranslations(current_user.langSelection, 'predictions', 'common')
                placeholder = setText(commonTranslations, 'instance-selector-placeholder',
                                      'dashboard.predictions.common')

                # NORMAL FLOW
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                ds = model_x.data_set_data.getElement("dataset")
                if class_value is not None:
                    x_test = ds[ds[model_x.getElement("target_row")] == class_value].drop(
                        columns=model_x.getElement("target_row"))
                    options = []
                    indexes_list = model_x.getElement('indexesList') if model_x.indexesList else None
                    for index, _ in x_test.iterrows():
                        options.append(
                            {"label": index if indexes_list is None else indexes_list[index], "value": index})

                    drop_down = dcc.Dropdown(
                        id="select",
                        placeholder=placeholder,
                        options=options,
                    )

                    return (
                        drop_down,
                    )

                else:
                    x_test = ds.drop(columns=model_x.getElement("target_row"))
                    drop_down = dcc.Dropdown(
                        id="select",
                        placeholder=placeholder,
                        options=getIndexesDropOptions(model_x, x_test),
                    )
                    return drop_down
            except Exception as e:
                print(e)

            raise PreventUpdate


    else:
        @app.callback(
            Output("prototypes-dropdown", "style"),
            Output(f"{id_sufix[2]}-container", "style"),
            Input("path", "href"),
        )
        def hide_classifier_stuffs(cl):
            return {'display': 'none'}, {'display': 'none'}
