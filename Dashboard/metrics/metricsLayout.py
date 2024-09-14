import math

import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from flask_login import current_user
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Dashboard.utils import get_target_dropdown
from app.API.utils import setText, findTranslationsParent, getDashboardTranslations
from app.proccessor.model.dataset_interaction_methods import (
    get_y_transformed,
    update_y_pred,
)
from app.proccessor.models import ExplainedClassifierModel, ExplainedModel

from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go

id_sufix = ["confusion-matrix", "roc-curve", "pred-real", "real-pred"]


# REGRESSOR FUNCTIONS
def generate_regression_metrics(y, y_pred, keys, parametersTranslations):
    mse = mean_squared_error(y, y_pred)
    regression_metrics = {
        f"{keys['mse']}": round(mse, 4),
        f"{keys['rmse']}": round(np.sqrt(mse), 4),
        f"{keys['mae']}": round(mean_absolute_error(y, y_pred), 4),
        f"{keys['mape']}": round(np.mean(np.abs((y - y_pred) / y)) * 100, 4),
        f"{keys['r2']}": round(r2_score(y, y_pred), 4),
    }

    descriptions = {
        f"{keys['mse']}": setText(parametersTranslations, 'mse-tooltip', 'dashboard.metrics.regressor.parameters'),
        f"{keys['rmse']}": setText(parametersTranslations, 'rmse-tooltip', 'dashboard.metrics.regressor.parameters'),
        f"{keys['mae']}": setText(parametersTranslations, 'mae-tooltip', 'dashboard.metrics.regressor.parameters'),
        f"{keys['mape']}": setText(parametersTranslations, 'mape-tooltip', 'dashboard.metrics.regressor.parameters'),
        f"{keys['r2']}": setText(parametersTranslations, 'r2-tooltip', 'dashboard.metrics.regressor.parameters'),
    }

    regression_metrics_df = (
        pd.DataFrame(regression_metrics,
                     index=[setText(parametersTranslations, 'value', 'dashboard.metrics.regressor.parameters')])
        .transpose()[0:]
        .rename_axis(setText(parametersTranslations, 'metrics', 'dashboard.metrics.regressor.parameters'))
        .reset_index()
    )

    regression_metrics_table = dbc.Table(
        [
            html.Thead(
                html.Tr([html.Th(col) for col in regression_metrics_df.columns])
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                [
                                    html.Span(cell, id=f"{cell}-{row}-{index}"),
                                    dbc.Tooltip(
                                        f"{descriptions[cell]}",
                                        target=f"{cell}-{row}-{index}",
                                    ),
                                ]
                                if index == 0
                                else cell,
                                style={'white-space': 'nowrap', 'padding': '0.5rem'}
                            )
                            for index, (col, cell) in enumerate(zip(regression_metrics_df.columns, row))
                        ]
                    )
                    for row in regression_metrics_df.values.tolist()
                ]
            ),
        ],
        className="rules-table",
        bordered=True,
        hover=True,
    )

    return regression_metrics_table


def addAxisNames(fig, axisTranslations):
    fig = setBottomLegend(fig)
    fig.update_layout(
        yaxis_title=setText(axisTranslations, 'y', 'dashboard.metrics.regressor.partial-dependence.labels'),
        xaxis_title=setText(axisTranslations, 'x', 'dashboard.metrics.regressor.partial-dependence.labels'),
    )
    return fig


def generateDependencePlots(X: pd.DataFrame, qualitative_dict,
                            random_forest_model: RandomForestClassifier | RandomForestRegressor, feature,
                            legendTranslations):
    graph = {"predictor": feature, "graph_data": []}
    feature_idx = list(random_forest_model.feature_names_in_).index(feature)
    isObject = False
    variableNames = []
    for q_var in qualitative_dict:
        if q_var['column_name'] == feature:
            isObject = True
            variableNames = q_var['variables']
    dependence = partial_dependence(random_forest_model, X, [feature_idx])
    partial_dependence_values = dependence['average'][0]
    axes = dependence['grid_values'][0]
    if isObject:
        for value, ax in zip(partial_dependence_values, axes):
            name = ax
            for var in variableNames:
                if var['old_value'] == ax:
                    name = var['new_value']
                    break
            graph["graph_data"].append(
                go.Bar(name=name, x=[name], y=[value])
            )
    else:
        x = []
        y = []
        # for value, ax in sorted(zip(partial_dependence_values, axes), key=lambda x: x):
        for value, ax in zip(partial_dependence_values, axes):
            x.append(ax)
            y.append(value)
        graph['graph_data'].append(go.Scatter(
            x=x,
            y=y,
            name=setText(legendTranslations, 'line', 'dashboard.metrics.regressor.partial-dependence.legend'),
            line=dict(color="royalblue", width=1, dash="dot"),
        ))
        graph['graph_data'].append(
            go.Bar(
                name=setText(legendTranslations, 'bars', 'dashboard.metrics.regressor.partial-dependence.legend'),
                x=x, y=y, width=0.5
            )
        )

    return graph


def metricsRegressorLayout(metricsTranslations):
    regressorTranslations = findTranslationsParent(metricsTranslations, 'regressor')
    regressorPredRealTranslations = findTranslationsParent(regressorTranslations, 'prediction-real')
    regressorPredRealTooltipTranslations = findTranslationsParent(regressorPredRealTranslations, 'tooltip')

    regressorRealPredTranslations = findTranslationsParent(regressorTranslations, 'real-prediction')
    regressorRealPredTooltipTranslations = findTranslationsParent(regressorRealPredTranslations, 'tooltip')

    regressorDependenceTranslations = findTranslationsParent(regressorTranslations, 'partial-dependence')

    layout = html.Div(
        [
            html.Div(
                [
                    dbc.Row(
                        [
                            html.Div(id="regression-metrics",
                                     style={'width': 'max-content', 'margin': 'auto', 'overflow': 'scroll'}),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
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
                                                            setText(regressorPredRealTooltipTranslations, 'text-1',
                                                                    'dashboard.metrics.regressor.prediction-real'
                                                                    '.tooltip'),
                                                            html.Strong(
                                                                setText(regressorPredRealTooltipTranslations, 'text-2',
                                                                        'dashboard.metrics.regressor.prediction-real'
                                                                        '.tooltip')
                                                            ),
                                                            setText(regressorPredRealTooltipTranslations, 'text-3',
                                                                    'dashboard.metrics.regressor.prediction-real'
                                                                    '.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            setText(regressorPredRealTooltipTranslations, 'text-4',
                                                                    'dashboard.metrics.regressor.prediction-real'
                                                                    '.tooltip'),
                                                            html.Strong(
                                                                setText(regressorPredRealTooltipTranslations, 'text-5',
                                                                        'dashboard.metrics.regressor.prediction-real'
                                                                        '.tooltip')
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="personalized-tooltip",
                                                target=f"{id_sufix[2]}-info",
                                            ),
                                        ],
                                        style={"display": "flex", "justify-content": "end"},
                                    ),
                                    dbc.Row([html.Div(id="regression-simple-vs-actual")]),
                                ],
                                xs=12,
                                sm=12,
                                md=6,
                                lg=6,
                                xl=6,
                                xxl=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                id=f"{id_sufix[3]}-info",
                                                className="fa fa-info-circle info-icon",
                                            ),
                                            dbc.Tooltip(
                                                [
                                                    html.Plaintext(
                                                        [
                                                            setText(regressorRealPredTooltipTranslations, 'text-1',
                                                                    'dashboard.metrics.regressor.real-prediction'
                                                                    '.tooltip'),
                                                            html.Strong(
                                                                setText(regressorRealPredTooltipTranslations, 'text-2',
                                                                        'dashboard.metrics.regressor.real-prediction'
                                                                        '.tooltip')
                                                            ),
                                                            setText(regressorRealPredTooltipTranslations, 'text-3',
                                                                    'dashboard.metrics.regressor.real-prediction'
                                                                    '.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            setText(regressorRealPredTooltipTranslations, 'text-4',
                                                                    'dashboard.metrics.regressor.real-prediction'
                                                                    '.tooltip'),
                                                            html.Strong(
                                                                setText(regressorRealPredTooltipTranslations, 'text-5',
                                                                        'dashboard.metrics.regressor.real-prediction'
                                                                        '.tooltip')
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="personalized-tooltip",
                                                target=f"{id_sufix[3]}-info",
                                            ),
                                        ],
                                        style={"display": "flex", "justify-content": "end"},
                                    ),
                                    html.Div(
                                        id="regression-simple-vs-pred",
                                    ),
                                ],
                                xs=12,
                                sm=12,
                                md=6,
                                lg=6,
                                xl=6,
                                xxl=6,
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            html.Plaintext(
                                setText(regressorDependenceTranslations, 'title',
                                        'dashboard.metrics.regressor.partial-dependence'),
                                className="rules-title",
                            ),
                            dbc.Col([
                                html.Plaintext(
                                    setText(regressorDependenceTranslations, 'numeric-title',
                                            'dashboard.metrics.regressor.partial-dependence'),
                                    className="rules-title",
                                ),
                                html.Div(id="numeric-dependence-plot")
                            ],
                                xs=12,
                                sm=12,
                                md=6,
                                lg=6,
                                xl=6,
                                xxl=6,
                            ),
                            dbc.Col([
                                html.Plaintext(
                                    setText(regressorDependenceTranslations, 'q-title',
                                            'dashboard.metrics.regressor.partial-dependence'),
                                    className="rules-title",
                                ),
                                html.Div(id="object-dependence-plot"),
                            ],
                                xs=12,
                                sm=12,
                                md=6,
                                lg=6,
                                xl=6,
                                xxl=6,
                            ),
                        ]
                    ),
                ], className="container"
            )
        ],
        style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
    )

    return layout


# CLASSIFIER FUNCTIONS
def generateMatrixExplanationLayout(matrix_explanation, parametersTranslations, keys):
    matrix_generals = matrix_explanation
    matrix_explanation = matrix_generals.pop("matrix_explanation")

    generals_df = (
        pd.DataFrame(matrix_generals, index=[
            setText(parametersTranslations, 'value', 'dashboard.metrics.classifier.matrix.parameters')
        ])
        .transpose()[1:]
        .rename_axis(
            setText(parametersTranslations, 'general-factors', 'dashboard.metrics.classifier.matrix.parameters')
        )
        .reset_index()
    )
    expl = [
        dbc.Table.from_dataframe(
            generals_df,
            bordered=True,
            hover=True,
            className="rules-table",
        )
    ]

    def create_column(m):
        return (
            pd.DataFrame(m["explanation"], index=[m["current_class"]])
            .transpose()
            .rename_axis(
                setText(parametersTranslations, 'individual', 'dashboard.metrics.classifier.matrix.parameters')
            )
        )

    if matrix_explanation != {}:
        descriptions = {
            f"{keys['tpr']}": setText(parametersTranslations, 'sensibility-tooltip',
                                      'dashboard.metrics.classifier.matrix.parameters'),
            f"{keys['fpr']}": setText(parametersTranslations, 'fp-tooltip',
                                      'dashboard.metrics.classifier.matrix.parameters'),
            f"{keys['f1']}": setText(parametersTranslations, 'f1-tooltip',
                                     'dashboard.metrics.classifier.matrix.parameters'),
            f"{keys['precision']}": setText(parametersTranslations, 'precision-tooltip',
                                            'dashboard.metrics.classifier.matrix.parameters'),
        }

        explanation_df = pd.concat(
            [create_column(m) for m in matrix_explanation], axis=1
        ).reset_index()
        expl.append(
            dbc.Table(
                [
                    html.Thead(
                        html.Tr([html.Th(col) for col in explanation_df.columns])
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(
                                        [
                                            html.Span(cell, id=f"{cell}-{row}-{index}"),
                                            dbc.Tooltip(
                                                f"{descriptions[cell]}",
                                                target=f"{cell}-{row}-{index}",
                                            ),
                                        ]
                                        if index == 0
                                        else cell
                                    )
                                    for index, (col, cell) in enumerate(zip(explanation_df.columns, row))
                                ]
                            )
                            for row in explanation_df.values.tolist()
                        ]
                    ),
                ],
                className="rules-table",
                bordered=True,
                hover=True,
            )
        )
    return html.Div(expl)


def get_matrix_explanation(cm, class_names, keys):
    matrix_explanation = []
    true_values = 0
    false_values = 0
    for current_class in range(len(class_names)):
        other_indexes = list(range(len(cm)))
        other_indexes.remove(current_class)
        true_positive = cm[current_class][current_class]
        true_negatives = []
        false_positives = []
        false_negatives = []
        for index in other_indexes:
            true_negatives.append(cm[index][index])
            false_positives.append(cm[index][current_class])
            false_negatives.append(cm[current_class][index])
        if current_class == 0:
            true_values = true_positive + sum(true_negatives)
            false_values = sum(false_positives) + sum(false_negatives)

        explanation = {
            f"{keys['precision']}": (
                (true_positive / (true_positive + sum(false_positives)))
                if true_positive > 0
                else 0
            ),
            f"{keys['tpr']}": (
                (true_positive / (true_positive + sum(false_negatives)))
                if true_positive > 0
                else 0
            ),
            f"{keys['fpr']}": sum(false_positives) / (sum(false_positives) + sum(true_negatives)),
        }

        explanation[keys["f1"]] = (
            (
                    2
                    * explanation[keys["precision"]]
                    * explanation[keys["tpr"]]
                    / (explanation[keys["precision"]] + explanation[keys["tpr"]])
            )
            if (explanation[keys["precision"]] + explanation[keys["tpr"]]) > 0
            else 0
        )

        for elm in explanation:
            explanation[elm] = (
                f"{round((explanation[elm] if not math.isnan(explanation[elm]) else 0) * 100, 2)} %"
            )

        matrix_explanation.append(
            {
                "current_class": class_names[current_class],
                "explanation": explanation,
            }
        )

    return {
        "dtype": "object",
        f"{keys['tv']}": true_values,
        f"{keys['fv']}": false_values,
        f"{keys['accuracy']}": f"{round((true_values / (true_values + false_values)) * 100, 2)} %",
        "matrix_explanation": matrix_explanation,
    }


def __create_matrix(cm, class_names, matrixTranslations):
    labelsTranslations = findTranslationsParent(matrixTranslations, 'labels')
    fig = px.imshow(
        img=cm,
        title=setText(matrixTranslations, 'title', 'dashboard.metrics.classifier.matrix'),
        labels=dict(
            x=setText(labelsTranslations, 'x', 'dashboard.metrics.classifier.matrix.labels'),
            y=setText(labelsTranslations, 'y', 'dashboard.metrics.classifier.matrix.labels'),
            color=setText(labelsTranslations, 'color', 'dashboard.metrics.classifier.matrix.labels')),
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
        old_class_names,
        matrixTranslations,
        keys
):
    y_pred_new = classifier_model.predict(x_test)

    if dropdown_value is not None:

        try:
            positive_class = int(dropdown_value)
        except Exception as e:
            str(e)
            positive_class = dropdown_value
        probability_predictions = classifier_model.predict_proba(x_test)

        try:
            y_pred_new = update_y_pred(
                old_class_names=old_class_names,
                prediction=y_pred_new,
                probability_predictions=probability_predictions,
                cut_off=slider_value,
                positive_class=positive_class,
            )
        except Exception as e:
            print(e)

    # Generate the confusion matrix
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_new)
    return (__create_matrix(cm=cm, class_names=class_names, matrixTranslations=matrixTranslations),
            get_matrix_explanation(
                cm, class_names, keys
            ))


def create_curve(y_scores, y_true, options, pointers, curveTranslations, useScatter=False):
    parameterTranslations = findTranslationsParent(curveTranslations, 'parameters')
    data = []
    trace1 = go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False
    )

    data.append(trace1)
    cont = 0
    for i in range(y_scores.shape[1]):
        y_score = y_scores[:, i]

        pointer = pointers[i]
        fpr, tpr, _ = metrics.roc_curve(get_y_transformed(y_true), y_score, pos_label=i)
        auc_score = metrics.auc(fpr, tpr)

        if pointer >= 0 or not useScatter:
            name = (f"{options[cont]['label']} "
                    f"({setText(parameterTranslations, 'auc', 'dashboard.metrics.classifier.roc.parameters')}="
                    f"{auc_score * 100:.2f} %)")
            trace2 = go.Scatter(x=fpr, y=tpr, name=name, mode="lines")
            data.append(trace2)

            markerText = setText(parameterTranslations, 'marker', 'dashboard.metrics.classifier.roc.parameters')
            if useScatter:
                scatterPointer = int(len(fpr) * pointer)
                trace3 = go.Scatter(
                    x=[fpr[scatterPointer]],
                    y=[tpr[scatterPointer]],
                    legendgroup=f"{markerText} {options[cont]['label']}",
                    name=f"{markerText} {options[cont]['label']}",
                )
                trace4 = go.Scatter(
                    x=[0, fpr[scatterPointer]],
                    y=[tpr[scatterPointer], tpr[scatterPointer]],
                    mode="lines",
                    legendgroup=f"{markerText} {options[cont]['label']}",
                    name=f"{setText(parameterTranslations, 'tpr', 'dashboard.metrics.classifier.roc.parameters')}"
                         f" {round(tpr[scatterPointer] * 100, 2)} %",
                    line=dict(dash="dash"),
                )
                trace5 = go.Scatter(
                    x=[fpr[scatterPointer], fpr[scatterPointer]],
                    y=[0, tpr[scatterPointer]],
                    mode="lines",
                    legendgroup=f"{markerText} {options[cont]['label']}",
                    name=f"{setText(parameterTranslations, 'fpr', 'dashboard.metrics.classifier.roc.parameters')}"
                         f" {round(fpr[scatterPointer] * 100, 2)} %",
                    line=dict(dash="dash"),
                )
                data.append(trace3)
                data.append(trace4)
                data.append(trace5)
        cont += 1

    labelsTexts = findTranslationsParent(curveTranslations, 'labels')
    layout = go.Layout(
        title=setText(curveTranslations, 'title', 'dashboard.metrics.classifier.roc'),
        yaxis=dict(title=setText(labelsTexts, 'y', 'dashboard.metrics.classifier.roc.labels')),
        xaxis=dict(title=setText(labelsTexts, 'x', 'dashboard.metrics.classifier.roc.labels')),
    )

    fig = go.Figure(data=data, layout=layout)

    return setBottomLegend(fig)


def metricsClassifierLayout(metricsTranslations):
    classifierTranslations = findTranslationsParent(metricsTranslations, 'classifier')
    classifierMatrixTranslations = findTranslationsParent(classifierTranslations, 'matrix')
    classifierMatrixTooltipTranslations = findTranslationsParent(classifierMatrixTranslations, 'tooltip')
    classifierMatrixParametersTranslations = findTranslationsParent(classifierMatrixTranslations, 'parameters')

    classifierRocTranslations = findTranslationsParent(classifierTranslations, 'roc')
    classifierRocTooltipTranslations = findTranslationsParent(classifierRocTranslations, 'tooltip')

    cutoff = dbc.Switch(
        label=setText(classifierTranslations, 'cutoff', 'dashboard.metrics.classifier'),
        value=False,
        id="check-cutoff",
        style={"display": "flex", "gap": "1rem"},
    )

    class_selector = dcc.Dropdown(
        value=None,
        id="positive-class-selector",
        placeholder=setText(classifierTranslations, 'class-selector', 'dashboard.metrics.classifier'),
        disabled=True,
    )

    slider = dcc.Slider(0.01, 0.99, 0.1, value=0.5, id="cutoff-slider", disabled=True)

    ROCcutoff = dbc.Switch(
        label=setText(classifierTranslations, 'cutoff', 'dashboard.metrics.classifier'),
        value=False,
        id="ROC-check-cutoff",
        style={"display": "flex", "gap": "1rem"},
    )

    ROCclass_selector = dcc.Dropdown(
        value=None,
        id="ROC-positive-class-selector",
        placeholder=setText(classifierTranslations, 'class-selector', 'dashboard.metrics.classifier'),
        disabled=True,
    )

    ROCslider = dcc.Slider(0.01, 0.99, value=0.5, id="ROC-cutoff-slider", disabled=True)

    layout = html.Div(
        [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Row([html.Div(id="matrix-output-upload")]),
                                    dbc.Tooltip(
                                        [
                                            html.Plaintext(
                                                [
                                                    setText(classifierTranslations, 'cutoff-tooltip',
                                                            'dashboard.metrics.classifier'),
                                                ]
                                            ),
                                        ],
                                        className="personalized-tooltip",
                                        target="check-cutoff",
                                    ),
                                    dbc.Row(
                                        [html.Div([cutoff], style={"padding-left": "20px"})]
                                    ),
                                    dbc.Row([class_selector]),
                                    dbc.Row([slider], style={"padding-top": "20px"}),
                                ],
                                xs=12,
                                sm=12,
                                md=7,
                                lg=7,
                                xl=7,
                                xxl=7,
                            ),
                            dbc.Col(
                                [
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
                                                            setText(classifierMatrixTooltipTranslations, 'text-1',
                                                                    'dashboard.metrics.classifier.matrix.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            html.Strong(
                                                                setText(classifierMatrixTooltipTranslations, 'text-2',
                                                                        'dashboard.metrics.classifier.matrix.tooltip')),
                                                            setText(classifierMatrixTooltipTranslations, 'text-3',
                                                                    'dashboard.metrics.classifier.matrix.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            html.Strong(
                                                                setText(classifierMatrixTooltipTranslations, 'text-4',
                                                                        'dashboard.metrics.classifier.matrix.tooltip')),
                                                            setText(classifierMatrixTooltipTranslations, 'text-5',
                                                                    'dashboard.metrics.classifier.matrix.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            html.Strong(
                                                                setText(classifierMatrixTooltipTranslations, 'text-6',
                                                                        'dashboard.metrics.classifier.matrix.tooltip')),
                                                            setText(classifierMatrixTooltipTranslations, 'text-7',
                                                                    'dashboard.metrics.classifier.matrix.tooltip'),
                                                        ]
                                                    ),
                                                    html.Plaintext(
                                                        [
                                                            html.Strong(
                                                                setText(classifierMatrixTooltipTranslations, 'text-8',
                                                                        'dashboard.metrics.classifier.matrix.tooltip')),
                                                            setText(classifierMatrixTooltipTranslations, 'text-9',
                                                                    'dashboard.metrics.classifier.matrix.tooltip'),
                                                        ]
                                                    ),
                                                ],
                                                className="personalized-tooltip",
                                                target=f"{id_sufix[0]}-info",
                                            ),
                                        ],
                                        style={"display": "flex", "justify-content": "end"},
                                    ),
                                    html.Plaintext(
                                        setText(classifierMatrixParametersTranslations, 'title',
                                                'dashboard.metrics.classifier.matrix.parameters'),
                                        style={"color": "black"}
                                    ),
                                    html.Div(
                                        id="matrix-explanation",
                                    ),
                                ],
                                xs=12,
                                sm=12,
                                md=5,
                                lg=5,
                                xl=5,
                                xxl=5,
                            ),
                        ],
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
                                            setText(classifierRocTooltipTranslations, 'text-1',
                                                    'dashboard.metrics.classifier.roc.tooltip'),
                                            html.Strong(
                                                setText(classifierRocTooltipTranslations, 'text-2',
                                                        'dashboard.metrics.classifier.roc.tooltip')),
                                            setText(classifierRocTooltipTranslations, 'text-3',
                                                    'dashboard.metrics.classifier.roc.tooltip'),
                                            html.Strong(
                                                setText(classifierRocTooltipTranslations, 'text-4',
                                                        'dashboard.metrics.classifier.roc.tooltip')),
                                            setText(classifierRocTooltipTranslations, 'text-5',
                                                    'dashboard.metrics.classifier.roc.tooltip'),
                                        ]
                                    ),
                                    html.Plaintext(
                                        [
                                            setText(classifierRocTooltipTranslations, 'text-6',
                                                    'dashboard.metrics.classifier.roc.tooltip'),
                                        ]
                                    ),
                                ],
                                className="personalized-tooltip",
                                target=f"{id_sufix[1]}-info",
                            ),
                        ],
                        style={"display": "flex", "justify-content": "end"},
                    ),
                    dbc.Row(
                        [
                            dbc.Row(id="roc-output-upload"),
                            dbc.Tooltip(
                                [
                                    html.Plaintext(
                                        [
                                            setText(classifierTranslations, 'cutoff-tooltip',
                                                    'dashboard.metrics.classifier'),
                                        ]
                                    ),
                                ],
                                className="personalized-tooltip",
                                target="ROC-check-cutoff",
                            ),
                            dbc.Row(
                                [html.Div([ROCcutoff], style={"padding-left": "20px"})]
                            ),
                            dbc.Row([ROCclass_selector]),
                            dbc.Row([ROCslider], style={"padding-top": "20px"}),
                        ]
                    ),
                ], className="container"
            )
        ],
        style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
    )

    return layout


# COMMON FUNCTIONS
def setBottomLegend(fig):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="right", x=1)
    )
    return fig


def metricsCallbacks(app, furl, isRegressor: bool = False):
    if isRegressor:
        @app.callback(
            Output("n-dependence-graph", "children"),
            Output("q-dependence-graph", "children"),
            State("path", "href"),
            Input("q-dependence-vars-dropdown", "value"),
            Input("n-dependence-vars-dropdown", "value"),
        )
        def graph_explainers(cl, q_var, n_var):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                regressorTranslations = getDashboardTranslations(current_user.langSelection, 'metrics', 'regressor')
                regressorDependenceTranslations = findTranslationsParent(regressorTranslations, 'partial-dependence')
                regressorDependenceLabelsTranslations = findTranslationsParent(regressorDependenceTranslations,
                                                                               'labels')
                regressorDependenceLegendTranslations = findTranslationsParent(regressorDependenceTranslations,
                                                                               'legend')

                # NORMAL FLOW
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                df: pd.DataFrame = model_x.data_set_data.getElement("dataset")

                qualitative_graph = generateDependencePlots(df.drop(columns=model_x.getElement('target_row')),
                                                            model_x.getElement('q_variables_dict'),
                                                            model_x.getElement('model'),
                                                            q_var, regressorDependenceLegendTranslations)
                numeric_graph = generateDependencePlots(df.drop(columns=model_x.getElement('target_row')),
                                                        model_x.getElement('q_variables_dict'),
                                                        model_x.getElement('model'),
                                                        n_var, regressorDependenceLegendTranslations)
                return (
                    dcc.Graph(
                        figure=addAxisNames(
                            go.Figure(
                                data=numeric_graph["graph_data"],
                                layout=dict(title=numeric_graph["predictor"]),
                            ),
                            regressorDependenceLabelsTranslations
                        )
                    ),
                    dcc.Graph(
                        figure=addAxisNames(
                            go.Figure(
                                data=qualitative_graph["graph_data"],
                                layout=dict(title=qualitative_graph["predictor"], ),
                            ),
                            regressorDependenceLabelsTranslations
                        )
                    ),
                )
            except Exception as e:
                print(e)
            raise PreventUpdate

        @app.callback(
            Output("numeric-dependence-plot", "children"),
            Output("object-dependence-plot", "children"),
            Input("path", "href"),
        )
        def upload_dependence_graphs(cl):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                regressorTranslations = getDashboardTranslations(current_user.langSelection, 'metrics', 'regressor')
                regressorDependenceTranslations = findTranslationsParent(regressorTranslations, 'partial-dependence')
                regressorDependenceLabelsTranslations = findTranslationsParent(regressorDependenceTranslations,
                                                                               'labels')
                regressorDependenceLegendTranslations = findTranslationsParent(regressorDependenceTranslations,
                                                                               'legend')

                # NORMAL FLOW
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()
                original_df: pd.DataFrame = model_x.data_set_data.getElement("dataset")
                q_vars_names = [variable['column_name'] for variable in model_x.getElement('q_variables_dict')]
                n_vars_names = list(model_x.getElement('model').feature_names_in_)

                for elm in q_vars_names:
                    n_vars_names.remove(elm)

                qualitative_graph = generateDependencePlots(
                    original_df.drop(columns=model_x.getElement('target_row')),
                    model_x.getElement('q_variables_dict'),
                    model_x.getElement('model'),
                    q_vars_names[0],
                    regressorDependenceLegendTranslations
                ) if q_vars_names else None
                numeric_graph = generateDependencePlots(
                    original_df.drop(columns=model_x.getElement('target_row')),
                    model_x.getElement('q_variables_dict'),
                    model_x.getElement('model'),
                    n_vars_names[0],
                    regressorDependenceLegendTranslations
                ) if n_vars_names else None

                no_data_text = setText(regressorDependenceTranslations, 'no-data',
                                       'dashboard.metrics.regressor.partial-dependence.no-data')
                return (
                    [
                        dcc.Dropdown(
                            id="n-dependence-vars-dropdown",
                            value=n_vars_names[0],
                            options=[{'label': name, 'value': name} for name in n_vars_names],
                            clearable=False,
                            className='predictor-selector',
                        ),
                        dbc.Col(
                            id=f"n-dependence-graph",
                            children=dcc.Graph(
                                figure=addAxisNames(
                                    go.Figure(
                                        data=numeric_graph["graph_data"],
                                        layout=dict(title=numeric_graph["predictor"]),
                                    ),
                                    regressorDependenceLabelsTranslations
                                )
                            ),
                        )
                    ] if numeric_graph else no_data_text,
                    [
                        dcc.Dropdown(
                            id="q-dependence-vars-dropdown",
                            value=q_vars_names[0],
                            options=[{'label': name, 'value': name} for name in q_vars_names],
                            clearable=False,
                            className='predictor-selector',
                        ),
                        dbc.Col(
                            id=f"q-dependence-graph",
                            children=dcc.Graph(
                                figure=addAxisNames(
                                    go.Figure(
                                        data=qualitative_graph["graph_data"],
                                        layout=dict(title=qualitative_graph["predictor"]),
                                    ),
                                    regressorDependenceLabelsTranslations
                                )
                            ),
                        )
                    ] if qualitative_graph else no_data_text,
                )
            except Exception as e:
                print(e)
                raise PreventUpdate

        @app.callback(
            Output("regression-simple-vs-actual", "children"),
            Output("regression-simple-vs-pred", "children"),
            Output("regression-metrics", "children"),
            Input("path", "href"),
        )
        def graph_pred_vs_real(cl):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                regressorTranslations = getDashboardTranslations(current_user.langSelection, 'metrics', 'regressor')
                regressorParametersTranslations = findTranslationsParent(regressorTranslations, 'parameters')
                regressorPredRealTranslations = findTranslationsParent(regressorTranslations, 'prediction-real')
                regressorPredRealLabelsTranslations = findTranslationsParent(regressorPredRealTranslations, 'labels')
                regressorRealPredTranslations = findTranslationsParent(regressorTranslations, 'real-prediction')
                regressorRealPredLabelsTranslations = findTranslationsParent(regressorRealPredTranslations, 'labels')

                # NORMAL FLOW

                regrKeys = {
                    "mse": setText(regressorParametersTranslations, 'mse', 'dashboard.metrics.regressor.parameters'),
                    "rmse": setText(regressorParametersTranslations, 'rmse', 'dashboard.metrics.regressor.parameters'),
                    "mae": setText(regressorParametersTranslations, 'mae', 'dashboard.metrics.regressor.parameters'),
                    "mape": setText(regressorParametersTranslations, 'mape', 'dashboard.metrics.regressor.parameters'),
                    "r2": setText(regressorParametersTranslations, 'r2', 'dashboard.metrics.regressor.parameters'),
                }

                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                regressor_model: RandomForestRegressor = model_x.getElement("model")
                regressor_dataset: pd.DataFrame = model_x.data_set_data.getElement(
                    "dataset"
                )
                y = regressor_dataset[model_x.getElement('target_row')]
                y_pred = regressor_model.predict(regressor_dataset.drop(columns=model_x.getElement('target_row')))
                fig = px.scatter(x=y, y=y_pred, title=setText(regressorPredRealTranslations, 'title',
                                                              'dashboard.metrics.regressor.prediction-real'),
                                 marginal_y='histogram',
                                 labels={
                                     'x': setText(regressorPredRealLabelsTranslations, 'x',
                                                  'dashboard.metrics.regressor.prediction-real.labels'),
                                     'y': setText(regressorPredRealLabelsTranslations, 'y',
                                                  'dashboard.metrics.regressor.prediction-real.labels')
                                 })
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=y.min(), y0=y.min(),
                    x1=y.max(), y1=y.max()
                )

                fig_1 = dcc.Graph(figure=fig)

                fig = px.scatter(x=y_pred, y=y, title=setText(regressorRealPredTranslations, 'title',
                                                              'dashboard.metrics.regressor.real-prediction'),
                                 marginal_y='histogram',
                                 labels={
                                     'x': setText(regressorRealPredLabelsTranslations, 'x',
                                                  'dashboard.metrics.regressor.real-prediction.labels'),
                                     'y': setText(regressorRealPredLabelsTranslations, 'y',
                                                  'dashboard.metrics.regressor.real-prediction.labels')
                                 })
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=y_pred.min(), y0=y_pred.min(),
                    x1=y_pred.max(), y1=y_pred.max()
                )

                fig_2 = dcc.Graph(figure=fig)

                return fig_1, fig_2, generate_regression_metrics(y, y_pred, regrKeys, regressorParametersTranslations)
            except Exception as e:
                print(e)
                raise PreventUpdate

    else:
        @app.callback(
            Output("matrix-output-upload", "children"),
            Output("matrix-explanation", "children"),
            Output("cutoff-slider", "disabled"),
            Output("positive-class-selector", "disabled"),
            Output("positive-class-selector", "options"),
            Input("check-cutoff", "value"),
            Input("positive-class-selector", "value"),
            Input("cutoff-slider", "value"),
            Input("path", "href"),
        )
        def graph_matrix(cutoff, positive_class, slider, cl):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                classifierTranslations = getDashboardTranslations(current_user.langSelection, 'metrics', 'classifier')
                classifierMatrixTranslations = findTranslationsParent(classifierTranslations, 'matrix')
                classifierMatrixParametersTranslations = findTranslationsParent(classifierMatrixTranslations,
                                                                                'parameters')

                # NORMAL FLOW

                matrixKeys = {
                    "precision": setText(classifierMatrixParametersTranslations, 'precision',
                                         'dashboard.metrics.classifier.matrix.parameters'),
                    "tpr": setText(classifierMatrixParametersTranslations, 'sensibility',
                                   'dashboard.metrics.classifier.matrix.parameters'),
                    "fpr": setText(classifierMatrixParametersTranslations, 'fp',
                                   'dashboard.metrics.classifier.matrix.parameters'),
                    "f1": setText(classifierMatrixParametersTranslations, 'f1',
                                  'dashboard.metrics.classifier.matrix.parameters'),
                    "tv": setText(classifierMatrixParametersTranslations, 'tv',
                                  'dashboard.metrics.classifier.matrix.parameters'),
                    "fv": setText(classifierMatrixParametersTranslations, 'fv',
                                  'dashboard.metrics.classifier.matrix.parameters'),
                    "accuracy": setText(classifierMatrixParametersTranslations, 'accuracy',
                                        'dashboard.metrics.classifier.matrix.parameters'),
                }

                classifier_dbmodel: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                    ExplainedClassifierModel.explainer_model_id == model_id
                ).first()

                model_x = classifier_dbmodel.explainer_model

                classifier_model: RandomForestClassifier = model_x.getElement("model")
                classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement(
                    "dataset"
                )

                target_description = classifier_dbmodel.getElement("target_names_dict")
                class_names = [
                    element["new_value"] for element in target_description["variables"]
                ]
                old_class_names = [
                    element["old_value"] for element in target_description["variables"]
                ]

                if positive_class or slider or cutoff:
                    if cutoff and positive_class is not None:
                        matrix_graph, matrix_explanation = initialize_matrix(
                            dropdown_value=positive_class,
                            slider_value=slider,
                            y_test=classifier_dataset[model_x.target_row],
                            x_test=classifier_dataset.drop(columns=model_x.target_row),
                            classifier_model=classifier_model,
                            class_names=class_names,
                            old_class_names=old_class_names,
                            matrixTranslations=classifierMatrixTranslations,
                            keys=matrixKeys,
                        )
                        return (
                            dcc.Graph(figure=matrix_graph),
                            generateMatrixExplanationLayout(matrix_explanation, classifierMatrixParametersTranslations,
                                                            matrixKeys),
                            False,
                            False,
                            get_target_dropdown(target_description["variables"]),
                        )
                    else:
                        matrix_graph, matrix_explanation = initialize_matrix(
                            dropdown_value=None,
                            slider_value=None,
                            y_test=classifier_dataset[model_x.target_row],
                            x_test=classifier_dataset.drop(columns=model_x.target_row),
                            classifier_model=classifier_model,
                            class_names=class_names,
                            old_class_names=old_class_names,
                            matrixTranslations=classifierMatrixTranslations,
                            keys=matrixKeys,
                        )
                        div = dcc.Graph(figure=matrix_graph)
                        explanation = generateMatrixExplanationLayout(matrix_explanation,
                                                                      classifierMatrixParametersTranslations,
                                                                      matrixKeys)
                        if cutoff and positive_class is None:
                            return (
                                div,
                                explanation,
                                True,
                                False,
                                get_target_dropdown(target_description["variables"]),
                            )
                        else:
                            return (
                                div,
                                explanation,
                                True,
                                True,
                                get_target_dropdown(target_description["variables"]),
                            )
                else:
                    return (
                        None,
                        None,
                        None,
                        get_target_dropdown(target_description["variables"]),
                    )
            except Exception as e:
                print(e)
                raise PreventUpdate

        @app.callback(
            Output("roc-output-upload", "children"),
            Output("ROC-cutoff-slider", "disabled"),
            Output("ROC-positive-class-selector", "disabled"),
            Output("ROC-positive-class-selector", "options"),
            Input("ROC-check-cutoff", "value"),
            Input("ROC-positive-class-selector", "value"),
            Input("ROC-cutoff-slider", "value"),
            Input("path", "href"),
        )
        def graph_roc(cutoff, positive_class, slider, cl):
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                # TRANSLATIONS
                classifierTranslations = getDashboardTranslations(current_user.langSelection, 'metrics', 'classifier')
                classifierRocTranslations = findTranslationsParent(classifierTranslations, 'roc')

                # NORMAL FLOW

                classifier_dbmodel: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                    ExplainedClassifierModel.explainer_model_id == model_id
                ).first()

                model_x = classifier_dbmodel.explainer_model

                classifier_model: RandomForestClassifier = model_x.getElement("model")
                classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement(
                    "dataset"
                )

                target_description = classifier_dbmodel.getElement("target_names_dict")

                if positive_class or slider or cutoff:
                    if cutoff and positive_class is not None:
                        pointers = [
                            slider / (len(target_description["variables"]) - 1)
                            for _ in target_description["variables"]
                        ]
                        pointers[positive_class] = 1 - slider
                        return (
                            dcc.Graph(
                                figure=create_curve(
                                    y_scores=classifier_model.predict_proba(
                                        classifier_dataset.drop(columns=model_x.target_row)
                                    ),
                                    y_true=classifier_dataset[model_x.target_row],
                                    options=get_target_dropdown(
                                        target_description["variables"]
                                    ),
                                    pointers=pointers,
                                    useScatter=True,
                                    curveTranslations=classifierRocTranslations
                                )
                            ),
                            False,
                            False,
                            get_target_dropdown(target_description["variables"]),
                        )
                    else:
                        pointers = [-1 for _ in target_description["variables"]]
                        div = (
                            dcc.Graph(
                                figure=create_curve(
                                    y_scores=classifier_model.predict_proba(
                                        classifier_dataset.drop(columns=model_x.target_row)
                                    ),
                                    y_true=classifier_dataset[model_x.target_row],
                                    options=get_target_dropdown(
                                        target_description["variables"]
                                    ),
                                    pointers=pointers,
                                    curveTranslations=classifierRocTranslations
                                )
                            ),
                        )
                        if cutoff and positive_class is None:
                            return (
                                div,
                                True,
                                False,
                                get_target_dropdown(target_description["variables"]),
                            )
                        else:
                            return (
                                div,
                                True,
                                True,
                                get_target_dropdown(target_description["variables"]),
                            )
                else:
                    return (
                        None,
                        None,
                        None,
                        get_target_dropdown(target_description["variables"]),
                    )
            except Exception as e:
                print(e)
                raise PreventUpdate
