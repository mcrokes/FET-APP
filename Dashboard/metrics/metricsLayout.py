from math import nan
import math
from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from app.proccessor.model.dataset_interaction_methods import update_y_pred
from app.proccessor.models import ExplainedClassifierModel, ModelForProccess

from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go


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


def generateMatrixExplanationLayout(matrix_explanation):
    matrix_generals = matrix_explanation
    matrix_explanation = matrix_generals.pop("matrix_explanation")

    generals_df = (
        pd.DataFrame(matrix_generals, index=["Generals"])
        .transpose()[1:]
        .rename_axis("Parameters")
        .reset_index()
    )
    expl = [
        dbc.Table.from_dataframe(generals_df, striped=True, bordered=True, hover=True)
    ]

    def create_column(m):
        return (
            pd.DataFrame(m["explanation"], index=[m["current_class"]])
            .transpose()
            .rename_axis("Parameters")
        )

    if matrix_explanation != {}:

        explanation_df = pd.concat(
            [create_column(m) for m in matrix_explanation], axis=1
        ).reset_index()
        expl.append(
            dbc.Table.from_dataframe(
                explanation_df, striped=True, bordered=True, hover=True
            )
        )
    return html.Div(expl)


def get_matrix_explanation(cm, class_names, positive_class):
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

        keys = {"precision": "Precisión", "tpr": "TP Rate Recall (Sensibilidad)", "fpr": "FP Rate", "f1": "F1 Score"}
        explanation = {
            f"{keys["precision"]}": true_positive / (true_positive + sum(false_positives)),
            f"{keys['tpr']}": true_positive
            / (true_positive + sum(false_negatives)),
            f"{keys['fpr']}": sum(false_positives)
            / (sum(false_positives) + sum(true_negatives)),
        }

        explanation[keys["f1"]] = (
            2
            * explanation[keys["precision"]]
            * explanation[keys['tpr']]
            / (explanation[keys["precision"]] + explanation[keys['tpr']])
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
        "true_values": true_values,
        "false_values": false_values,
        "accuracy": f"{round((true_values / (true_values + false_values)) *100, 2)} %",
        "matrix_explanation": matrix_explanation,
    }


def __create_matrix(cm, class_names):

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

    # Generate the confusion matrix
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_new)

    return __create_matrix(cm=cm, class_names=class_names), get_matrix_explanation(
        cm, class_names, dropdown_value
    )


def create_curve(y_scores, y_true, options, pointers, useScatter=False):
    data = []
    trace1 = go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False
    )

    data.append(trace1)
    cont = 0
    for i in range(y_scores.shape[1]):
        y_score = y_scores[:, i]

        pointer = pointers[i]

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=i)
        auc_score = metrics.auc(fpr, tpr)

        if pointer >= 0 or not useScatter:
            name = f"{options[cont]['label']} (AUC={auc_score*100:.2f} %)"
            trace2 = go.Scatter(x=fpr, y=tpr, name=name, mode="lines")
            data.append(trace2)

            if useScatter:
                scatterPointer = int(len(fpr) * pointer)
                trace3 = go.Scatter(
                    x=[fpr[scatterPointer]],
                    y=[tpr[scatterPointer]],
                    legendgroup=f"Marker {options[cont]['label']}",
                    name=f"Marker {options[cont]['label']}",
                )
                trace4 = go.Scatter(
                    x=[0, fpr[scatterPointer]],
                    y=[tpr[scatterPointer], tpr[scatterPointer]],
                    mode="lines",
                    legendgroup=f"Marker {options[cont]['label']}",
                    name=f"TPR {round(tpr[scatterPointer] * 100, 2)} %",
                    line=dict(dash="dash"),
                )
                trace5 = go.Scatter(
                    x=[fpr[scatterPointer], fpr[scatterPointer]],
                    y=[0, tpr[scatterPointer]],
                    mode="lines",
                    legendgroup=f"Marker {options[cont]['label']}",
                    name=f"FPR {round(fpr[scatterPointer] * 100, 2)} %",
                    line=dict(dash="dash"),
                )
                data.append(trace3)
                data.append(trace4)
                data.append(trace5)
        cont += 1

    layout = go.Layout(
        title="ROC-AUC curva",
        yaxis=dict(title="Tasa de Positivos"),
        xaxis=dict(title="Tasa de Falsos Positivos"),
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


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

ROCcutoff = dbc.Switch(
    label="USAR COTOFF",
    value=False,
    id="ROC-check-cutoff",
)

ROCclass_selector = dcc.Dropdown(
    value=None,
    id="ROC-positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    disabled=True,
)

ROCslider = dcc.Slider(0.01, 0.99, value=0.5, id="ROC-cutoff-slider", disabled=True)

metricsLayout = html.Div(
    [
        html.Div(
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
                                    ["TEXTO CON EXPLICACION DEL GRÁFICO"],
                                    id="matrix-explanation",
                                )
                            ],
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
                        dbc.Row(id="roc-output-upload"),
                        dbc.Row(
                            [html.Div([ROCcutoff], style={"padding-left": "20px"})]
                        ),
                        dbc.Row([ROCclass_selector]),
                        dbc.Row([ROCslider], style={"padding-top": "20px"}),
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
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == model_id
            ).first()

            classifier_model: RandomForestClassifier = model_x.getElement("model")
            classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement("dataset")

            target_description = {
                "column_name": "Sobreviviente",
                "variables": [
                    {"old_value": 0, "new_value": "Muere"},
                    {"old_value": 1, "new_value": "Vive"},
                ],
            }
            class_names = [
                element["new_value"] for element in target_description["variables"]
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
                    )
                    return (
                        dcc.Graph(figure=matrix_graph),
                        generateMatrixExplanationLayout(matrix_explanation),
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
                    )
                    div = dcc.Graph(figure=matrix_graph)
                    explanation = generateMatrixExplanationLayout(matrix_explanation)
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
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == model_id
            ).first()

            classifier_model: RandomForestClassifier = model_x.getElement("model")
            classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement("dataset")

            target_description = model_x.getElement("target_names_dict")

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
                            )
                        ),
                        False,
                        False,
                        get_target_dropdown(target_description["variables"]),
                    )
                else:
                    pointers = [-1 for element in target_description["variables"]]
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
