import math
from pyclbr import Function

import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.proccessor.model.dataset_interaction_methods import (
    get_y_transformed,
    update_y_pred,
)
from app.proccessor.models import ExplainedClassifierModel, ExplainedModel

from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go

keys = {
    "precision": "Precisión",
    "tpr": "Sensibilidad",
    "fpr": "Tasa FP",
    "f1": "Medición F1",
    "tv": "Predicciones Correctas",
    "fv": "Predicciones Fallidas",
    "accuracy": "Precisión Global",
    "mse": "Error Cuadrático Medio",
    "rmse": "Raíz del Error Cuadrático Medio",
    "mae": "Error Absoluto Medio",
    "mape": "Error Absoluto Porcentual Medio",
    "r2": "Coeficiente de Determinación R²",
}


def generate_regression_metrics(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    regression_metrics = {
        f"{keys['mse']}": round(mse, 4),
        f"{keys['rmse']}": round(np.sqrt(mse), 4),
        f"{keys['mae']}": round(mean_absolute_error(y, y_pred), 4),
        f"{keys['mape']}": round(np.mean(np.abs((y - y_pred) / y)) * 100, 4),
        f"{keys['r2']}": round(r2_score(y, y_pred), 4),
    }

    descriptions = {
        f"{keys['mse']}": "El MSE mide la media de los errores cuadrados entre los valores reales y los valores "
                          "predichos. Un MSE bajo indica que el modelo se ajusta bien a los datos.",
        f"{keys['rmse']}": "El RMSE es la raíz cuadrada del MSE. Es una métrica más fácil de interpretar que el MSE, "
                           "ya que tiene la misma unidad que los valores reales.",
        f"{keys['mae']}": "El MAE mide la media de los errores absolutos entre los valores reales y los valores "
                          "predichos. Un MAE bajo indica que el modelo se ajusta bien a los datos.",
        f"{keys['mape']}": "El MAPE mide la media de los errores absolutos porcentuales entre los valores reales y "
                           "los valores predichos. Un MAPE bajo indica que el modelo se ajusta bien a los datos.",
        f"{keys['r2']}": "El R² mide la proporción de la varianza en los valores reales que es explicada por el "
                         "modelo. Un R² alto indica que el modelo se ajusta bien a los datos.",
    }

    regression_metrics_df = (
        pd.DataFrame(regression_metrics, index=["Valor"])
        .transpose()[0:]
        .rename_axis("Métricas")
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

    print(regression_metrics_df)

    print("Error Cuadrático Medio (mean-squared-error):", round(regression_metrics[keys['mse']], 4))
    print("Raíz del Error Cuadrático Medio (root-mean-squared-error):", round(regression_metrics[keys['rmse']], 4))
    print("Error Absoluto Medio (mean-absolute-error):", round(regression_metrics[keys['mae']], 4))
    print("Error Absoluto Porcentual Medio (mean-absolute-percentage-error):",
          round(regression_metrics[keys['mape']], 4))
    print("Coeficiente de Determinación (R-squared):", round(regression_metrics[keys['r2']], 4))

    return regression_metrics_table


def setBottomLegend(fig):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="right", x=1)
    )
    return fig


def get_target_dropdown(values_dict):
    return [
        {"label": value["new_value"], "value": index}
        for index, value in enumerate(values_dict)
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
        pd.DataFrame(matrix_generals, index=["Factores Generales"])
        .transpose()[1:]
        .rename_axis("Valor")
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
            .rename_axis("Parámetros Individuales")
        )

    if matrix_explanation != {}:
        descriptions = {
            f"{keys['tpr']}": "Proporción de predicciones correctas entre todos los valores reales de la clase. ("
                              "Mayor es Mejor)",
            f"{keys['fpr']}": "Proporción de predicciones erroneas entre todos los valores reales de otras clases "
                              "clase. (Menor es Mejor)",
            f"{keys['f1']}": "Media armónica de la precisión y la sensibilidad. (Mayor es Mejor)",
            f"{keys['precision']}": "Proporción de verdaderos positivos entre todos los positivos predichos. (Mayor "
                                    "es Mejor)",
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
                                    for index, (col, cell) in enumerate(
                                    zip(explanation_df.columns, row)
                                )
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


def get_matrix_explanation(cm, class_names):
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
            f"{keys['fpr']}": sum(false_positives)
                              / (sum(false_positives) + sum(true_negatives)),
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
        old_class_names,
):
    y_pred_new = classifier_model.predict(x_test)

    if dropdown_value is not None:
        try:
            positive_class = int(dropdown_value)
        except:  # noqa: E722
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
    return __create_matrix(cm=cm, class_names=class_names), get_matrix_explanation(
        cm, class_names
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
        fpr, tpr, _ = metrics.roc_curve(get_y_transformed(y_true), y_score, pos_label=i)
        auc_score = metrics.auc(fpr, tpr)

        if pointer >= 0 or not useScatter:
            name = f"{options[cont]['label']} (AUC={auc_score * 100:.2f} %)"
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

    return setBottomLegend(fig)


cutoff = dbc.Switch(
    label="Punto de Corte",
    value=False,
    id="check-cutoff",
    style={"display": "flex", "gap": "1rem"},
)

class_selector = dcc.Dropdown(
    value=None,
    id="positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    disabled=True,
)

slider = dcc.Slider(0.01, 0.99, 0.1, value=0.5, id="cutoff-slider", disabled=True)

ROCcutoff = dbc.Switch(
    label="Punto de Corte",
    value=False,
    id="ROC-check-cutoff",
    style={"display": "flex", "gap": "1rem"},
)

ROCclass_selector = dcc.Dropdown(
    value=None,
    id="ROC-positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    disabled=True,
)

ROCslider = dcc.Slider(0.01, 0.99, value=0.5, id="ROC-cutoff-slider", disabled=True)

id_sufix = ["confusion-matrix", "roc-curve", "pred-real", "real-pred"]

metricsClassifierLayout = html.Div(
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
                                                "Punto porcentual en el que el modelo decide clasificar una clase como verdadera.",
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
                                                        "Matriz de Confusión: Evalúa la precisión del modelo en la clasificación de ejemplos.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        html.Strong("VP: "),
                                                        "ejemplos positivos correctamente clasificados.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        html.Strong("VN: "),
                                                        "ejemplos negativos correctamente clasificados.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        html.Strong("FP: "),
                                                        "ejemplos negativos incorrectamente clasificados como positivos.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        html.Strong("FN: "),
                                                        "ejemplos positivos incorrectamente clasificados como negativos.",
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
                                    "Parámetros Obtenidos", style={"color": "black"}
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
                                        "Curva ROC: Muestra la relación entre la tasa de Verdaderos Positivos (",
                                        html.Strong("Sensibilidad"),
                                        ") y la tasa de Falsos Positivos (",
                                        html.Strong("1-Especificidad"),
                                        ") a diferentes umbrales de clasificación.",
                                    ]
                                ),
                                html.Plaintext(
                                    [
                                        """
                                        * El área bajo la curva (AUC) es una métrica que resume 
                                        la performance del modelo. Un AUC cercano a 1 indica un modelo 
                                        que puede distinguir perfectamente entre positivos y negativos, mientras que un 
                                        AUC cercano a 0.5 indica un modelo que no es mejor que una elección aleatoria.
                                        """,
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
                                        "Punto porcentual en el que el modelo decide clasificar una clase como verdadera.",
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

metricsRegressorLayout = html.Div(
    [
        html.Div(
            [
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
                                                        "Evalúa la precisión del modelo en la "
                                                        "regresión de resultados de modo que un mejor modelo "
                                                        "está representado por los ",
                                                        html.Strong("puntos más cercanos a la linea discontinua "),
                                                        "central.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        "La línea discontínua representa los ",
                                                        html.Strong("valores reales "),
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
                                                        "Evalúa la precisión del modelo en la "
                                                        "regresión de resultados de modo que un mejor modelo "
                                                        "está representado por los ",
                                                        html.Strong("puntos más cercanos a la linea discontinua "),
                                                        "central.",
                                                    ]
                                                ),
                                                html.Plaintext(
                                                    [
                                                        "La línea discontínua representa la ",
                                                        html.Strong("predicción "),
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
                        html.Div(id="regression-metrics", style={'width': 'max-content', 'margin': 'auto'}),
                    ],
                ),
            ], className="container"
        )
    ],
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def metricsCallbacks(app, furl: Function, isRegressor: bool = False):
    if isRegressor:
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
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                regressor_model: RandomForestRegressor = model_x.getElement("model")
                regressor_dataset: pd.DataFrame = model_x.data_set_data.getElement(
                    "dataset"
                )
                y = regressor_dataset[model_x.getElement('target_row')]
                y_pred = regressor_model.predict(regressor_dataset.drop(columns=model_x.getElement('target_row')))
                fig = px.scatter(x=y, y=y_pred, title='Prediccción vs Valores Reales', marginal_y='histogram',
                                 labels={'x': 'Valor Real', 'y': 'Predicción'})
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=y.min(), y0=y.min(),
                    x1=y.max(), y1=y.max()
                )

                fig_1 = dcc.Graph(figure=fig)

                fig = px.scatter(x=y_pred, y=y, title='Valores Reales vs Prediccción', marginal_y='histogram',
                                 labels={'x': 'Prediccción', 'y': 'Valor Real'})
                fig.add_shape(
                    type="line", line=dict(dash='dash'),
                    x0=y_pred.min(), y0=y_pred.min(),
                    x1=y_pred.max(), y1=y_pred.max()
                )

                fig_2 = dcc.Graph(figure=fig)

                return fig_1, fig_2, generate_regression_metrics(y, y_pred)
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
                            old_class_names=old_class_names,
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
