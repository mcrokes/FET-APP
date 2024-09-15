from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from flask_login import current_user

from app.API.utils import setText, findTranslationsParent, getDashboardTranslations
from app.processor.models import ExplainedModel


def generateDataSetDistributions(df: pd.DataFrame, feature, legendTranslations):
    graph = {"predictor": feature, "graph_data": []}
    values = list(set(df[feature]))
    counts = df[feature].value_counts()
    if len(values) < 5:
        for value in values:
            graph["graph_data"].append(
                go.Bar(name=value, x=[value], y=[counts[value]])
            )
    else:
        x = []
        y = []
        for value, count in sorted(zip(values, counts), key=lambda x: x):
            x.append(value)
            y.append(count)
        graph['graph_data'].append(go.Scatter(
            x=x,
            y=y,
            name=setText(legendTranslations, 'line', 'dashboard.data.common.distributions.legend'),
            line=dict(color="royalblue", width=1, dash="dot"),
        ))
        graph['graph_data'].append(
            go.Bar(name=setText(legendTranslations, 'bars', 'dashboard.data.common.distributions.legend'), x=x, y=y,
                   width=0.5))
    return graph


def datasetLayout(dataTranslations):
    commonTranslations = findTranslationsParent(dataTranslations, 'common')
    distributionTranslations = findTranslationsParent(commonTranslations, 'distributions')
    datasetTranslations = findTranslationsParent(commonTranslations, 'dataset')
    datasetTooltipTranslations = findTranslationsParent(datasetTranslations, 'tooltip')
    correlationTranslations = findTranslationsParent(commonTranslations, 'correlation')
    correlationTooltipTranslations = findTranslationsParent(correlationTranslations, 'tooltip')

    layout = html.Div(
        [
            dcc.Loading(
                [
                    html.Div(id='dummy', style={'display': 'none'}),
                    html.Div([
                        html.Plaintext(
                            [
                                setText(commonTranslations, 'title', 'dashboard.data.common'),
                                html.Strong(id="dataset-title"),
                            ],
                            className="rules-title",
                            style={"font-size": "30px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Plaintext(
                                            setText(datasetTranslations, 'modified-title',
                                                    'dashboard.data.common.dataset'),
                                            className="rules-title",
                                        ),
                                        html.I(
                                            id="dataset-info",
                                            className="fa fa-info-circle info-icon",
                                        ),
                                        dbc.Tooltip(
                                            [
                                                html.Plaintext(
                                                    [
                                                        setText(datasetTooltipTranslations, 'text-1',
                                                                'dashboard.data.common.dataset.tooltip'),
                                                        html.Strong(setText(datasetTooltipTranslations, 'text-2',
                                                                            'dashboard.data.common.dataset.tooltip')),
                                                        setText(datasetTooltipTranslations, 'text-3',
                                                                'dashboard.data.common.dataset.tooltip'),
                                                        html.Strong(setText(datasetTooltipTranslations, 'text-4',
                                                                            'dashboard.data.common.dataset.tooltip')),
                                                        setText(datasetTooltipTranslations, 'text-5',
                                                                'dashboard.data.common.dataset.tooltip'),
                                                        html.Strong(setText(datasetTooltipTranslations, 'text-6',
                                                                            'dashboard.data.common.dataset.tooltip')),
                                                        setText(datasetTooltipTranslations, 'text-7',
                                                                'dashboard.data.common.dataset.tooltip'),
                                                    ]
                                                ),
                                            ],
                                            className="personalized-tooltip",
                                            target="dataset-info",
                                        ),
                                    ],
                                    className="title-hint-container",
                                ),
                                html.Div(
                                    id="modified-dataset-view", style={"overflow": "scroll", "border-radius": "5px"}
                                ),
                                html.Div(id="test-tert"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Plaintext(
                                    setText(datasetTranslations, 'title', 'dashboard.data.common.dataset'),
                                    className="rules-title",
                                ),
                                html.Div(id="dataset-view", style={"overflow": "scroll", "border-radius": "5px"}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Plaintext(
                                    setText(commonTranslations, 'description-title', 'dashboard.data.common'),
                                    className="rules-title",
                                ),
                                html.Div(id="features-description"),
                            ],
                            style={'max-width': '800px', 'margin': 'auto auto 5rem auto'}
                        ),
                        dbc.Row(
                            [
                                html.Plaintext(
                                    setText(distributionTranslations, 'title', 'dashboard.data.common.distributions'),
                                    className="rules-title",
                                ),
                                dbc.Col([
                                    html.Plaintext(
                                        setText(distributionTranslations, 'numeric-title',
                                                'dashboard.data.common.distributions'),
                                        className="rules-title",
                                    ),
                                    html.Div(id="numeric-plot")],
                                    xs=12,
                                    sm=12,
                                    md=6,
                                    lg=6,
                                    xl=6,
                                    xxl=6, ),

                                dbc.Col([
                                    html.Plaintext(
                                        setText(distributionTranslations, 'q-title',
                                                'dashboard.data.common.distributions'),
                                        className="rules-title",
                                    ),
                                    html.Div(id="object-plot"),
                                ],
                                    xs=12,
                                    sm=12,
                                    md=6,
                                    lg=6,
                                    xl=6,
                                    xxl=6, ),
                                html.Div(
                                    [
                                        html.Plaintext(
                                            setText(correlationTranslations, 'title',
                                                    'dashboard.data.common.correlation'),
                                            className="rules-title",
                                        ),
                                        html.I(
                                            id="correlation-info",
                                            className="fa fa-info-circle info-icon",
                                        ),
                                        dbc.Tooltip(
                                            [
                                                html.Plaintext(
                                                    [
                                                        setText(
                                                            correlationTooltipTranslations, 'text-1',
                                                            'dashboard.data.common.correlation.tooltip'),
                                                        setText(
                                                            correlationTooltipTranslations, 'text-2',
                                                            'dashboard.data.common.correlation.tooltip'),
                                                        html.Strong(
                                                            setText(correlationTooltipTranslations, 'text-3',
                                                                    'dashboard.data.common.correlation.tooltip')),
                                                        setText(correlationTooltipTranslations, 'text-4',
                                                                'dashboard.data.common.correlation.tooltip'),
                                                        html.Strong(
                                                            setText(correlationTooltipTranslations, 'text-5',
                                                                    'dashboard.data.common.correlation.tooltip')),
                                                        setText(correlationTooltipTranslations, 'text-6',
                                                                'dashboard.data.common.correlation.tooltip'),
                                                        html.Strong(
                                                            setText(correlationTooltipTranslations, 'text-7',
                                                                    'dashboard.data.common.correlation.tooltip')),
                                                        setText(correlationTooltipTranslations, 'text-8',
                                                                'dashboard.data.common.correlation.tooltip'),
                                                    ]
                                                ),
                                            ],
                                            className="personalized-tooltip",
                                            target="correlation-info",
                                        ),
                                    ],
                                    className="title-hint-container",
                                ),
                                html.Div(id="correlation-plot"),
                            ],
                        ),
                    ], className="container")
                ],
            )
        ],
        className="section-content",
        style={"margin": "auto"},
    )
    return layout


def datasetCallbacks(app, furl, isRegressor: bool = False):
    def setBottomLegend(fig):
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.4, xanchor="right", x=1
            ),
        )
        return fig

    def addAxisNames(fig, axisTranslations):
        fig = setBottomLegend(fig)
        fig.update_layout(
            yaxis_title=setText(axisTranslations, 'y', 'dashboard.data.common.distributions.labels'),
            xaxis_title=setText(axisTranslations, 'x', 'dashboard.data.common.distributions.labels'),
        )
        return fig

    @app.callback(
        Output("n-graph", "children", allow_duplicate=True),
        Output("q-graph", "children"),
        State("path", "href"),
        Input("q-vars-dropdown", "value"),
        Input("n-vars-dropdown", "value"),
        prevent_initial_call=True
    )
    def graph_explainers(cl, q_var, n_var):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            # TRANSLATIONS
            translationsCommon = getDashboardTranslations(current_user.langSelection, 'data', 'common')
            translationsDistributions = findTranslationsParent(translationsCommon, 'distributions')
            legendTranslations = findTranslationsParent(translationsDistributions, 'legend')
            axisTranslations = findTranslationsParent(translationsDistributions, 'labels')

            # NORMAL FLOW
            model_x: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == model_id
            ).first()

            df: pd.DataFrame = model_x.data_set_data.getElement("dataset_modified")
            qualitative_graph = generateDataSetDistributions(df, q_var, legendTranslations)
            numeric_graph = generateDataSetDistributions(df, n_var, legendTranslations)
            return (
                dcc.Graph(
                    figure=addAxisNames(
                        go.Figure(
                            data=numeric_graph["graph_data"],
                            layout=dict(title=numeric_graph["predictor"]),
                        ),
                        axisTranslations
                    )
                ),
                dcc.Graph(
                    figure=addAxisNames(
                        go.Figure(
                            data=qualitative_graph["graph_data"],
                            layout=dict(title=qualitative_graph["predictor"], ),
                        ),
                        axisTranslations
                    )
                ),
            )
        except Exception as e:
            print(e)
        raise PreventUpdate

    @app.callback(
        Output("dataset-title", "children", allow_duplicate=True),
        Output("modified-dataset-view", "children"),
        Output("dataset-view", "children"),
        Output("features-description", "children"),
        Output("numeric-plot", "children"),
        Output("object-plot", "children"),
        Output("correlation-plot", "children"),
        Input("path", "href"),
        prevent_initial_call=True
    )
    def graph_explainers(cl):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            # TRANSLATIONS
            translationsCommon = getDashboardTranslations(current_user.langSelection, 'data', 'common')
            translationsDistributions = findTranslationsParent(translationsCommon, 'distributions')
            legendTranslations = findTranslationsParent(translationsDistributions, 'legend')
            axisTranslations = findTranslationsParent(translationsDistributions, 'labels')

            # NORMAL FLOW
            model_x: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == model_id
            ).first()

            descriptions = model_x.getElement("features_description")
            descriptions_df: pd.DataFrame = pd.DataFrame(list(descriptions.items()), columns=[
                setText(translationsCommon, 'columns-header', 'dashboard.data.columns-header'),
                setText(translationsCommon, 'descriptions-header', 'dashboard.data.descriptions-header'),
            ])

            def replace_empty_values(x):
                if x.strip() == '':
                    print('x: ', x)
                    print('setText: ', setText(translationsCommon, 'no-description', 'dashboard.data.no-description'))
                    return setText(translationsCommon, 'no-description', 'dashboard.data.no-description'),
                else:
                    return x

            descriptions_df = descriptions_df.map(lambda x: replace_empty_values(x))

            print('descriptions_df: ', descriptions_df)

            original_df: pd.DataFrame = model_x.data_set_data.getElement("dataset")
            original_df_with_index = original_df.rename_axis("Índice").reset_index()
            df: pd.DataFrame = model_x.data_set_data.getElement("dataset_modified")
            df_with_index = df.copy()
            if isRegressor:
                df_with_index[model_x.getElement('target_row')] = df_with_index[model_x.getElement('target_row')].apply(
                    lambda x: str(x) + f" {model_x.explainer_regressor.getElement('unit')}")
            if model_x.indexColumnName:
                df_with_index.insert(0, model_x.indexColumnName, model_x.getElement('indexesList'))
            else:
                df_with_index = df_with_index.rename_axis("Índice").reset_index()

            dtt = model_x.explainer_regressor.getElement(
                "name") if isRegressor else model_x.explainer_classifier.getElement("name")

            q_vars_names = [variable['column_name'] for variable in model_x.getElement('q_variables_dict')]
            n_vars_names = list(df.columns)
            if not isRegressor:
                q_vars_names.append(model_x.getElement('target_row'))

            for elm in q_vars_names:
                n_vars_names.remove(elm)

            qualitative_graph = generateDataSetDistributions(df, q_vars_names[0],
                                                             legendTranslations) if q_vars_names else None
            numeric_graph = generateDataSetDistributions(df, n_vars_names[0],
                                                         legendTranslations) if n_vars_names else None
            corr_matrix = original_df.drop(
                columns=model_x.getElement("target_row")
            ).corr(method="pearson")

            no_data_text = setText(translationsDistributions, 'no-data', 'dashboard.data.common.distributions')
            return (
                dtt,
                html.Div(
                    [
                        dash_table.DataTable(
                            data=df_with_index.to_dict("records"),
                            columns=[
                                {"name": i, "id": i} for i in df_with_index.columns
                            ],
                            page_size=10,
                            filter_action="native",
                            filter_options={
                                "placeholder_text": setText(translationsCommon, 'filter', 'dashboard.data.common')},
                            sort_action="native",
                            sort_mode="multi",
                        )
                    ],
                    className="rules-table",
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            data=original_df_with_index.to_dict("records"),
                            columns=[
                                {"name": i, "id": i}
                                for i in original_df_with_index.columns
                            ],
                            page_size=10,
                            filter_action="native",
                            filter_options={
                                "placeholder_text": setText(translationsCommon, 'filter', 'dashboard.data.common')},
                            sort_action="native",
                            sort_mode="multi",
                        )
                    ],
                    className="rules-table",
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            data=descriptions_df.to_dict("records"),
                            columns=[
                                {"name": i, "id": i}
                                for i in descriptions_df.columns
                            ],
                            page_size=10,
                            sort_action="native",
                            sort_mode="multi",
                        )
                    ],
                    className="rules-table no_center",
                ),
                [
                    dcc.Dropdown(
                        id="n-vars-dropdown",
                        value=n_vars_names[0],
                        options=[{'label': name, 'value': name} for name in n_vars_names],
                        clearable=False,
                        className='predictor-selector',
                    ),
                    dbc.Col(
                        id=f"n-graph",
                        children=dcc.Graph(
                            figure=addAxisNames(
                                go.Figure(
                                    data=numeric_graph["graph_data"],
                                    layout=dict(title=numeric_graph["predictor"]),
                                ),
                                axisTranslations
                            )
                        ),
                    )
                ] if numeric_graph else no_data_text,
                [
                    dcc.Dropdown(
                        id="q-vars-dropdown",
                        value=q_vars_names[0],
                        options=[{'label': name, 'value': name} for name in q_vars_names],
                        clearable=False,
                        className='predictor-selector',
                    ),
                    dbc.Col(
                        id=f"q-graph",
                        children=dcc.Graph(
                            figure=addAxisNames(
                                go.Figure(
                                    data=qualitative_graph["graph_data"],
                                    layout=dict(title=qualitative_graph["predictor"]),
                                ),
                                axisTranslations
                            )
                        ),
                    )
                ] if qualitative_graph else no_data_text,
                dcc.Graph(
                    figure=setBottomLegend(
                        go.Figure(
                            data=go.Heatmap(
                                z=corr_matrix,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                text=round(corr_matrix, 2),
                                texttemplate="%{text}",
                            ),
                        )
                    )
                ),
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
