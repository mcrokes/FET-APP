# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: mcrokes
"""
from dash import Dash, html

from Dashboard.dataset.datasetLayout import datasetCallbacks, datasetLayout
from Dashboard.importances.importancesLayout import (
    importancesCallbacks,
    importancesLayout,
)
from Dashboard.metrics.metricsLayout import metricsCallbacks, metricsClassifierLayout
from Dashboard.predictions.predictionsLayout import (
    predictionsCallbacks,
    predictionsLayout,
)
from Dashboard.surrogate.surrogateLayout import surrogateCallbacks, surrogateLayout
from Dashboard.specificTrees.specificTreesLayout import (
    specificTreesCallbacks,
    specificTreesLayout,
)

from .Dash_fun import apply_layout_with_auth, load_object, save_object
import dash_bootstrap_components as dbc
from furl import furl

from app.API.routes import find_translations

url_base = "/dash/classification_dashboard/"

holder = html.Plaintext("No se ha insertado ninugún modelo")


def setTooltip(innerText, id):
    tooltip = dbc.Tooltip(
        html.Plaintext(innerText),
        target=id,
        className="personalized-tooltip",
    )

    return tooltip


def createLayout(currentLanguage):
    print('currentLanguage: ', currentLanguage)
    translations = find_translations(currentLanguage, ['dashboard'])['text']
    tab0_content = dbc.Card(
        dbc.CardBody([html.Div([datasetLayout(translations.get('data') if translations.get('data') else {})],
                               id="dataset-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab1_content = dbc.Card(
        dbc.CardBody([html.Div([importancesLayout(translations.get('importance') if translations.get('data') else {})], id="importance-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab2_content = dbc.Card(
        dbc.CardBody([html.Div([metricsClassifierLayout], id="graph-metrics-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab3_content = dbc.Card(
        dbc.CardBody([html.Div([surrogateLayout], id="surrogate-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab4_content = dbc.Card(
        dbc.CardBody(
            [html.Div([specificTreesLayout], id="specificTrees-layout-output-upload")]
        ),
        className="mt-3 section-card",
    )

    tab5_content = dbc.Card(
        dbc.CardBody(
            [html.Div([predictionsLayout], id="tryit-yourself-layout-output-upload")]
        ),
        className="mt-3 section-card",
    )

    print(translations)

    translationsTabs = translations.get('tabs')
    translationsDataTab = translationsTabs.get('data') if translationsTabs else {}
    translationsImportanceTab = translationsTabs.get('importance') if translationsTabs else {}
    translationsMetricsTab = translationsTabs.get('metrics') if translationsTabs else {}
    translationsSurrogateTab = translationsTabs.get('surrogate') if translationsTabs else {}
    translationsTreesTab = translationsTabs.get('trees') if translationsTabs else {}
    translationsPredictionsTab = translationsTabs.get('predictions') if translationsTabs else {}

    tabs = dbc.Tabs(
        [
            dbc.Tab(
                [
                    tab0_content,
                    setTooltip(
                        translationsDataTab.get('tooltip') if translationsDataTab else 'dashboard.tabs.data'
                                                                                       '.tooltip',
                        "data-tooltip-id"),
                ],
                id="data-tooltip-id",
                label=translationsDataTab.get('title') if translationsDataTab else 'dashboard.tabs.data.title',
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab1_content,
                    setTooltip(
                        translationsImportanceTab.get('tooltip') if translationsImportanceTab else 'dashboard.tabs.data'
                                                                                                   '.tooltip',
                        "importance-tooltip-id"),
                ],
                id="importance-tooltip-id",
                label=translationsImportanceTab.get(
                    'title') if translationsImportanceTab else 'dashboard.tabs.importance.title',
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab2_content,
                    setTooltip(
                        translationsMetricsTab.get('tooltip') if translationsMetricsTab else 'dashboard.tabs.data'
                                                                                             '.tooltip',
                        "metrics-tooltip-id"),
                ],
                id="metrics-tooltip-id",
                label=translationsMetricsTab.get('title') if translationsMetricsTab else 'dashboard.tabs.metrics.title',
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab3_content,
                    setTooltip(
                        translationsSurrogateTab.get('tooltip') if translationsSurrogateTab else 'dashboard.tabs.data'
                                                                                                 '.tooltip',
                        "surrogate-tooltip-id"),
                ],
                id="surrogate-tooltip-id",
                label=translationsSurrogateTab.get(
                    'title') if translationsSurrogateTab else 'dashboard.tabs.surrogate.title',
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab4_content,
                    setTooltip(translationsTreesTab.get('tooltip') if translationsTreesTab else 'dashboard.tabs.data'
                                                                                                '.tooltip',
                               "trees-tooltip-id"),
                ],
                id="trees-tooltip-id",
                label=translationsTreesTab.get('title') if translationsTreesTab else 'dashboard.tabs.trees.title',
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab5_content,
                    setTooltip(translationsPredictionsTab.get(
                        'tooltip') if translationsPredictionsTab else 'dashboard.tabs.data'
                                                                      '.tooltip',
                               "predictions-tooltip-id"),
                ],
                id="predictions-tooltip-id",
                label=translationsPredictionsTab.get(
                    'title') if translationsPredictionsTab else 'dashboard.tabs.data.title',
                className="classifier-tab",
            ),
        ],
        id="classifier-tabs",
    )
    return html.Div(
        [tabs],
        id="classifier-tabs-container",
        style={"width": "100%"},
    )


def addCallbacks(app):
    # print('currentLanguage: ', currentLanguage)
    # translations = find_translations(currentLanguage, ['dashboard'])['text']
    datasetCallbacks(app, furl)
    importancesCallbacks(app, furl)
    metricsCallbacks(app, furl)
    surrogateCallbacks(app, furl)
    specificTreesCallbacks(app, furl)
    predictionsCallbacks(app, furl)


def Add_Dash(server):
    app = Dash(
        server=server,
        url_base_pathname=url_base,
        external_stylesheets=[
            "/static/assets/CYBORG/bootstrap.min.css",
            "/static/assets/fontawesome-free/css/all.min.css",
            "/static/assets/styles.css",
        ],
    )
    apply_layout_with_auth(app, createLayout, addCallbacks)

    return app.server
