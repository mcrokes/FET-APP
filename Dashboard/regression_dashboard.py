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
from Dashboard.metrics.metricsLayout import metricsCallbacks, metricsRegressorLayout
from Dashboard.predictions.predictionsLayout import (
    predictionsCallbacks,
    predictionsLayout,
)
from Dashboard.surrogate.surrogateLayout import surrogateCallbacks, surrogateLayout
from Dashboard.specificTrees.specificTreesLayout import (
    specificTreesCallbacks,
    specificTreesLayout,
)
from app.API.routes import find_translations
from app.API.utils import findTranslationsParent, setText

from .Dash_fun import apply_layout_with_auth
import dash_bootstrap_components as dbc
from furl import furl

url_base = "/dash/regression_dashboard/"

holder = html.Plaintext("No se ha insertado ninugún modelo")


def setTooltip(innerText, id):
    tooltip = dbc.Tooltip(
        html.Plaintext(innerText),
        target=id,
        className="personalized-tooltip",
    )

    return tooltip


def createLayout(currentLanguage):
    translations = find_translations(currentLanguage, ['dashboard'])['text']

    tab0_content = dbc.Card(
        dbc.CardBody([html.Div([datasetLayout(findTranslationsParent(translations, 'data'))],
                               id="dataset-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab1_content = dbc.Card(
        dbc.CardBody([html.Div([importancesLayout(findTranslationsParent(translations, 'importance'))],
                               id="importance-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab2_content = dbc.Card(
        dbc.CardBody([html.Div([metricsRegressorLayout(findTranslationsParent(translations, 'metrics'))],
                               id="graph-metrics-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab3_content = dbc.Card(
        dbc.CardBody([html.Div([surrogateLayout(findTranslationsParent(translations, 'surrogate'))],
                               id="surrogate-layout-output-upload")]),
        className="mt-3 section-card",
    )

    tab4_content = dbc.Card(
        dbc.CardBody(
            [html.Div([specificTreesLayout(findTranslationsParent(translations, 'trees'))],
                      id="specificTrees-layout-output-upload")]
        ),
        className="mt-3 section-card",
    )

    tab5_content = dbc.Card(
        dbc.CardBody(
            [html.Div([predictionsLayout(findTranslationsParent(translations, 'predictions'))],
                      id="tryit-yourself-layout-output-upload")]
        ),
        className="mt-3 section-card",
    )

    translationsTabs = findTranslationsParent(translations, 'tabs')
    translationsDataTab = findTranslationsParent(translationsTabs, 'data')
    translationsImportanceTab = findTranslationsParent(translationsTabs, 'importance')
    translationsMetricsTab = findTranslationsParent(translationsTabs, 'metrics')
    translationsSurrogateTab = findTranslationsParent(translationsTabs, 'surrogate')
    translationsTreesTab = findTranslationsParent(translationsTabs, 'trees')
    translationsPredictionsTab = findTranslationsParent(translationsTabs, 'predictions')

    tabs = dbc.Tabs(
        [
            dbc.Tab(
                [
                    tab0_content,
                    setTooltip(setText(translationsDataTab, 'tooltip', 'dashboard.tabs.data'),
                               "data-tooltip-id"),
                ],
                id="data-tooltip-id",
                label=setText(translationsDataTab, 'title', 'dashboard.tabs.data'),
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab1_content,
                    setTooltip(setText(translationsImportanceTab, 'tooltip', 'dashboard.tabs.importance'),
                               "importance-tooltip-id"),
                ],
                id="importance-tooltip-id",
                label=setText(translationsImportanceTab, 'title', 'dashboard.tabs.importance'),
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab2_content,
                    setTooltip(setText(translationsMetricsTab, 'tooltip', 'dashboard.tabs.metrics'),
                               "metrics-tooltip-id"),
                ],
                id="metrics-tooltip-id",
                label=setText(translationsMetricsTab, 'title', 'dashboard.tabs.metrics'),
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab3_content,
                    setTooltip(setText(translationsSurrogateTab, 'tooltip', 'dashboard.tabs.surrogate'),
                               "surrogate-tooltip-id"),
                ],
                id="surrogate-tooltip-id",
                label=setText(translationsSurrogateTab, 'title', 'dashboard.tabs.surrogate'),
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab4_content,
                    setTooltip(setText(translationsTreesTab, 'tooltip', 'dashboard.tabs.trees'),
                               "trees-tooltip-id"),
                ],
                id="trees-tooltip-id",
                label=setText(translationsTreesTab, 'title', 'dashboard.tabs.trees'),
                className="classifier-tab",
            ),
            dbc.Tab(
                [
                    tab5_content,
                    setTooltip(setText(translationsPredictionsTab, 'tooltip', 'dashboard.tabs.predictions'),
                               "predictions-tooltip-id"),
                ],
                id="predictions-tooltip-id",
                label=setText(translationsPredictionsTab, 'title', 'dashboard.tabs.predictions'),
                className="classifier-tab",
            ),
        ],
        id="classifier-tabs",
    )

    return html.Div(
        [tabs],
        id="regressor-tabs-container",
        style={"width": "100%"},
    )


def addCallbacks(app):
    datasetCallbacks(app, furl, True)
    importancesCallbacks(app, furl, True)
    metricsCallbacks(app, furl, True)
    surrogateCallbacks(app, furl, True)
    specificTreesCallbacks(app, furl, True)
    predictionsCallbacks(app, furl, True)


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
