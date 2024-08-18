from . import blueprint
from flask import render_template, request
from flask_login import login_required
from Dashboard import regression_dashboard, classification_dashboard


@blueprint.route('/regression')
@login_required
def regression_template():
    return render_template('dashboard.html', dash_url=regression_dashboard.url_base + f"?model_id={request.args['model_id']}")


@blueprint.route('/classification')
@login_required
def classification_template():
    return render_template('dashboard.html', dash_url=classification_dashboard.url_base + f"?model_id={request.args['model_id']}")


@blueprint.route('/classifier/comparison')
@login_required
def comparison_classifier_template():
    return render_template('comparison_dashboard.html', dash_url1=classification_dashboard.url_base + f"?model_id={request.args['model_1_id']}", dash_url2=classification_dashboard.url_base + f"?model_id={request.args['model_2_id']}")


@blueprint.route('/regression/comparison')
@login_required
def comparison_regression_template():
    return render_template('comparison_dashboard.html', dash_url1=regression_dashboard.url_base + f"?model_id={request.args['model_1_id']}", dash_url2=regression_dashboard.url_base + f"?model_id={request.args['model_2_id']}")
