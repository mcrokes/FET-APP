from . import blueprint
from flask import g, render_template, request
from flask_login import login_required, current_user
from Dashboard import Dash_App1, Dash_App2

@blueprint.route('/app1')
@login_required
def app1_template():
    return render_template('app1.html', dash_url = Dash_App1.url_base)

@blueprint.route('/app2')
@login_required
def app2_template():
    print(request.args['model_id'])
    print(current_user)
    print(g)
    return render_template('app2.html', dash_url = Dash_App2.url_base + f"?user_id={request.args['model_id']}")