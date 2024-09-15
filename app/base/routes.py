from flask import jsonify, render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

from ..extensions import db, login_manager
from . import blueprint
from .forms import LoginForm, CreateAccountForm
from .models import User


@blueprint.route('/')
def route_default():
    return redirect(url_for('base_blueprint.login'))


@blueprint.route('/page_<error>')
def route_errors(error):
    return render_template('errors/page_{}.html'.format(error))


## Login & Registration


@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:
        user = User.query.filter_by(username=request.form['username']).first()
        if user:
            if user.checkpw(request.form['password']):
                login_user(user)
                return redirect(url_for('home_blueprint.index'))
            else:
                status = 'Password Error !'
        else:
            status = "User doesn't exist !"
        return render_template('login/login.html', login_form=login_form, status=status)

    if current_user.is_authenticated:
        return redirect(url_for('home_blueprint.index'))
    return render_template('login/login.html', login_form=login_form, status='')


@blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('base_blueprint.login'))


# @blueprint.route('/shutdown')
# def shutdown():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()
#     return 'Server shutting down...'


## Errors


@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('errors/page_403.html'), 403
