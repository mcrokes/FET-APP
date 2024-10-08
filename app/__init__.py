from flask import Flask, url_for, redirect
from flask_login import current_user
from .extensions import db, login_manager
from importlib import import_module
from .base.models import User
from Dashboard import regression_dashboard, classification_dashboard
from os import path, environ
import logging


def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)


def register_blueprints(app):
    for module_name in ("base", "home", "dashboard", "setting", "processor", "API"):
        module = import_module("app.{}.routes".format(module_name))
        app.register_blueprint(module.blueprint)


def configure_database(app):
    if not app._got_first_request:
        with app.app_context():
            db.create_all()
            # admin_username = app.config["ADMIN"]["username"]

            user = User.query.filter_by(is_admin=True).first()
            if not user:
                User(**app.config["ADMIN"]).add_to_db()
                # user.delete_from_db()

            # user.delete_from_db()
            # User(
            #     **{
            #         "username": "pepe",
            #         "email": "pepe@gmail.com",
            #         "password": "pepe",
            #     }
            # ).add_to_db()
            # User(
            #     **{
            #         "username": "mario",
            #         "email": "mario@gmail.com",
            #         "password": "mario",
            #     }
            # ).add_to_db()

    # @app.before_first_request
    # def initialize_database():
    #     db.create_all()
    #     admin_username = app.config["ADMIN"]["username"]
    #     user = User.query.filter_by(username=admin_username).first()
    #     if user:
    #         user.delete_from_db()
    #     User(**app.config["ADMIN"]).add_to_db()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()


def configure_logs(app):
    # for combine gunicorn logging and flask built-in logging module
    if __name__ != "__main__":
        gunicorn_logger = logging.getLogger("gunicorn.error")
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    # endif


def apply_themes(app):
    """
    Add support for themes.

    If DEFAULT_THEME is set then all calls to
      url_for('static', filename='')
      will modfify the url to include the theme name

    The theme parameter can be set directly in url_for as well:
      ex. url_for('static', filename='', theme='')

    If the file cannot be found in the /static/<theme>/ lcation then
      the url will not be modified and the file is expected to be
      in the default /static/ location
    """

    @app.context_processor
    def override_url_for():
        Is_admin = (
                current_user.is_authenticated
                and current_user.username == app.config["ADMIN"]["username"]
        )
        return dict(url_for=_generate_url_for_theme, Is_admin=Is_admin)

    def _generate_url_for_theme(endpoint, **values):
        if endpoint.endswith("static"):
            themename = values.get("theme", None) or app.config.get(
                "DEFAULT_THEME", None
            )
            if themename:
                theme_file = "{}/{}".format(themename, values.get("filename", ""))
                if path.isfile(path.join(app.static_folder, theme_file)):
                    values["filename"] = theme_file
        return url_for(endpoint, **values)


def configure_file_upload(app):
    app.config['UPLOAD_FOLDER'] = environ.get('UPLOAD_FOLDER')


def handle_403(e):
    print('ERROR: ', e)
    return redirect('/page_403')


def handle_404(e):
    print('ERROR: ', e)
    return redirect('/page_404')


def handle_500(e):
    print('ERROR: ', e)
    return redirect('/page_500')


def create_app(config, selenium=False):
    app = Flask(__name__, static_folder="../configs/static")
    app.config.from_object(config)
    if selenium:
        app.config["LOGIN_DISABLED"] = True
    configure_file_upload(app)
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)
    configure_logs(app)
    apply_themes(app)
    app = regression_dashboard.Add_Dash(app)
    app = classification_dashboard.Add_Dash(app)
    app.config['TRAP_HTTP_EXCEPTIONS'] = True
    app.register_error_handler(403, handle_403)
    app.register_error_handler(404, handle_404)
    app.register_error_handler(500, handle_500)
    return app
