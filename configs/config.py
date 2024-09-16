import os


class Config(object):
    SECRET_KEY = "key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ADMIN = {
        "username": "admin",
        "email": "admin",
        "password": "admin",
        "is_admin": True,
    }

    # THEME SUPPORT
    #  if set then url_for('static', filename='', theme='')
    #  will add the theme name to the static URL:
    #    /static/<DEFAULT_THEME>/filename
    # DEFAULT_THEME = "themes/dark"
    DEFAULT_THEME = None


class ProductionConfig(Config):
    DEBUG = False

    # SQL database
    SQLALCHEMY_DATABASE_URI = "sqlite://{}:{}@{}:{}/{}".format(
        os.environ.get("FET_DATABASE_USER", "admin"),
        os.environ.get("FET_DATABASE_PASSWORD", "admin"),
        os.environ.get("FET_DATABASE_HOST", "db"),
        os.environ.get("FET_DATABASE_PORT", 5432),
        os.environ.get("FET_DATABASE_NAME", "fet_database"),
    )


class DebugConfig(Config):
    DEBUG = True


config_dict = {"Production": DebugConfig, "Debug": DebugConfig}
