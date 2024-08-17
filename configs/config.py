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

    # # PostgreSQL database
    # SQLALCHEMY_DATABASE_URI = "postgresql://{}:{}@{}:{}/{}".format(
    #     os.environ.get("FET_DATABASE_USER", "gentelella"),
    #     os.environ.get("FET_DATABASE_PASSWORD", "gentelella"),
    #     os.environ.get("FET_DATABASE_HOST", "db"),
    #     os.environ.get("FET_DATABASE_PORT", 5432),
    #     os.environ.get("FET_DATABASE_NAME", "gentelella"),
    # )
    # PostgreSQL database
    SQLALCHEMY_DATABASE_URI = "sqlite://{}:{}@{}:{}/{}".format(
        os.environ.get("FET_DATABASE_USER", "admin"),
        os.environ.get("FET_DATABASE_PASSWORD", "admin"),
        os.environ.get("FET_DATABASE_HOST", "db"),
        os.environ.get("FET_DATABASE_PORT", 5432),
        os.environ.get("FET_DATABASE_NAME", "fet_database"),
    )


class DebugConfig(Config):
    DEBUG = True


config_dict = {"Production": ProductionConfig, "Debug": DebugConfig}
