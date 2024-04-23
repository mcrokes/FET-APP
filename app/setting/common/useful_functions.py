from flask import current_app
from flask_login import current_user


def check_admin():
    admin_user = current_app.config["ADMIN"]["username"]
    if current_user.username == admin_user:
        return True
    return False
