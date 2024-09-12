from flask_wtf import FlaskForm
from wtforms import EmailField, StringField, PasswordField

## login and registration


class add_user_Form(FlaskForm):
    username = StringField(id="username-create")
    email = EmailField(id="email-create")
    password = PasswordField(id="pwd-create")


class delete_user_Form(FlaskForm):
    username = StringField(id="username-delete")


class setting_password_Form(FlaskForm):
    password = PasswordField("Contraseña", id="pwd-setting")


class change_password_Form(FlaskForm):
    origin_password = PasswordField("Contraseña Original", id="origin-password")
    new_password = PasswordField("Nueva Contraseña", id="new-password")
    new_password2 = PasswordField("Repite la Nueva Contraseña", id="new-password2")
