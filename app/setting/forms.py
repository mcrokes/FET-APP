from flask_wtf import FlaskForm
from wtforms import EmailField, StringField, PasswordField

## login and registration


class add_user_Form(FlaskForm):
    username = StringField("Nombre de Usuario", id="username_create")
    email = EmailField("Email")
    password = PasswordField("Contraseña", id="pwd_create")


class delete_user_Form(FlaskForm):
    username = StringField("Nombre de Usuario", id="username_delete")


class setting_password_Form(FlaskForm):
    password = PasswordField("Contraseña", id="pwd_setting")


class change_password_Form(FlaskForm):
    origin_password = PasswordField("Contraseña Original", id="origin_password")
    new_password = PasswordField("Nueva Contraseña", id="new_password")
    new_password2 = PasswordField("Repite la Nueva Contraseña", id="new_password2")
