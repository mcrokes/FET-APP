from flask_wtf import FlaskForm
from wtforms import FileField, StringField, TextAreaField


## login and registration


class add_model(FlaskForm):
    name = StringField(id="model-name", default='')
    model = FileField(id="model", render_kw={'accept': ".joblib"})
    description = TextAreaField(id="model-description", default='')
    dataset = FileField(
        description="Data Set",
        id="model-data-set",
        render_kw={'accept': ".csv, .xlsx"}
    )
