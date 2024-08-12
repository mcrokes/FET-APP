from flask_wtf import FlaskForm
from wtforms import FileField, StringField, TextAreaField


## login and registration


class add_model(FlaskForm):
    name = StringField("Model Name", id="model-name", default='')
    model = FileField("Model", id="model", render_kw={'accept': ".joblib"})
    description = TextAreaField("Description", id="model-description", default='')
    dataset = FileField(
        "Conjunto de Datos",
        description="Data Set",
        id="model-data-set",
        render_kw={'accept': ".csv, .xlsx"}
    )
