from flask_wtf import FlaskForm
from wtforms import FileField, StringField, TextAreaField

## login and registration


class add_classifier(FlaskForm):
    name = StringField("Model Name", id="model-name")
    description = TextAreaField("Description", id="model-description")
    model = FileField("Model", id="model")
    dataset = FileField("Data Set", id="model-data-set")