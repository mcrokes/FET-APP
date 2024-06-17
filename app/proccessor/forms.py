from flask_wtf import FlaskForm
from wtforms import FileField, StringField, TextAreaField

## login and registration


class add_classifier(FlaskForm):
    name = StringField("Model Name", id="model-name")
    model = FileField("Model", id="model", render_kw={'accept': ".joblib"})
    description = TextAreaField("Description", id="model-description")    
    dataset = FileField("Data Set", id="model-data-set", render_kw={'accept': ".csv"})