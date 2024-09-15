from flask import Blueprint

blueprint = Blueprint(
    'processor_blueprint',
    __name__,
    url_prefix='/processor',
    template_folder='templates',
    static_folder='static'
)
