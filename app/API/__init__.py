from flask import Blueprint

blueprint = Blueprint(
    'API_blueprint',
    __name__,
    url_prefix='/INTERNAL_API',
    template_folder='templates',
    static_folder='static'
)
