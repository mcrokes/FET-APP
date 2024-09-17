from dotenv import load_dotenv
from flask_migrate import Migrate
from configs.config import config_dict
from app import create_app, db
import os
import sys

load_dotenv()

get_config_mode = os.environ.get("FET_CONFIG_MODE", "Debug")

try:
    config_mode = config_dict[get_config_mode.capitalize()]
except KeyError:
    sys.exit("Error: Invalid FET_CONFIG_MODE environment variable entry.")

app = create_app(config_mode)
Migrate(app, db)

if __name__ == "__main__":
    # with app.app_context():
    #    db.create_all()
    app.run()
    # app.run(debug=True)
