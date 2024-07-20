from flask import Flask
from flask_cors import CORS, cross_origin

from classeg.server.utils.constants import DEFAULT_PROJECT

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/projects')
@cross_origin()
def get_projects():
    return [DEFAULT_PROJECT]
