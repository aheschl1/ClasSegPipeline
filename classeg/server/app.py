from flask import Flask
from flask_cors import CORS, cross_origin

from classeg.server.dataset_queries.read_available import get_available_datasets
from classeg.server.utils.constants import DEFAULT_PROJECT
from classeg.server.utils.project import Project

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/projects')
@cross_origin()
def get_projects():
    datasets = get_available_datasets()
    projects = [Project(dataset).to_dict() for dataset in datasets]
    return projects


@app.route('/projects/<project_id>')
@cross_origin()
def get_project(project_id):
    project = Project(project_id)
    return project.to_dict()
