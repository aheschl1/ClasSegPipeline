import os.path

from flask import Flask, send_file
from flask_cors import CORS, cross_origin

from classeg.server.dataset_queries.read_available import get_available_datasets
from classeg.server.utils.caching import cache_array_as_image, clear_cache
from classeg.server.utils.project import Project

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/projects')
@cross_origin()
def get_projects():
    clear_cache()
    datasets = get_available_datasets()
    projects = [Project(dataset).to_dict() for dataset in datasets]
    return projects


@app.route('/projects/<project_id>')
@cross_origin()
def get_project(project_id):
    project = Project(project_id)
    return project.to_dict()


@app.route('/projects/<project_id>/raw/<case>')
@cross_origin()
def get_raw(project_id, case):
    case = int(case)
    project = Project(project_id)
    return send_file(project.get_raw(case).im_path)


@app.route('/projects/<project_id>/preprocessed/<case>')
@cross_origin()
def get_preprocessed(project_id, case):
    case = int(case)
    project = Project(project_id)
    point = project.get_preprocessed(case)
    image, _ = point.get_data()
    path = cache_array_as_image(image)
    return send_file(path)


@app.route('/projects/<project_id>/experiments')
@cross_origin()
def get_experiments(project_id):
    project = Project(project_id)
    experiments = project.get_experiments()
    return experiments


@app.route('/projects/<project_id>/experiments/<experiment_id>')
@cross_origin()
def get_experiment(project_id, experiment_id):
    project = Project(project_id)
    return project.get_experiment(experiment_id)


@app.route("/README.md")
@cross_origin()
def get_readme():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return send_file(f"{repo_root}/README.md")