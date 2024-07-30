import json
import logging
import os.path
import subprocess

from flask import Flask, send_file
from flask_cors import CORS, cross_origin

from classeg.ui_server.dataset_queries.read_available import get_available_datasets
from classeg.ui_server.utils.caching import cache_array_as_image, clear_cache
from classeg.ui_server.utils.project import Project
from classeg.ui_server.utils.utils import get_terminal_command

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


@app.route("/train/<project_id>/<experiment_name>/<fold>/<model>/<extension_name>/<config>", methods=["POST"])
@cross_origin()
def train(project_id, experiment_name, fold, model, extension_name, config):
    dataset_id = int(project_id.split("_")[-1])
    dataset_desc = None
    if project_id.count("_") > 1:
        dataset_desc = project_id.split("_")[1]
    fold = int(fold)
    model = model if model != "None" else None
    extension_name = extension_name if extension_name != "None" else None
    gpus = 1
    name = experiment_name

    command = f"classegTrain -d {dataset_id} -f {fold} -g {gpus} -c {config} -n {name}"
    if dataset_desc is not None:
        command += f" -dd {dataset_desc}"
    if model is not None:
        command += f" -m {model}"
    if extension_name is not None:
        command += f" -ext {extension_name}"

    logging.info(f"Training command: {command}")
    command = get_terminal_command(command)
    if command is None:
        return json.dumps({"error": "No terminal found"}), 500
    subprocess.Popen(command, shell=True)
    return json.dumps({'message': f'Process started successfully'}), 200




