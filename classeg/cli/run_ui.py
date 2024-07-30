import logging
import os
import click
from multiprocessing import Process
import time

"""
export FLASK_ENV=development
export FLASK_APP=app.py
"""
CLIENT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'client')
APP_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui_server', 'app.py')


def _run_frontend():
    os.system(f"cd {CLIENT_ROOT}; npm start")


def _run_backend():
    os.system(f"flask run --port 3001")


@click.command()
@click.option('--install', help='Run npm install on the frontend react app.', is_flag=True)
@click.option('--server_only', help='Run only the server.', is_flag=True)
def main(install, server_only=False):
    os.environ['FLASK_ENV'] = 'development'
    os.environ["FLASK_APP"] = APP_ROOT
    if install:
        os.system(f"cd {CLIENT_ROOT}; npm install")

    server_p = Process(target=_run_backend)
    server_p.start()
    if not server_only:
        logging.info("Waiting for the server to start...")
        time.sleep(2)
        client_p = Process(target=_run_frontend)
        client_p.start()


if __name__ == '__main__':
    main()
