from flask import Flask
import os
import flask_cors
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)
flask_cors.CORS(app, supports_credentials=True)
executor = ThreadPoolExecutor(1)


@app.route("/train")
def trigger_train():
    executor.submit(train)
    return "Success"


def train():
    os.system("python ")