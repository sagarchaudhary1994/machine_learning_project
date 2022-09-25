from flask import Flask, request, abort, send_file, render_template
from housing.logger import logging
from housing.exception import Housing_Exception
import sys
import os
from housing.constant import *


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        raise Exception("We are testing exception module")
    except Exception as e:
        housing_obj = Housing_Exception(e, sys)
        logging.info(housing_obj.errr_message)
        logging.info("Testing logging module")
        return "<p>CICD is done for this project</p>"


if __name__ == "__main__":
    app.run(debug=True)
