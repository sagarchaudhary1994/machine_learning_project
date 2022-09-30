from genericpath import exists, isfile
from flask import Flask, request, abort, send_file, render_template
from housing.logger import logging
from housing.exception import Housing_Exception
import sys
import os
from constant import *
from housing.entity.housing_predictor import HousingData, HousingPredictor
from test import call_training_pipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        raise Housing_Exception(e, sys) from e


@app.route("/train", methods=['GET', 'POST'])
def train():
    try:
        call_training_pipeline()
        return render_template('train.html')
    except Exception as e:
        raise Housing_Exception(e, sys) from e


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        context = {
            HOUSING_DATA_KEY: None,
            MEDIAN_HOUSING_VALUE_KEY: None
        }
        if request.method == "POST":
            print(request.form)
            longitude = float(request.form['longitude'])
            latitude = float(request.form['latitude'])
            housing_median_age = float(request.form["housingmedianage"])
            total_rooms = float(request.form["totalrooms"])
            total_bedrooms = float(request.form["totalbedrooms"])
            population = float(request.form["population"])
            households = float(request.form["households"])
            median_income = float(request.form["medianincome"])
            ocean_proximity = str(request.form["oceanproximity"])

            housing_data = HousingData(
                longitude=longitude,
                latitude=latitude,
                housing_median_age=housing_median_age,
                total_rooms=total_rooms,
                total_bedrooms=total_bedrooms,
                population=population,
                households=households,
                median_income=median_income,
                ocean_proximity=ocean_proximity,
            )

            housing_df = HousingData.get_input_dataframe(housing_data)
            housing_predictor = HousingPredictor(model_dir=MODEL_DIR)
            median_housing_value = housing_predictor.predict(housing_df)

            context = {
                HOUSING_DATA_KEY: housing_data.get_housing_data_as_dict(),
                MEDIAN_HOUSING_VALUE_KEY: median_housing_value
            }
            return render_template('predict.html', context=context)
        return render_template('predict.html', context=context)
    except Exception as e:
        raise Housing_Exception(e, sys) from e


@app.route('/saved_models', defaults={'req_path': 'housing/artifact/saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models(req_path):
    try:
        os.makedirs(os.path.join("housing", "artifact",
                    "saved_models"), exist_ok=True)
        # Joining the base and req_path
        print(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        print(abs_path)
        # Return 404 if path does not exist
        if not os.path.exists(abs_path):
            return abort(404)

        # Check is path is file and serve it

        if os.path.isfile(abs_path):
            return send_file(abs_path)

        # show directory content
        files = {os.path.join(abs_path, file)                 : file for file in os.listdir(abs_path)}

        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }

        return render_template("saved_models_files.html", result=result)
    except Exception as e:
        raise Housing_Exception(e, sys) from e


@app.route('/housing_logs', defaults={'req_path': 'housing_logs'})
@app.route('/housing_logs/<path:req_path>')
def logs(req_path):
    try:
        os.makedirs("housing_logs", exist_ok=True)
        # Joining the base and req_path
        print(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        print(abs_path)

        # Return 404 if path does not exist
        if not os.path.exists(abs_path):
            return abort(404)

        # Check is path is file and serve it

        if os.path.isfile(abs_path):
            return send_file(abs_path)

        # show directory content
        files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }

        return render_template("log_files.html", result=result)
    except Exception as e:
        raise Housing_Exception(e, sys) from e


@app.route('/artifact', defaults={'req_path': 'housing/artifact'})
@app.route('/artifact/<path:req_path>')
def artifact(req_path):
    try:
        os.makedirs("housing", exist_ok=True)
        # Joining the base and req_path
        print(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        print(abs_path)

        # Return 404 if path does not exist
        if not os.path.exists(abs_path):
            return abort(404)

        # Check is path is file and serve it

        if os.path.isfile(abs_path):
            if ".html" in abs_path:
                with open(abs_path, "r", encoding="utf-8") as file:
                    content = ""
                    for line in file.readlines():
                        content = f"{content}{line}"
                    return content

        # show directory content
        files = {os.path.join(abs_path, file)
                              : file for file in os.listdir(abs_path)}

        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }

        return render_template("files.html", result=result)
    except Exception as e:
        raise Housing_Exception(e, sys) from e


if __name__ == "__main__":
    app.run(debug=True)
