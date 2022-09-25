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


if __name__ == "__main__":
    app.run(debug=True)
