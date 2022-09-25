import os
import sys
import pandas as pd
from housing.exception import Housing_Exception
from housing.logger import logging
from housing.util.util import load_object


class HousingData:

    def __init__(self,
                 longitude: float,
                 latitude: float,
                 housing_median_age: float,
                 total_rooms: float,
                 total_bedrooms: float,
                 population: float,
                 households: float,
                 median_income: float,
                 ocean_proximity: str,
                 median_house_value: float = None) -> None:
        try:
            self.longitude = longitude
            self.latitude = latitude
            self.housing_median_age = housing_median_age
            self.total_rooms = total_rooms
            self.total_bedrooms = total_bedrooms
            self.population = population
            self.households = households
            self.median_income = median_income
            self.ocean_proximity = ocean_proximity
            self.median_house_value = median_house_value
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_input_dataframe(self):
        try:
            input_data_dict = self.get_housing_data_as_dict()
            input_dataframe = pd.DataFrame(input_data_dict)
            return input_dataframe
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_housing_data_as_dict(self):
        try:
            input_data = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "total_bedrooms": [self.total_bedrooms],
                "total_rooms": [self.total_rooms],
                "population": [self.population],
                "households": [self.households],
                "housing_median_age": [self.housing_median_age],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity],
                "median_house_value": [self.median_house_value]
            }

            return input_data
        except Exception as e:
            raise Housing_Exception(e, sys) from e


class HousingPredictor:

    def __init__(self, model_dir: str) -> None:
        try:
            logging.info(f"{'='*20}Housing Predictor Logs Started. {'='*20}")
            self.model_dir = model_dir
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(
                self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            logging.info(f"Latest Model File Path- [ {latest_model_path} ]")
            return latest_model_path
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(model_path)
            median_house_value = model.predict(X)
            logging.info(
                f"Median House Value Prediction for data: [{X}] is [{median_house_value}]")
            return median_house_value[0]
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Housing Predictor Logs Completed. {'='*20}")
