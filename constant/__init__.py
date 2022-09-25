import os
import sys

# Constant for app.py

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "housing"
PIPELINE_ARTIFACT_DIR_NAME = "artifact"
SAVED_MODEL_DIR_NAME = "saved_model"

LOG_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME,
                         PIPELINE_ARTIFACT_DIR_NAME, SAVED_MODEL_DIR_NAME)

HOUSING_DATA_KEY = "housing_data"
MEDIAN_HOUSING_VALUE_KEY = "median_house_value"
