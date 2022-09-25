from housing.exception import Housing_Exception
from housing.logger import logging
import os
import sys
from housing.component.data_transformation import DataTransformation
from housing.pipeline.pipeline import Pipeline
from housing.entity.model_factory import ModelFactory, get_sample_model_config_yaml_file


def call_training_pipeline():
    try:
        Pipeline().run_pipeline()
    except Exception as e:
        raise Housing_Exception(e, sys) from e
