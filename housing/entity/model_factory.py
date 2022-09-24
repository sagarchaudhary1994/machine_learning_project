from ast import Mod
import os
import statistics
import sys
from housing.logger import logging
from housing.exception import Housing_Exception
from collections import namedtuple
from typing import List
import numpy as np
import yaml
import importlib

GRID_SEARCH_KEY = "grid_search"
MODULE_KEY = "module"
CLASS_KEY = "class"
PARAM_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number",
                                     "model",
                                     "param_grid_search",
                                     "model_name"])

GridSearchBestModel = namedtuple("GridSearchBestModel",
                                 ["model_serial_number",
                                  "model",
                                  "best_model",
                                  "best_parameters",
                                  "best_score"])

BestModel = namedtuple("BestModel",
                       ["model_serial_number",
                        "model",
                        "best_model",
                        "best_parameters",
                        "best_score"])

"""
grid_search:
    module: sklearn.model_selection
    class: GridSearchCV
    params:
    cv: 3
    verbose: 1

model_selection:
    model_0:
        module: sklearn.tree
        class: DecisionTreeRegressor
        params:
            criterion: squared_error
            min_samples_leaf: 2
        search_param_grid:
            max_depth:
                - 2
                - 3
"""


def get_sample_model_config_yaml_file(export_dir: str) -> str:
    """
    saves the sample yaml file to get the configurations for model training

    Return: Sample YAML file path
    export_dir: location to save sample model configuration file path

    """
    try:
        model_configuration = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }
            },

            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_module",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY: {
                        "param_name1": "value1",
                        "param_name2": "value2"
                    },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ["param_value_1", "param_value_2"]
                    }
                }
            }
        }

        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")

        with open(export_file_path, "wb") as file_obj:
            yaml.dump(model_configuration, file_obj)

        return export_file_path
    except Exception as e:
        raise Housing_Exception(e, sys)


class ModelFactory:

    def __init__(self, model_config_path: str = None):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_properties: dict = self.config[GRID_SEARCH_KEY][PARAM_KEY]
            self.models_initialization_config: dict = self.config[MODEL_SELECTION_KEY]

            self.initialized_model_list = None
            self.grid_search_best_model_list = None
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as file_obj:
                return yaml.safe_load(file_obj)
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            # load the module, will raise error if module cannot be loaded
            module = importlib.import_module(module_name)

            # get the class, will raise Attribute error if class cannot be found
            class_ref = getattr(module, class_name)
            return class_ref

        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception(
                    "property_data parameter required to be dictionary")

            for key, value in property_data.items():
                setattr(instance_ref, key, value)

            return instance_ref
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail,
                                      input_feature, output_feature) -> GridSearchBestModel:
        """
        Function will perform grid search to find the best model parameters
            - estimator: Model Object
            - param_grid: dictionary of parameter to perform search operation
            - input feature: Input feature for model
            - output_feature: Target/dependent feature

        return: Function return GridSearchBestModel object
        """
        try:
            # instantiating GridSearchCv class
            message = "*" * \
                50, f"training {type(initialized_model.model).__name__}", "*"*50
            logging.info(message)

            grid_search_cv_ref = ModelFactory.class_for_name(self.grid_search_cv_module,
                                                             self.grid_search_class_name)

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)

            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_properties)

            grid_search_cv.fit(input_feature, output_feature)

            grid_search_best_model = GridSearchBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_)

            return grid_search_best_model

        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_initialized_model_list(self,) -> List[InitializedModelDetail]:
        """
        Function will return a list of model details
        """
        try:
            initialized_model_list = []

            for model_serial_number in self.models_initialization_config.keys():
                model_intitialization_config = self.models_initialization_config[
                    model_serial_number]

                model_object_ref = ModelFactory.class_for_name(
                    module_name=model_intitialization_config[MODULE_KEY],
                    class_name=model_intitialization_config[CLASS_KEY])

                # Initialized the model
                model = model_object_ref()

                if PARAM_KEY in model_intitialization_config:
                    model_obj_property_data = dict(
                        model_intitialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(
                        instance_ref=model,
                        property_data=model_obj_property_data
                    )
                param_grid_search = model_intitialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_intitialization_config[MODULE_KEY]}.{model_intitialization_config[CLASS_KEY]}"

                model_intitialization_config = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model=model,
                    param_grid_search=param_grid_search,
                    model_name=model_name
                )

                initialized_model_list.append(model_intitialization_config)
            self.initialized_model_list = initialized_model_list

            return self.initialized_model_list

        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self,
                                                             intitialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchBestModel:
        """
        Function will perform the Grid Search operation to find the best params
        for inititalised model

        Params:
        initialized_model: Model for which best params to find
        input_feature: Dependent Feature of dataset
        output_feature: Independent Feature of dataset

        return: returns the GridSearchBestModel object 
        """
        try:
            return self.execute_grid_search_operation(
                initialized_model=InitializedModelDetail,
                input_feature=input_feature,
                output_feature=output_feature
            )
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_models_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchBestModel]:
        """
        This function performs the Grid Search operation to find best params for each of
        the model initialized

        Input:
        initialized_models_list: List of all the models initialized in the configuration file
        input_feature: independent features
        output_feature: Target feature

        Return:
        returns the GridSearchModel object for each of the initialised model in List
        """
        try:
            self.grid_search_best_model_list = []

            for intialized_model in initialized_models_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    intitialized_model=intialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_search_best_model_list.append(
                    grid_searched_best_model)

            return self.grid_search_best_model_list
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(self,
                                                          grid_searched_best_model_list: List[GridSearchBestModel],
                                                          base_accuracy: float = 0.6) -> BestModel:
        """
        Function returns the best model from list of Grid Searched best models
        Input:
        grid_searched_best_model: List of Grid Searched Best Models
        base_aacuracy: Threshold accuracy for any model to considered for comparision  
                        with other models in the list

        Return: returns the best model out of all the Grid Searched Best Models
        """
        try:
            best_model: BestModel = None
            for grid_searched_model in grid_searched_best_model_list:
                if(grid_searched_model.best_score > base_accuracy):
                    base_accuracy = grid_searched_model.best_score
                    best_model = grid_searched_model
                    logging.info(f"Acceptable model found: [ {best_model} ]")
            if not best_model:
                raise Exception(
                    f"None of the model has the base accuracy: [{base_accuracy}]")
            logging.info(f"Best Model: [ {best_model} ]")

            return best_model
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        """
        Function returns the best model from list of initialised models

        Input:
        X : Input feature
        y: Output feature
        base_accuracy: Threshold accuracy for model
        """
        try:
            logging.info(f"Started initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(
                f"Intialized model list: [ {initialized_model_list} ]")

            grid_search_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_models_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )

            get_best_model = ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_search_best_model_list,
                base_accuracy=base_accuracy
            )

            return get_best_model
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def get_model_details(model_list: List[InitializedModelDetail],
                          model_serial_number: str) -> InitializedModelDetail:
        """
        Function returns the Model details from list of models initialized
        using the model serial number

        Input:
        model_details: List of models initialized
        model_serial_number: Model serial number as defined in model.yaml config file
        """
        try:
            for model in model_list:
                if model.model_serial_number == model_serial_number:
                    return model
        except Exception as e:
            raise Housing_Exception(e, sys) from e
