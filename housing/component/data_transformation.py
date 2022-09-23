import os
import sys

from sklearn import preprocessing
from housing.exception import Housing_Exception
from housing.logger import logging
import pandas as pd
import numpy as np
from housing.entity.config_entity import DataTransformationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
import dill
from sklearn.base import BaseEstimator, TransformerMixin
from housing.util.util import read_yaml_file
from housing.constant import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from housing.util.util import save_object

COLUMN_TOTAL_ROOMS = "total_rooms"
COLUMN_POPULATION = "population"
COLUMN_HOUSEHOLDS = "households"
COLUMN_TOTAL_BEDROOM = "total_bedrooms"

#   longitude: float
#   latitude: float
#   housing_median_age: float
#   total_rooms: float
#   total_bedrooms: float
#   population: float
#   households: float
#   median_income: float
#   median_house_value: float
#   ocean_proximity: category
#   income_cat: float


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self,
                 add_bedrooms_per_room: bool = True,
                 total_room_ix: int = 3,
                 population_ix: int = 5,
                 households_ix: int = 6,
                 total_bedrooms_ix: int = 4, columns=None
                 ):
        """
        Feature Generator Initialization
        add_bedrooms_per_room: bool
        total_room_ix: int index number of total rooms column
        total_population_ix:int index number of total population column
        household_ix:int index number of household column
        total_bedrooms_ix:int index number of total_bedroom column
        """
        try:
            self.columns = columns
            if self.columns is not None:
                total_room_ix = self.columns.index(COLUMN_TOTAL_ROOMS)
                population_ix = self.columns.index(COLUMN_POPULATION)
                households_ix = self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_ix = self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.total_room_ix = total_room_ix
            self.population_ix = population_ix
            self.households_ix = households_ix
            self.total_bedrooms_ix = total_bedrooms_ix

        except Exception as e:
            raise Housing_Exception(e, sys)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            logging.info(f"Feature generation started")
            room_per_household = X[:, self.total_room_ix] /\
                X[:, self.households_ix]

            population_per_household = X[:, self.population_ix] /\
                X[:, self.households_ix]

            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] /\
                    X[:, self.total_room_ix]
                generated_feature = np.c_[
                    X, room_per_household,
                    population_per_household,
                    bedrooms_per_room
                ]
            else:
                generated_feature = np.c_[
                    X,
                    room_per_household,
                    population_per_household
                ]
            logging.info(f"Feature generation completed")
            return generated_feature
        except Exception as e:
            raise Housing_Exception(e, sys)


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artificat: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact) -> None:
        """
        Data Transformation Initialization 
        data_transformation_config: DataTranforamtion Configuration object
        data_ingestion_artifact: Data Ingestion Artifact
        data_validation_artifact: Data Validation Artifact
        """
        try:
            logging.info(f"{'='*20} Data Transformation Started. {'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artificat
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
        """
        returns: Dataframe
        file_path: File path
        schema_file_path: Schema File path
        """
        try:
            # reading the dataset schema file
            dataset_schema = read_yaml_file(schema_file_path)

            # extracting columns info from the schema file
            schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

            # reading the dataset
            dataframe = pd.read_csv(file_path)
            error_message = ""

            for column in dataframe.columns:
                if column in list(schema.keys()):
                    dataframe[column].astype(schema[column])
                else:
                    error_message = f"{error_message} \nColumn: [ {column} ] not in the schema"

            if len(error_message) > 0:
                raise Exception(error_message)

            return dataframe
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_data_transformer(self) -> ColumnTransformer:
        try:
            # reading configurations
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(schema_file_path)
            data_transformation_config = self.data_transformation_config

            # splitting input columns and target columns
            target_column = dataset_schema[DATASET_SCHEMA_TARGET_COLUMN_KEY]
            columns = list(dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].keys())
            columns.remove(target_column)

            categorical_column = []
            for column_name, data_type in dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].items():
                if data_type == "category" and column_name != target_column:
                    categorical_column.append(column_name)

            numerical_column = list(
                filter(lambda x: x not in categorical_column, columns))

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("feature_generator", FeatureGenerator(
                    add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,
                    columns=numerical_column)),
                ("scaler", StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoding', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical Columns: [ {numerical_column} ]")
            logging.info(f"Categorical Columns: [ {categorical_column} ]")

            preprocessing = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_column),
                ("cat_pipeline", cat_pipeline, categorical_column)
            ])

            return preprocessing

        except Exception as e:
            raise Housing_Exception(e, sys)

    @staticmethod
    def save_numpy_array_data(file_path: str, array: np.array) -> None:
        """
        Save numpy array data to file
        file_path:str Location of file to save
        array:np.array Numpy array to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    @staticmethod
    def load_numpy_array(file_path: str) -> np.array:
        """
        Load numpy array from file
        file_path:str Location of file from where to load the array
        """
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"""train file path: [ {train_file_path} ]
            test file path: [ {test_file_path} ]
            schema file path: [ {schema_file_path} ]""")

            # loading the dataset
            logging.info(f"Loading train and test dataset...")

            train_dataframe = DataTransformation.load_data(
                file_path=train_file_path,
                schema_file_path=schema_file_path)

            test_dataframe = DataTransformation.load_data(
                file_path=test_file_path,
                schema_file_path=schema_file_path
            )

            logging.info(f"Dataset loaded successfully")

            # read the target column from schema file
            target_column = read_yaml_file(schema_file_path)[
                DATASET_SCHEMA_TARGET_COLUMN_KEY]
            logging.info(f"Target column: [ {target_column} ]")

            # target column
            logging.info(f"Converting target column into numpy array")
            train_target_arr = np.array(train_dataframe[target_column])
            test_target_arr = np.array(test_dataframe[target_column])
            logging.info(
                f"Conversion of target column to numpy array completed")

            logging.info(f"Creating preprocessing object")

            preprocessing = self.get_data_transformer()
            logging.info(f"Creating pre-processing object completed.")
            logging.info(
                f"Preprocessing object learning started on training dataset")
            logging.info(f"Transformation started on training dataset")
            train_input_arr = preprocessing.fit_transform(train_dataframe)
            logging.info(
                f"Preprocessing object learning completed on training dataset")

            logging.info("Transformation started on testing data")
            test_input_arr = preprocessing.transform(test_dataframe)
            logging.info(f"Transformation completed on test data")

            # adding target column back to numpy array
            logging.info(
                f"Started- concatenatation of target column to train and test data")
            train_arr = np.c_[train_input_arr, train_target_arr]
            test_arr = np.c_[test_input_arr, test_target_arr]
            logging.info(
                f"Completed- concatenation of target column to train and test data")

            # generating file name such as housing_transformed.py
            file_name = os.path.basename(train_file_path)
            file_name = file_name.split(".")[0] + "_transformed.npy"
            logging.info(f"File Name: [ {file_name} ] for transfomed dataset")

            # preparing paths to save the transformed train and test data into directory
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            transformed_train_file_path = os.path.join(
                transformed_train_dir, file_name)
            transformed_test_file_path = os.path.join(
                transformed_test_dir, file_name)
            logging.info(
                f"Transformed train file path- [ {transformed_train_file_path} ]")
            logging.info(f"Transformed test file path: [ {test_file_path} ]")

            # saving the transformed data
            logging.info(f"Saving transformed train and test data")

            DataTransformation.save_numpy_array_data(
                transformed_train_file_path, train_arr)
            logging.info(
                f"Train data array saved to [ {transformed_train_file_path} ]")

            DataTransformation.save_numpy_array_data(
                transformed_test_file_path, test_arr)
            logging.info(
                f"Test data array saved to [ {transformed_test_file_path} ]")

            logging.info(f"Saving preprocessing object")
            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path

            # saving the preprocessed object
            save_object(file_path=preprocessed_object_file_path,
                        obj=preprocessing)
            logging.info(
                f"Preprocessed object saved at [ {preprocessed_object_file_path} ]")

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Data Transformation successfully completed",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessed_object_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'=*20'} Data Transformation Completed. {'=*20'}")
