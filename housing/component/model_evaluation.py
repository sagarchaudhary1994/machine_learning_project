import numpy
from housing.constant import DATASET_SCHEMA_TARGET_COLUMN_KEY
from housing.entity.config_entity import ModelEvaluationConfig
from housing.exception import Housing_Exception
from housing.logger import logging
import os
import sys
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, MetricInfoArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from housing.component.data_transformation import DataTransformation
from housing.component.model_trainer import ModelTrainer
from housing.util.util import load_object, read_yaml_file, write_yaml_file

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'='*20} Model Evaluation Started {'='*20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                data=None)
                return model

            model_eval_file_content = read_yaml_file(
                model_evaluation_file_path)
            model_eval_file_content = dict(
            ) if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(
                file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])

            return model

        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_evaluation_content = read_yaml_file(eval_file_path)
            model_evaluation_content = dict(
            ) if model_evaluation_content is None else model_evaluation_content

            previous_best_model = None
            if BEST_MODEL_KEY in model_evaluation_content:
                previous_best_model = model_evaluation_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_evaluation_content}")

            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }

            if previous_best_model is not None:
                model_history = {
                    self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_evaluation_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_evaluation_content[HISTORY_KEY].update(model_history)

            model_evaluation_content.update(eval_result)
            logging.info(
                f"Updated evaluation result: {model_evaluation_content}")

            write_yaml_file(
                file_path=eval_file_path,
                data=model_evaluation_content
            )
            logging.info(f"Config updated to [ {eval_file_path} ]")
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def intitiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = DataTransformation.load_data(
                file_path=train_file_path,
                schema_file_path=schema_file_path
            )

            test_dataframe = DataTransformation.load_data(
                file_path=test_file_path,
                schema_file_path=schema_file_path
            )

            schema_content = read_yaml_file(schema_file_path)
            target_column_name = schema_content[DATASET_SCHEMA_TARGET_COLUMN_KEY]

            # target column
            logging.info(f"Converting target column into numpy array")

            train_target_arr = numpy.array(train_dataframe[target_column_name])
            test_target_arr = numpy.array(test_dataframe[target_column_name])
            logging.info(
                f"Conversion of Target column to numpy array in train and test data completed")

            # dropping target column from the dataframe
            logging.info(
                f"Dropping of target column {target_column_name} from train and test started")
            train_dataframe.drop(columns=[target_column_name], inplace=True)
            test_dataframe.drop(columns=[target_column_name], inplace=True)
            logging.info(
                f"Dropping of target column {target_column_name} from train and test completed")

            model = self.get_best_model()

            if model is None:
                logging.info(
                    "Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(
                    model_evaluation_artifact=model_evaluation_artifact)
                logging.info(
                    f"Model accepted and model evaluation artifact created: {model_evaluation_artifact}")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]
            metric_info_artifact: MetricInfoArtifact = ModelTrainer.evaluate_model(
                model_list=model_list,
                X_train=train_dataframe,
                y_train=train_target_arr,
                X_test=test_dataframe,
                y_test=test_target_arr,
                base_accuracy=self.model_trainer_artifact.model_accuracy
            )

            logging.info(
                f"Model evaluation completed. Model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path
                )
                logging.info(response)

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    evaluated_model_path=trained_model_file_path,
                    is_model_accepted=True
                )

                self.update_evaluation_report(
                    model_evaluation_artifact=model_evaluation_artifact)
                logging.info(
                    f"Model accpeted. Model Evaluation Artifact: {model_evaluation_artifact}")
            else:
                logging.info(
                    f"Trained model is no better than the existing model hance not accepting the trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path
                )

            return model_evaluation_artifact

        except Exception as e:
            raise Housing_Exception(e, sys)

    def __del__(self):
        logging.info(f"{'='*20} Model Evaluation Completed {'='*20}")
