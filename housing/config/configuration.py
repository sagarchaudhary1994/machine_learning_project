from housing.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig, TrainingPipelineConfig
from housing.exception import Housing_Exception
from housing.util.util import read_yaml_file
from housing.constant import *
from housing.logger import logging
import sys


class Configuration:

    def __init__(self, config_file_path: str = CONFIG_FILE_PATH,
                 current_time_stamp: str = CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]

            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]

            data_ingestion_artifact_dir = os.path.join(artifact_dir,
                                                       DATA_INGESTION_ARTIFACT_DIR, CURRENT_TIME_STAMP)
            raw_data_dir = os.path.join(
                data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])

            tgz_download_dir = os.path.join(
                data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY])

            ingested_data_dir = os.path.join(data_ingestion_artifact_dir,
                                             data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])

            ingested_train_dir = os.path.join(
                ingested_data_dir, data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])

            ingested_test_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])

            data_ingestion_config = DataIngestionConfig(
                dataset_download_url,
                tgz_download_dir,
                raw_data_dir, ingested_train_dir, ingested_test_dir)

            logging.info(f"Data Ingestion Config: {data_ingestion_config}")

            return data_ingestion_config
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_artifact_dir = os.path.join(
                artifact_dir, DATA_VALIDATION_ARTIFACT_DIR, CURRENT_TIME_STAMP)

            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])

            report_file_name = os.path.join(
                data_validation_artifact_dir,
                data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])

            report_page_file_path = os.path.join(
                data_validation_artifact_dir,
                data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])

            data_validation_config = DataValidationConfig(
                schema_file_path, report_file_name, report_page_file_path)

            logging.info(f"Data Validation Config- {data_validation_config}")

            return data_validation_config

        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_artifact_dir = os.path.join(
                artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR, CURRENT_TIME_STAMP)

            add_bedroom_per_room = data_transformation_info[DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]

            transformed_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY])
            transformed_train_dir = os.path.join(
                transformed_dir, data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY])
            transformed_test_dir = os.path.join(
                transformed_dir, data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY])

            preprocessed_dir = os.path.join(
                artifact_dir, data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])
            preprocessed_object_file_path = os.path.join(
                preprocessed_dir, data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY])

            data_transformation_config = DataTransformationConfig(
                add_bedroom_per_room,
                transformed_train_dir,
                transformed_test_dir,
                preprocessed_object_file_path)
            logging.info(
                f"Data Transformation Config- {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir
            model_trainer_artifact_dir = os.path.join(
                artifact_dir, MODEL_TRAINER_ARTIFACT_DIR, CURRENT_TIME_STAMP)

            base_accuracy = model_trainer_info[MODEL_TRAINER_BASE_ACCURACY_KEY]

            trained_model_dir = os.path.join(
                model_trainer_artifact_dir, MODEL_TRAINER_TRAINED_MODEL_DIR_KEY)
            trained_model_file_path = os.path.join(
                trained_model_dir, model_trainer_info[MODEL_TRAINER_MODEL_FILE_NAME_KEY])

            model_config_dir = os.path.join(
                ROOT_DIR, model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY])
            model_config_file_path = os.path.join(
                model_config_dir, model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_FILE_KEY])

            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path, base_accuracy, model_config_file_path)
            logging.info(f"Model Trainer Config- {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_info = self.config_info[MODEL_EVALUATION_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir
            model_evaluation_artifact_dir = os.path.join(
                artifact_dir, MODEL_EVALUATION_ARTIFACT_DIR, CURRENT_TIME_STAMP)

            model_evaluation_file_path = os.path.join(
                model_evaluation_artifact_dir, model_evaluation_info[MODEL_EVALUATION_FILE_NAME_KEY])

            model_evaluation_config = ModelEvaluationConfig(
                model_evaluation_file_path, CURRENT_TIME_STAMP)
            logging.info(f"Model Evaluation Config- {model_evaluation_config}")
            return model_evaluation_config
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            model_pusher_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            artifact_dir = self.training_pipeline_config.artifact_dir
            export_dir_path = os.path.join(
                artifact_dir, model_pusher_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY], CURRENT_TIME_STAMP.replace("-",""))
            model_pusher_config = ModelPusherConfig(export_dir_path)
            logging.info(f"Model Pusher Config- {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise Housing_Exception(e, sys)

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )

            training_pipeline_config = TrainingPipelineConfig(
                artifact_dir=artifact_dir)
            logging.info(
                f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config

        except Exception as e:
            raise Housing_Exception(e, sys) from e
