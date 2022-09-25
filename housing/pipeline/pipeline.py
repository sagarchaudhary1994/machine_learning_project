from housing.component.data_transformation import DataTransformation
from housing.component.model_evaluation import ModelEvaluation
from housing.config.configuration import Configuration
from housing.constant import DATA_TRANSFORMATION_CONFIG_KEY, MODEL_TRAINER_CONFIG_KEY
from housing.logger import logging
from housing.exception import Housing_Exception
from housing.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact,\
    MetricInfoArtifact, ModelTrainerArtifact
from housing.component.data_ingestion import DataIngestion
from housing.component.data_validation import DataValidation
from housing.component.model_trainer import ModelTrainer
import os
import sys


class Pipeline:
    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(self.config.get_data_validation_config(),
                                             data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation = DataTransformation(
                self.config.get_data_transformation_config(),
                data_ingestion_artifact,
                data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transformation_artifact=data_transformation_artifact
                                         )

            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_model_evaluation(self,data_ingestion_artifact: DataIngestionArtifact,
    data_validation_artifact:DataValidationArtifact,
    model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config= self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)

            return model_evaluation.intitiate_model_evaluation()

        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def start_model_pusher(self):
        try:
            pass
        except Exception as e:
            raise Housing_Exception(sys, e) from e

    def run_pipeline(self):
        try:
            # data ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # data validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact)

            # data transformation
            data_tranformation_artifact = self.start_data_transformation(
                data_ingestion_artifact,
                data_validation_artifact
            )
            # model trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_tranformation_artifact)

            #model evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

        except Exception as e:
            raise Housing_Exception(e, sys) from e
