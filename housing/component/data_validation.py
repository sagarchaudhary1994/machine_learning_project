import os
import sys
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from housing.exception import Housing_Exception
from housing.logger import logging
from housing.config.configuration import Configuration
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs.data_drift_tab import DataDriftTab
import pandas as pd
import json


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20} Data Validation is started {'='*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def is_train_test_file_exist(self) -> bool:
        try:
            logging.info(f"Checking if train and test file exists or not")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            logging.info(f"""Is train file exist at location [ {train_file_path} ] ?- [{is_train_file_exist}]
            Is test file exist at location [ {test_file_path} ] ?- [ {is_test_file_exist} ]""")
            is_files_available = is_train_file_exist and is_test_file_exist

            if is_files_available == False:
                message = None
                if is_train_file_exist == False:
                    message = f"Training file: {train_file_path} is not available."
                elif is_train_file_exist == False:
                    message = f"Testing_file: {test_file_path} is not available"
                else:
                    message = f"Training file: {train_file_path} or Testing_file: {test_file_path} is not available."

                raise Exception(message)

            return is_files_available
        except Exception as e:
            raise Housing_Exception(e, sys)

    def validate_dataset_schema(self) -> bool:
        try:
            validation_status = False
            # Validate training and testing dataset using schema file
            # 1 Number of Column
            # 2. Check the value of ocean proximity
            # 3 Check column names

            validation_status = True
            return validation_status
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df, test_df = self.get_train_and_test_dataframe()
            profile.calculate(train_df, test_df)
            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_file_dir = os.path.dirname(report_file_path)

            os.makedirs(report_file_dir, exist_ok=True)
            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)

            return report
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_and_save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_dataframe()
            dashboard.calculate(train_df, test_df)

            report_page_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_path)

            os.makedirs(report_page_dir, exist_ok=True)

            dashboard.save(report_page_path)
            return report_page_path
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def get_train_and_test_dataframe(self):
        try:
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def is_data_drift_found(self):
        try:
            report = self.get_and_save_data_drift_report()
            report_page_path = self.get_and_save_data_drift_report_page()
            return True, report, report_page_path
        except Exception as e:
            raise Housing_Exception(e, sys) from e

    def detect_outliers(self):
        try:
            pass
        except Exception as e:
            raise Housing_Exception(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            _ = self.is_train_test_file_exist()

            is_validated = self.validate_dataset_schema()
            _, report, report_page_path = self.is_data_drift_found()
            message = "Data validation completed."

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=report,
                report_page_file_path=report_page_path,
                is_validated=is_validated,
                message=message)
            logging.info(
                f"Data Validation Artifact - [ {data_validation_artifact} ]")
            return data_validation_artifact
        except Exception as e:
            raise Housing_Exception(e, sys)

    def __del__(self):
        logging.info(f"{'='*20}Data Validation log completed. {'='*20}\n\n")
