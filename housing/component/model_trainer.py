from housing.logger import logging
from housing.exception import Housing_Exception
from housing.entity.artifact_entity import ModelTrainerArtifact, MetricInfoArtifact, DataTransformationArtifact
from housing.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from housing.util.util import save_object, load_object
import os, sys
from housing.entity.model_factory import ModelFactory
from housing.component.data_transformation import DataTransformation
from sklearn.metrics import r2_score, mean_squared_error

class TrainedModel:

    def __init__(self, preprocessing_object, trained_model_object):
        """
        preprocessing_object: Preprocessing Object
        trained_model_object: Trained Model Object
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise Housing_Exception(e, sys) from e
        
    def predict(self, X):
        """
        Function accepts the raw input and then does the preprocessing 
        and finally returns the predictions
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

    
class ModelTrainer:

    def __init__(self,model_trainer_config,
    data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'='*20}Model Trainer started. {'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise Housing_Exception(e, sys) from e
    
    @staticmethod
    def evaluate_model(model_list, X_train, y_train, X_test, 
    y_test, base_accuracy=0.5)->MetricInfoArtifact:
        """
        Function returns the MetricInfoArtifact for the list of GridSearched 
        best models
        """
        try:
            index_number = 0
            metric_info_artifact =None
            for model in model_list:
                model_name = str(model)
                logging.info(f"Started evaluating model: [ type{model}.__name__ ]")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_acc= r2_score(y_train,y_train_pred)
                test_acc = r2_score(y_test, y_test_pred)
                train_rmse = mean_squared_error(y_train, y_train_pred,)
                test_rmse = mean_squared_error(y_test, y_test_pred)

                # Calculate harmonic mean of train and test accuracy
                model_accuracy = (2*(train_acc*test_acc))/(train_acc + test_acc)
                diff_test_train_acc = abs(test_acc - train_acc)

                message = f"{'='*20}{model_name} metric info {'='*20}"
                logging.info(message)
                message = f"\n\t\tTrain Accuracy: [ {train_acc} ]"
                message += f"\n\t\tTest Accuracy: [ {test_acc} ]"
                message += f"\n\t\tTrain RMSE: [ {train_rmse} ]"
                message += f"\n\t\tTest RMSE: [ {test_rmse} ]"
                message += f"\n\t\Model Accuracy: [ {model_accuracy} ]"
                message += f"\n\t\Base Accuracy: [ {base_accuracy} ]"
                message += f"\n\t\Difference of test and train Accuracy: [ {diff_test_train_acc} ]"
                logging.info(f"{message}")

                if model_accuracy > base_accuracy and diff_test_train_acc<0.05:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(
                        model_name=model_name,
                        model_object= model,
                        train_rmse= train_rmse,
                        test_rmse= test_rmse,
                        train_accuracy= train_acc,
                        test_accuracy= test_acc,
                        model_accuracy=model_accuracy,
                        index_number= index_number
                    )
                    logging.info(f"Acceptable model found: [ {metric_info_artifact} ]")
                index_number+=1

                if metric_info_artifact is None:
                    logging.info(f"No model found with accuracy higher than the base accuracy")
                
                return metric_info_artifact
        except Exception as e:
            raise Housing_Exception(e, sys) from e
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_dataset = DataTransformation.load_numpy_array(train_file_path)
            test_dataset = DataTransformation.load_numpy_array(test_file_path)
            X_train, y_train = train_dataset[:,:-1], train_dataset[:,-1]
            X_test, y_test = test_dataset[:,:-1], test_dataset[:,-1]
            model_factory = ModelFactory(self.model_trainer_config.model_config_file_path)
            best_model_list = model_factory.initiate_best_parameter_search_for_initialized_models(
                initialized_models_list= model_factory.get_initialized_model_list(),
                input_feature= X_train,
                output_feature=y_train
            )
            model_list = [model.best_model for model in best_model_list]
            model_metric_artifact = ModelTrainer.evaluate_model(model_list=model_list,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            base_accuracy=self.model_trainer_config.base_accuracy)

            if model_metric_artifact is None:
                raise Exception("None of the models has accuracy higher than the base accuracy")

            preprocessed_object_file_path = self.data_transformation_artifact.preprocessed_object_file_path
            preprocessed_object = load_object(preprocessed_object_file_path)

            trained_model = TrainedModel(preprocessing_object=preprocessed_object,
            trained_model_object= model_metric_artifact.model_object)

            trained_model_path = self.model_trainer_config.trained_model_file_path
            logging.info(f"Saving trained model to: [ {trained_model_path} ]")
            save_object(trained_model_path,trained_model)
            logging.info(f"Trained model saved to: [ {trained_model_path} ]")

            response = ModelTrainerArtifact(
                is_trained=True,
                message ="Model trained successfully",
                trained_model_file_path= trained_model_path,
                train_rmse= model_metric_artifact.train_rmse,
                test_rmse= model_metric_artifact.test_rmse,
                train_accuracy= model_metric_artifact.train_accuracy,
                test_accuracy= model_metric_artifact.test_accuracy,
                model_accuracy= model_metric_artifact.model_accuracy
            )

            logging.info(f"Trained model artifact: [ {response} ]")
            return response

        except Exception as e:
            raise Housing_Exception(e, sys) from e
    def __del__(self):
        return f"{'='*20}Model trainer completed. {'='*20}"