from housing.pipeline.pipeline import Pipeline
from housing.exception import Housing_Exception
from housing.logger import logging
from housing.config.configuration import Configuration


def main():
    pipeline = Pipeline()
    pipeline.run_pipeline()
    # val_config = Configuration().get_model_evaluation_config()
    # print(val_config)


if __name__ == "__main__":
    main()
