from setuptools import setup
from typing import List

# Declaring variable for setup
PROJECT_NAME = "housing-predictor"
PROJECT_VERSION = "0.0.1"
PROJECT_AUTHOR = "Sagar Chaudhary"
PROJECT_DESC = "first end to end ml project"
PACKAGES = ["housing"]
REQUIREMENT_FILE_NAME = "requirements.txt"


def get_requirements_list() -> List[str]:
    """
    Description: This function is going to return the list of requirement
    mention in the requirement.txt file

    returns: This function is going to return the list which will contains name
    of libraries mentioned in the requirements.txt
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines()


setup(
    name=PROJECT_NAME,
    version=PROJECT_VERSION,
    author=PROJECT_AUTHOR,
    description=PROJECT_DESC,
    packages=PACKAGES,
    install_requires=get_requirements_list()
)

if __name__ == "__main__":
    print(get_requirements_list())
