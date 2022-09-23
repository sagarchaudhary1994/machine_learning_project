import yaml
from housing.exception import Housing_Exception
import os
import sys
import dill

def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and returns the content as dictionary.
    file_path: str
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Housing_Exception(e, sys) from e

def save_object(file_path:str, obj)->None:
    """
    Saving the serialization of object to a file
    
    file_path:str Location to save the object
    obj: Any object to be serialized and saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Housing_Exception(e,sys) from e

def load_object(file_path:str)->object:
    """
    Return the object by deserialization from file

    file_path: Location to read and deserialize the file
    """
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise Housing_Exception(e, sys) from e