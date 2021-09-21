import json
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union
from zipfile import ZipFile

import yaml

config_file_types = [
    ("YML", "*.yml"),
    ("YAML", "*.yaml"),
    ("JSON", "*.json"),
]
config_file_extensions = [
    config_file_type[1].replace('*', '')
    for config_file_type in config_file_types
]


def readVar2Tex(filename="var2tex.yml"):
    var2tex = loadConfig(filename)
    return var2tex


def readLayout(filename: str) -> dict:
    """
    Read YML file containing layout for window.
    
    :param filename: name of YML file
    """
    layout = loadConfig(filename)
    return layout


def readPrefixes(filename="prefix.yml"):
    prefixes = loadConfig(filename)
    return prefixes


def readStates(choice=None, filename: str = "states.yml") -> dict:
    """
    Get dictionary of default values for elements in window.
    
    :param choice: element to retrieve states for
    :param filename: name of file to retrieve states from
    """
    choices = loadConfig(filename)
    if choice is None:
        return choices
    elif choice is not None:
        return choices[choice]


def getStates(choice, name, filename="states.yml"):
    choices = readStates(choice, filename)
    return choices[name]


def getDimensions(keys: List[str], filename: str = "layout_dimensions.yml") -> Tuple[Optional[float], Optional[float]]:
    def getDimension(dimension):
        if dimension is not None:
            try:
                return int(dimension)
            except ValueError:
                return round(eval(dimension))
        elif dimension is None:
            return None

    dimensions = loadConfig(filename)

    if isinstance(keys, list):
        for key in keys:
            dimensions = dimensions[key]
        width, height = getDimension(dimensions["width"]), getDimension(dimensions["height"])
        return width, height
    else:
        raise TypeError("keys input must be str or list")


def loadConfig(filepath: str, archive: ZipFile = None) -> Union[dict, list]:
    """
    Load contents from config file ('*.yml', '*.yaml', '*.json')

    :param filepath: path of file to load contents from.
        Must be relative to *.zip file if :paramref:`~YML.loadConfig.archive` is given.
    """
    assert isinstance(filepath, str)

    file_extension = Path(filepath).suffix

    if file_extension in [".yml", ".yaml"]:
        load = partial(yaml.load, Loader=yaml.Loader)
    elif file_extension == ".json":
        load = json.load

    if archive is None:
        file = open(filepath, 'r')
    else:
        assert isinstance(archive, ZipFile)
        file = BytesIO(archive.read(filepath))
    
    contents = load(file)
    file.close()

    return contents

def saveConfig(contents: dict, filepath: str) -> BytesIO:
    """
    Save contents into config file for future retrieval.

    :param contents: contents/information to save into file
    :param filepath: name of path to save contents into
    :returns: Saved file object as bytes
    """
    assert isinstance(filepath, str)

    file_extension = Path(filepath).suffix

    if file_extension in [".yml", ".yaml"]:
        dump = partial(
            yaml.dump, 
            default_flow_style=None,
            sort_keys=False
        )
    elif file_extension == ".json":
        dump = json.dump

    with open(filepath, 'w') as file:
        dump(contents, file)
    
    return file
