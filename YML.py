import json
from io import BytesIO
from os.path import join
from pathlib import Path
from typing import List, Optional, Tuple, Union
from zipfile import ZipFile

import xmltodict
import yaml

specific_model_folderpath = "specific_model"
var2tex_filepath = join(specific_model_folderpath, "var2tex.yml")
states_filepath = join(specific_model_folderpath, "states.json")
prefix_filepath = join(specific_model_folderpath, "prefix.json")
layout_dimensions_filepath = join(specific_model_folderpath, "layout_dimensions.json")

config_file_types = [
    ("JavaScript Object Notation", "*.json"),
    ("YML", "*.yml"),
    ("YAML Ain't Markup Language", "*.yaml"),
    ("Extensible Markup Language", "*.xml")
]
config_file_extensions = [
    config_file_type[1].replace('*', '')
    for config_file_type in config_file_types
]


def readVar2Tex(filepath: str = var2tex_filepath):
    var2tex = loadConfig(filepath)
    return var2tex


def readLayout(filepath: str) -> dict:
    """
    Read YML file containing layout for window.

    :param filepath: path of YML file
    """
    layout = loadConfig(filepath)
    return layout


def readPrefixes(filepath: str = prefix_filepath) -> dict:
    prefixes = loadConfig(filepath)
    return prefixes


def readStates(choice=None, filepath: str = states_filepath) -> dict:
    """
    Get dictionary of default values for elements in window.

    :param choice: element to retrieve states for
    :param filepath: path of file to retrieve states from
    """
    choices = loadConfig(filepath)
    if choice is None:
        return choices
    elif choice is not None:
        return choices[choice]


def getStates(choice: str, name: str, filepath: str = states_filepath):
    choices = readStates(choice, filepath)
    return choices[name]


def getDimensions(keys: List[str], filepath: str = layout_dimensions_filepath) -> Tuple[Optional[float], Optional[float]]:
    def getDimension(dimension):
        if dimension is not None:
            try:
                return int(dimension)
            except ValueError:
                return round(eval(dimension))
        elif dimension is None:
            return None

    dimensions = loadConfig(filepath)

    if isinstance(keys, list):
        for key in keys:
            dimensions = dimensions[key]
        width, height = getDimension(dimensions["width"]), getDimension(dimensions["height"])
        return width, height
    else:
        raise TypeError("keys input must be str or list")


def loadConfig(filepath: str, archive: ZipFile = None) -> Union[dict, list]:
    """
    Load contents from config file ('*.json', '*.xml', '*.yml', '*.yaml')

    :param filepath: path of file to load contents from.
        Must be relative to *.zip file if :paramref:`~YML.loadConfig.archive` is given.
    :param archive: zip file to load file from


    """
    assert isinstance(filepath, str)

    if archive is None:
        try:
            file = open(filepath, 'r')
        except FileNotFoundError as error:
            print(str(error))
            return None
    else:
        assert isinstance(archive, ZipFile)
        file = BytesIO(archive.read(filepath))

    file_extension = Path(filepath).suffix
    if file_extension == ".json":
        contents = json.load(file)
    elif file_extension == ".xml":
        file_str = file.read()
        contents = xmltodict.parse(file_str, dict_constructor=dict)["root"]
    elif file_extension in [".yml", ".yaml"]:
        contents = yaml.load(file, Loader=yaml.Loader)

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

    with open(filepath, 'w') as file:
        if file_extension == ".json":
            json.dump(contents, file)
        elif file_extension == ".xml":
            xml_str = xmltodict.unparse({"root": contents})
            file.write(xml_str)
        elif file_extension in [".yml", ".yaml"]:
            yaml.dump(
                contents,
                file,
                default_flow_style=None,
                sort_keys=False
            )

    return file
