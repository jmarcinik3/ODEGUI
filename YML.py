from typing import List, Optional, Tuple

import yaml


def readVar2Tex(filename="var2tex.yml"):
    var2tex = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
    return var2tex


def readLayout(filename: str) -> dict:
    """
    Read YML file containing layout for window.
    
    :param filename: name of YML file
    """
    layout = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
    return layout


def readPrefixes(filename="prefix.yml"):
    prefixes = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
    return prefixes


def readStates(choice=None, filename: str = "states.yml") -> dict:
    """
    Get dictionary of default values for elements in window.
    
    :param choice: element to retrieve states for
    :param filename: name of file to retrieve states from
    """
    choices = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
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

    dimensions = yaml.load(open(filename, 'r'), Loader=yaml.Loader)

    if isinstance(keys, list):
        for key in keys:
            dimensions = dimensions[key]
        width, height = getDimension(dimensions["width"]), getDimension(dimensions["height"])
        return width, height
    else:
        raise TypeError("keys input must be str or list")
