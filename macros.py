"""
This file contains miscellaneous functions that are used often throughout the project.
"""
from os.path import isfile, join
from typing import Any, List, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from pint import Quantity


def unique(nonunique: list) -> list:
    """
    Get collection containing only unique elements.

    :param nonunique: collection to retrieve unique elements of
    """
    seen = set()
    unique_list = [element for element in nonunique if not (element in seen or seen.add(element))]
    return unique_list


def getIndicies(elements: Union[Any, List[Any]], element_list: list, element_class: type = None) -> Union[
    int, List[int]]:
    """
    Get indicies of specified element(s) in list.
    
    __Recursion Base__
        return single index: elements [element_class] or element_class [None]
    
    :param elements: elements to retreive indicies of
    :param element_list: collection to search for indicies in
    :param element_class: return only elements of this type.
        Acts as a filter.
    :returns: int if elements is a list.
        list of int if elements is not a list.
    """
    if isinstance(elements, list):
        return [getIndicies(element, element_list) for element in elements]
    elif element_class is None or isinstance(elements, element_class):
        return element_list.index(elements)
    else:
        raise TypeError("element input must be {element_class:s} or list")


def getElements(indicies: Union[int, List[int]], element_list: list) -> Union[Any, List[Any]]:
    """
    Get element(s) at index(es) in list.
    
    __Recursion Base__
        return single element: indicies [int]
    
    :param indicies: index(es) to retrieve element(s) at
    :param element_list: list to retrieve element(s) from
    :returns: object if indicies is int.
        list of objects if indicies is list of int.
    """
    if isinstance(indicies, list):
        return [getElements(index, element_list) for index in indicies]
    elif isinstance(indicies, int):
        return element_list[indicies]


def commonElement(set1: set, set2: set, n: int = 1) -> bool:
    """
    Determine whether list1 and list2 have at least n common elements.
        
    :param set1: first arbitrary set of elements
    :param set2: second arbitrary set of elements
    :param n: minimum number of common elements
    :returns: True if sets have at least minimum number of common elements satisfied.
        False otherwise.
    """
    return len(set1.intersection(set2)) >= n


def toList(obj: Any, object_class: type = None) -> list:
    """
    Convert input to list.
    
    :param obj: object to convert to a list
    :param object_class: class of object if not list
    :returns: object itself if object is a list.
        list containing only the object if object is not a list.
    """
    if object_class is None or isinstance(obj, object_class):
        return [obj]
    elif isinstance(obj, list):
        return obj
    else:
        raise TypeError(f"object input must be {object_class:s} or list")


def formatValue(quantity: Union[Quantity, float]) -> str:
    """
    __Purpose__
        Format value for quantity containing value and unit
        Display +/- 0.01 precision
        Remove trailing zeros
    __Inputs__
        quantity [metpy.Quantity]: quantity to format value of
    __Return__
        str
    """
    if isinstance(quantity, Quantity):
        magnitude = quantity.magnitude
    elif isinstance(quantity, float):
        magnitude = quantity
    else:
        raise TypeError("quantity must be Quantity or float")

    decimal = f"{magnitude:f}".rstrip('0').rstrip('.')

    scientific_splits = f"{magnitude:e}".split('e')
    scientific_float, scientific_exp = scientific_splits[0].rstrip('0').rstrip('.'), scientific_splits[1]
    scientific_full = scientific_float + 'e' + scientific_exp

    if float(decimal) != magnitude:
        return scientific_full
    elif len(decimal) <= len(scientific_full):
        return decimal
    elif len(decimal) > len(scientific_full):
        return scientific_full


def formatUnit(quantity: Quantity) -> str:
    """
    __Purpose__
        Format unit as string for quantity containing value and unit
        Display unit abbreviations
        Remove spaces between units
    __Inputs__
        quantity [metpy.Quantity]: quantity to format value of
    __Return__
        str
    """
    return f"{quantity.units:~}".replace(' ', '')


def formatQuantity(quantity: Quantity) -> str:
    """
    __Purpose__
        Format quantity as string for unit containing unit and quantity
    __Inputs__
        quantity [metpy.Quantity]: quantity to format value of
    __Return__
        str
    """
    value, unit = formatValue(quantity), formatUnit(quantity)
    formatted_quantity = f"{value:s} {unit:s}"
    return formatted_quantity


def getTexImage(name: str, tex_folder: str = "tex", **kwargs) -> Union[sg.Image, sg.Text]:
    """
    __Purpose__
        Get tex image associated with variable name
    __Inputs__
        name [str]: name of variable to retrieve tex image of
        **kwargs [**dict]: extra argument for PySimpleGUI.Image
    __Return__
        PySimpleGUI.Image
    """
    filename = join(tex_folder, name) + ".png"
    if isfile(filename):
        return sg.Image(filename=filename, **kwargs)
    else:
        return sg.Text(name, **kwargs)
