"""
This file contains miscellaneous functions that are used often throughout the project.
"""
import os
from typing import Any, List, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
import yaml
from pint import Quantity
from sympy.printing.preview import preview

import YML

def unique(nonunique: List[Any]) -> List[Any]:
    """
    Get collection containing only unique elements.

    :param nonunique: collection to retrieve unique elements of
    """
    seen = set()
    unique_list = [element for element in nonunique if not (element in seen or seen.add(element))]
    return unique_list

def getIndicies(
        elements: Union[Any, List[Any]], element_list: list, element_class: type = None
) -> Union[int, List[int]]:
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
    Format value for quantity or float.
    Display full precision.
    Remove trailing zeros.
    
    :param quantity: quantity or float to format
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
    Format unit as string from quantity.
    Display unit as abbreviations.
    Remove spaces between units.
    
    :param quantity: quantity to format
    """
    return f"{quantity.units:~}".replace(' ', '')

def formatQuantity(quantity: Quantity) -> str:
    """
    Format quantity as string with value and unit.
    
    :param quantity: quantity to format
    """
    value, unit = formatValue(quantity), formatUnit(quantity)
    formatted_quantity = f"{value:s} {unit:s}"
    return formatted_quantity

def getTexImage(name: str, tex_folder: str = "tex", **kwargs) -> Union[sg.Image, sg.Text]:
    """
    Get tex image associated with variable name.

    :param name: name of variable to retrieve image of
    :param tex_folder: folder to retrieve image from
    :returns: Image if image found in folder.
        Text if image not found in folder.
    """
    filename = os.path.join(tex_folder, name + ".png")
    if os.path.isfile(filename):
        return sg.Image(filename=filename, **kwargs)
    else:
        return sg.Text(name, **kwargs)

def form2tex(form: str, var2tex: str) -> str:
    """
    Get tex expression from equation form.
    
    :param form: form of equation to convert to tex
    :param var2tex: name of YML with variable-to-tex dictionary.
        Key is name of variable.
        Value is tex form of variable.
    """
    form_str = form
    var2tex = yaml.load(open(var2tex, 'r'), Loader=yaml.Loader)
    sorted_tex = sorted(var2tex.keys(), key=len, reverse=True)
    
    for var in sorted_tex:
        reg_expr = f"{var:s}"
        if reg_expr in form_str:
            tex = var2tex[var].replace('$', '')
            form_str = form_str.replace(var, tex)
    form_str = '$' + form_str + '$'
    form_str  = form_str.replace('**', '^')
    form_str = form_str.replace('*', '')
    
    return form_str

def tex2png(tex_text: str, filepath: str, overwrite: bool = False) -> None:
    """
    Save tex for given text as PNG.
    
    :param tex_text: tex form of text
    :param filepath: location to save PNG file
    :param overwrite: set True to overwrite existing quantity if name already exists.
        Set False to skip quantities previously saved as TeX image.
    """
    preview_kwargs = {
        "expr": tex_text,
        "filename": filepath,
        "output": "png",
        "packages": ("amsmath", "amsfonts", "amssymb"),
        "viewer": "file",
        "euler": False
    }
    if not os.path.isfile(filepath) or overwrite:
        preview(**preview_kwargs)

def form2png(name: str, form: str, folder: str, filename: str, var2tex: str) -> None:
    """
    Generate PNG image of tex for variable.
    
    :param name: name of variable to generate image for
    :param form: tex expression to generate image for
    :param folder: output folder for image
    :param filename: base filename for image
    :param var2tex: name of YML with variable-to-tex dictionary.
        Key is name of variable.
        Value is tex form of variable.
    """
    form_str = form2tex(form, var2tex)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    logpath = os.path.join(folder, "log.yml")
    if os.path.isfile(logpath):
        old_info = yaml.load(open(logpath, 'r'), Loader=yaml.Loader)
        if old_info is None:
            old_info = {}
    else:
        old_info = {}
    
    kwargs = {
        "tex_text": form_str,
        "filepath": os.path.join(folder, filename),
        "overwrite": name not in old_info.keys() or form_str != old_info[name]["expression"]
    }
    tex2png(**kwargs)

    new_info = old_info
    new_info[name] = {
        "expression": form_str,
        "filename": filename
    }
    yaml.dump(new_info, open(logpath, 'w'))
    
def tex2pngFromFile(output_folder: str, tex_filename: str, **kwargs) -> None:
    """
    Create PNG for quantity(s) in TeX form.
    File containing quantity name(s) to TeX math format must be made before call.

    :param output_folder: name of folder to save images in
    :param tex_filename: name of file containing name-to-TeX conversions.
        Keys in file are name of quantity.
        Values are corresponding TeX format.
    :param kwargs: additional arguments to pass into :meth:`~macros.tex2png`
    """
    tex_yml = YML.readVar2Tex(tex_filename)
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for key in YML.readVar2Tex(tex_filename):
        filepath = f"{output_folder:s}/{key:s}.png"
        tex2png(tex_yml[key], filepath, **kwargs)