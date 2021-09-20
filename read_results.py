import zipfile
from io import BytesIO
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type, Union

import PySimpleGUI as sg
import dill
import numpy as np
import requests
import yaml
from pint import Quantity
from sympy import Expr

from Function import Dependent, Derivative, Function, Independent, Model, NonPiecewise, Parameter, Piecewise, Variable, generateFunction, generateParameter, readFunctionsFromFiles, readParametersFromFiles
from Layout.SimulationWindow import SimulationWindowRunner
from Results import Results


def doi2bib(doi):
    """
    Return a bibTeX string of metadata for a given DOI.
    """
    url = "http://dx.doi.org/" + doi
    headers = {
        "accept": "application/x-bibtex"
    }
    r = requests.get(url, headers=headers)
    return r.text


def loadFunctions(file: Union[str, BytesIO], stem2path: Dict[str, str]) -> List[Function]:
    if isinstance(file, str):
        file = open(file, 'r')

    info = yaml.load(file, Loader=yaml.Loader)
    filestems = stem2path.keys()
    loaded_functions = []
    for key, value in info.items():
        if key in filestems:
            if isinstance(value, Iterable):
                path_from_stem = stem2path[key]
                functions_from_file = readFunctionsFromFiles(path_from_stem, names=value).values()
                loaded_functions.extend(functions_from_file)
            else:
                sg.PopupError(f"filestem {key:d} not found for function {key:s} (skipping)")
        else:
            loaded_functions.append(generateFunction(key, value))

    return loaded_functions


def loadParameters(file: Union[str, BytesIO], stem2path: Dict[str, str]) -> List[Parameter]:
    if isinstance(file, str):
        file = open(file, 'r')

    info = yaml.load(file, Loader=yaml.Loader)
    filestems = stem2path.keys()
    loaded_parameters = []
    for key, value in info.items():
        if key in filestems:
            if isinstance(value, Iterable):
                path_from_stem = stem2path[key]
                parameters_from_file = readParametersFromFiles(path_from_stem, names=value).values()
                loaded_parameters.extend(parameters_from_file)
            else:
                sg.PopupError(f"filestem {key:d} not found for parameters (skipping)")
        else:
            loaded_parameters.append(generateParameter(key, value))

    return loaded_parameters


def loadResultsFromFile(
        results_filepath: str, parameter_directory: str, equation_directory: str
) -> Results:
    parameter_filepaths = [
        join(parameter_directory, filepath)
        for filepath in listdir(parameter_directory)
        if isfile(join(parameter_directory, filepath))
    ]

    equation_filepaths = [
        join(equation_directory, filepath)
        for filepath in listdir(equation_directory)
        if isfile(join(equation_directory, filepath))
    ]

    stem2path_param = {
        Path(filepath).stem: filepath
        for filepath in parameter_filepaths
    }
    stem2path_func = {
        Path(filepath).stem: filepath
        for filepath in equation_filepaths
    }

    archive = zipfile.ZipFile(results_filepath, 'r')

    results = dill.load(
        BytesIO(archive.read("Results.pkl")), 
        'rb'
    )
    free_parameters = yaml.load(
        BytesIO(archive.read("FreeParameter.yml")), 
        Loader=yaml.Loader
    )
    time_evolution_types = yaml.load(
        BytesIO(archive.read("TimeEvolutionType.yml")), 
        Loader=yaml.Loader
    )
    function_objs = loadFunctions(
        BytesIO(archive.read("Function.yml")), 
        stem2path_func
    )
    parameter_objs = loadParameters(
        BytesIO(archive.read("Parameter.yml")), 
        stem2path_param
    )

    values = {}
    for name, value in free_parameters.items():
        values[name] = np.array(list(map(float, value["values"])))

    variable_objs = []
    for time_evolution_type, variable_names in time_evolution_types.items():
        for variable_name in variable_names:
            variable_obj = Variable(
                variable_name,
                time_evolution_type=time_evolution_type
            )
            variable_objs.append(variable_obj)

    model = Model(
        variables=variable_objs,
        functions=function_objs, 
        parameters=parameter_objs
    )

    results_obj = Results(
        model, 
        values, 
        results
    )

    return results_obj

def getSimulationFromResults(
        results_filepath: str, parameter_directory: str = "parameters", equation_directory: str = "equations"
) -> SimulationWindowRunner:
    results_obj = loadResultsFromFile(
        results_filepath,
        parameter_directory=parameter_directory,
        equation_directory=equation_directory
    )

    model = results_obj.getModel()

    archive = zipfile.ZipFile(results_filepath, 'r')
    free_parameters = yaml.load(
        BytesIO(archive.read("FreeParameter.yml")), 
        Loader=yaml.Loader
    )
    free_parameter_values = {}
    for name, value in free_parameters.items():
        quantity = Quantity(0, value["unit"])
        values = list(map(float, value["values"]))
        minimum, maximum, stepcount = min(values), max(values), len(values)
        free_parameter_values[name] = (minimum, maximum, stepcount, quantity)

    plot_choices = {
        "Variable": model.getVariables(return_type=str) + ['t'],
        "Function": model.getFunctionNames(),
        "Parameter": list(free_parameter_values.keys())
    }

    simulation_window = SimulationWindowRunner(
        "Simulation from Previous Results",
        results=results_obj,
        free_parameter_values=free_parameter_values,
        plot_choices=plot_choices
    )

    return simulation_window


if __name__ == "__main__":
    """# button = sg.ColorChooserButton("Color")
    # color_input = sg.Input(visible=False, enable_events=True, disabled=True, key='-IN-')
    # window = sg.Window("Choose Button", [[color_input, button]])"""

    """#dx_lamb = sym.lambdify((Symbol('t'), x), dx)
    #print(inspect.getsource(dx_lamb))
    dx_lamb = temp_func._lambdifygenerated"""

    sg.ChangeLookAndFeel("DarkGrey13")
    sg.SetOptions(
        element_padding=(1, 1),
        suppress_error_popups=True,
        suppress_raise_key_errors=False
    )

    file_types = (("Compressed File", "*.zip"), ("ALL files", "*.*"),)
    res_path = sg.PopupGetFile(
        message="Enter Filename to Load",
        title="Load Previous Results",
        file_types=file_types,
        multiple_files=False
    )
    simulation_window = getSimulationFromResults(
        res_path,
        parameter_directory="parameters",
        equation_directory="equations"
    )

    simulation_window.runWindow()
