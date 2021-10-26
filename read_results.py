from io import BytesIO
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, Iterable, List
from zipfile import ZipFile

import PySimpleGUI as sg
import dill
import numpy as np
import requests
import yaml
from pint import Quantity
from sympy import Expr

from Function import Function, Model, Parameter, Variable, \
    generateFunction, generateParameter, readFunctionsFromFiles, readParametersFromFiles, \
    generateVariablesFromFile as loadVariables
from Layout.SimulationWindow import SimulationWindowRunner
from Results import Results
from YML import loadConfig


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


def loadFunctions(
    filepath: str,
    stem2path: Dict[str, str],
    archive: ZipFile = None
) -> List[Function]:
    assert isinstance(filepath, str)

    contents = loadConfig(filepath, archive=archive)

    filestems = stem2path.keys()
    loaded_function_objs = []
    for key, value in contents.items():
        if key in filestems:
            if isinstance(value, Iterable):
                path_from_stem = stem2path[key]
                functions_from_file = readFunctionsFromFiles(path_from_stem, names=value).values()
                loaded_function_objs.extend(functions_from_file)
            else:
                sg.PopupError(f"filestem {key:d} not found for function {key:s} (skipping)")
        else:
            new_function_obj = generateFunction(key, value)
            loaded_function_objs.append(new_function_obj)

    return loaded_function_objs


def loadParameters(
    filepath: str,
    stem2path: Dict[str, str],
    archive: ZipFile = None
) -> List[Parameter]:
    assert isinstance(filepath, str)

    info = loadConfig(filepath, archive=archive)

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


def getResults_old(
    results_filepath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> Results:
    if results_filepath is None:
        file_types = (
            ("Compressed File", "*.zip"),
            ("ALL files", "*.*"),
        )

        results_filepath = sg.PopupGetFile(
            message="Enter Filename to Load",
            title="Load Previous Results",
            file_types=file_types,
            multiple_files=False
        )

    if parameter_directory is None:
        parameter_filepaths = []
    elif isinstance(parameter_directory, str):
        parameter_filepaths = [
            join(parameter_directory, filepath)
            for filepath in listdir(parameter_directory)
            if isfile(join(parameter_directory, filepath))
        ]

    if equation_directory is None:
        equation_filepaths = []
    elif isinstance(equation_directory, str):
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

    archive = ZipFile(results_filepath, 'r')

    results = dill.load(
        BytesIO(archive.read("Results.pkl")),
        'rb'
    )
    free_parameters = loadConfig("FreeParameter.yml", archive=archive)
    time_evolution_types = loadConfig("TimeEvolutionType.yml", archive=archive)

    function_objs = loadFunctions("Function.yml", stem2path_func, archive=archive)
    parameter_objs = loadParameters("Parameter.yml", stem2path_param, archive=archive)

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

    return results_obj, archive


def getResults(
    results_filepath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> Results:
    if results_filepath is None:
        file_types = (
            ("Compressed File", "*.zip"),
            ("ALL files", "*.*"),
        )

        results_filepath = sg.PopupGetFile(
            message="Enter Filename to Load",
            title="Load Previous Results",
            file_types=file_types,
            multiple_files=False
        )

    if parameter_directory is None:
        parameter_filepaths = []
    elif isinstance(parameter_directory, str):
        parameter_filepaths = [
            join(parameter_directory, filepath)
            for filepath in listdir(parameter_directory)
            if isfile(join(parameter_directory, filepath))
        ]

    if equation_directory is None:
        equation_filepaths = []
    elif isinstance(equation_directory, str):
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

    archive = ZipFile(results_filepath, 'r')

    results = dill.load(
        BytesIO(archive.read("Results.pkl")),
        'rb'
    )
    free_parameters = loadConfig("FreeParameter.json", archive=archive)
    variable_objs = loadVariables("Variable.json", archive=archive)

    function_objs = loadFunctions("Function.json", stem2path_func, archive=archive)
    parameter_objs = loadParameters("Parameter.json", stem2path_param, archive=archive)

    values = {}
    for name, value in free_parameters.items():
        values[name] = np.array(list(map(float, value["values"])))

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

    return results_obj, archive

def getSimulation(
    results_filepath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> SimulationWindowRunner:
    results_obj, archive = getResults(
        results_filepath=results_filepath,
        parameter_directory=parameter_directory,
        equation_directory=equation_directory
    )
    model = results_obj.getModel()

    free_parameter_contents = loadConfig("FreeParameter.json", archive=archive)

    free_parameter_values = {}
    for name, value in free_parameter_contents.items():
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
        plot_choices=plot_choices,
        include_simulation_tab=False
    )

    return simulation_window

def loadSimulation(
    results_filepath: str = None,
    parameter_directory: str = "parameters",
    equation_directory: str = "equations"
) -> None:
    simulation_window = getSimulation(
        results_filepath=results_filepath,
        parameter_directory=parameter_directory,
        equation_directory=equation_directory
    )
    simulation_window.runWindow()

if __name__ == "__main__":
    """# button = sg.ColorChooserButton("Color")
    # color_input = sg.Input(visible=False, enable_events=True, disabled=True, key='-IN-')
    # window = sg.Window("Choose Button", [[color_input, button]])"""

    sg.ChangeLookAndFeel("DarkGrey13")
    sg.SetOptions(
        element_padding=(1, 1),
        suppress_error_popups=True,
        suppress_raise_key_errors=False
    )

    loadSimulation()
