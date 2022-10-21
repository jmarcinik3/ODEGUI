from io import BytesIO
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
from zipfile import ZipFile

import PySimpleGUI as sg
import dill
import numpy as np
from pint import Quantity

from Function import FreeParameter, Function, Model, Parameter, Variable, \
    generateFunction, generateParameter, readFunctionsFromFiles, readParametersFromFiles, \
    generateVariablesFromFile as loadVariables
from Layout.AxisQuantity import AxisQuantityMetadata, generateMetadataFromFile
from Layout.GridSimulationWindow import GridSimulationWindowRunner
from Layout.OptimizationSimulationWindow import OptimizationSimulationWindowRunner
from Layout.SimulationWindow import SimulationWindowRunner
from Results import GridResults, OptimizationResults, OptimizationResultsFileHandler, Results
from Config import loadConfig
from macros import generateQuantityFromMetadata


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

    contents = loadConfig(filepath, archive=archive)

    filestems = stem2path.keys()
    loaded_parameters = []
    for key, value in contents.items():
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
) -> GridResults:
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

    results_obj = GridResults(
        model,
        values,
        results
    )

    return results_obj, archive


def getResults_old2(
    results_filepath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> GridResults:
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

    results_obj = GridResults(
        model,
        values,
        results
    )

    return results_obj, archive


def getResults(
    results_folderpath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> Tuple[Results, str]:
    if results_folderpath is None:
        results_folderpath = sg.PopupGetFolder(
            message="Enter Folder to Load",
            title="Load Previous Results"
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

    metadata_filepath = join(results_folderpath, "metadata.yml")
    metadata = loadConfig(metadata_filepath)
    try:
        simulation_type = metadata["simulation_type"]
    except TypeError:
        simulation_type = "grid"

    variable_objs_filepath = join(results_folderpath, "Variable.json")
    variable_objs = loadVariables(variable_objs_filepath)

    stem2path_func = {
        Path(filepath).stem: filepath
        for filepath in equation_filepaths
    }
    functions_objs_filepath = join(results_folderpath, "Function.json")
    function_objs = loadFunctions(functions_objs_filepath, stem2path_func)

    stem2path_param = {
        Path(filepath).stem: filepath
        for filepath in parameter_filepaths
    }
    parameter_objs_filepath = join(results_folderpath, "Parameter.json")
    parameter_objs = loadParameters(parameter_objs_filepath, stem2path_param)

    model = Model(
        variables=variable_objs,
        functions=function_objs,
        parameters=parameter_objs
    )

    free_parameters_filepath = join(results_folderpath, "FreeParameter.json")
    free_parameters_metadata = loadConfig(free_parameters_filepath)
    if simulation_type == "grid":
        free_parameter_name2values = {}
        for name, value in free_parameters_metadata.items():
            free_parameter_name2values[name] = np.array(list(map(float, value["values"])))

        results_obj = GridResults(
            model=model,
            folderpath=results_folderpath,
            free_parameter_name2values=free_parameter_name2values
        )
    elif simulation_type == "optimization":
        free_parameter_name2quantity = {}
        for free_parameter_name, free_parameter_metadata in free_parameters_metadata.items():
            initial_guess = free_parameter_metadata["initial_guess"]
            unit = free_parameter_metadata["unit"]
            bounds = tuple(free_parameter_metadata["bounds"])

            free_parameter_obj = FreeParameter(
                name=free_parameter_name,
                default_value=initial_guess,
                unit=unit,
                bounds=bounds
            )
            free_parameter_quantity = free_parameter_obj.getQuantity()
            free_parameter_name2quantity[free_parameter_name] = free_parameter_quantity
            model.addPaperQuantities(free_parameter_obj)

        fitdata_filepath = join(results_folderpath, "FitData.npy")

        fit_axis_quantity_filepath = join(results_folderpath, "FitAxisQuantity.json")
        fit_axis_quantity_metadata = loadConfig(fit_axis_quantity_filepath)
        fit_parameter_names = fit_axis_quantity_metadata["fit_parameter_names"]
        fit_axis_quantity_metadata = generateMetadataFromFile(fit_axis_quantity_filepath)
        
        general_metadata_filepath = join(results_folderpath, "metadata.yml")
        general_metadata = loadConfig(general_metadata_filepath)
        sample_sizes = tuple(general_metadata["sample_sizes"])
        
        results_obj = OptimizationResults(
            model,
            folderpath=results_folderpath,
            free_parameter_name2quantity=free_parameter_name2quantity,
            fit_parameter_names=fit_parameter_names,
            fitdata_filepath=fitdata_filepath,
            fit_axis_quantity_metadata=fit_axis_quantity_metadata,
            sample_sizes=sample_sizes
        )

    return results_obj, results_folderpath


def getSimulation(
    results_filepath: str = None,
    parameter_directory: str = None,
    equation_directory: str = None
) -> Union[GridSimulationWindowRunner, OptimizationSimulationWindowRunner]:
    results_obj, results_filepath = getResults(
        results_folderpath=results_filepath,
        parameter_directory=parameter_directory,
        equation_directory=equation_directory
    )
    model = results_obj.getModel()

    free_parameters_filepath = join(results_filepath, "FreeParameter.json")
    free_parameter_contents = loadConfig(free_parameters_filepath)

    if isinstance(results_obj, GridResults):
        free_parameter_name2metadata = {}
        for name, value in free_parameter_contents.items():
            quantity = Quantity(0, value["unit"])
            values = list(map(float, value["values"]))
            minimum, maximum, stepcount = min(values), max(values), len(values)
            free_parameter_name2metadata[name] = (minimum, maximum, stepcount, quantity)
        free_parameter_names = list(free_parameter_name2metadata.keys())
        
        plot_parameter_names = free_parameter_names
    elif isinstance(results_obj, OptimizationResults):
        free_parameter_name2quantity = {}
        for name, value in free_parameter_contents.items():
            default_value = value["initial_guess"]
            unit = value["unit"]
            quantity = Quantity(default_value, unit)
            free_parameter_name2quantity[name] = quantity
        free_parameter_names = list(free_parameter_name2quantity.keys())
        
        fit_axis_quantity_filepath = join(results_filepath, "FitAxisQuantity.json")
        fit_axis_quantity = generateMetadataFromFile(fit_axis_quantity_filepath)
        fitdata_filepath = join(results_filepath, "FitData.npy")
        
        results_file_handler: OptimizationResultsFileHandler = results_obj.getResultsFileHandler()
        fit_parameter_names = results_file_handler.getFitParameterNames()
        sample_sizes = results_file_handler.getSampleSizes()
        
        plot_parameter_names = fit_parameter_names

    plot_choices = {
        "Variable": model.getVariables(return_type=str) + ['t'],
        "Function": model.getFunctionNames(),
        "Parameter": plot_parameter_names
    }

    if isinstance(results_obj, GridResults):
        simulation_window = GridSimulationWindowRunner(
            "Simulation from Previous Results",
            results_obj=results_obj,
            plot_choices=plot_choices,
            include_simulation_tab=False,
            free_parameter_name2metadata=free_parameter_name2metadata
        )
    elif isinstance(results_obj, OptimizationResults):
        simulation_window = OptimizationSimulationWindowRunner(
            "Simulation from Previous Results",
            results_obj=results_obj,
            plot_choices=plot_choices,
            include_simulation_tab=False,
            free_parameter_name2quantity=free_parameter_name2quantity,
            fit_parameter_names=fit_parameter_names,
            fit_axis_quantity=fit_axis_quantity,
            fitdata_filepath=fitdata_filepath,
            sample_sizes=sample_sizes
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
    sg.ChangeLookAndFeel("DarkGrey13")
    sg.SetOptions(
        element_padding=(1, 1),
        suppress_error_popups=True,
        suppress_raise_key_errors=False
    )

    loadSimulation(r"D:\Marcinik\Recreation\Arnold Tongue\best_fit\fit6_5param\res6_7s_7000")
