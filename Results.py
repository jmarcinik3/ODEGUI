import itertools
import sys
from functools import partial
from math import prod
from os import makedirs
from os.path import dirname, exists, join
from typing import Callable, Dict, Iterable, List, Tuple, Union

import dill
import numpy as np
import PySimpleGUI as sg
from numpy import ndarray
from sympy import Expr, Symbol
from sympy.utilities.lambdify import lambdify

from CustomMath import *
from Function import Model, Parameter
from macros import StoredObject, recursiveMethod
from YML import loadConfig, saveConfig


class Transform(StoredObject):
    def __init__(self, name: str, transform_info: dict) -> None:
        """
        Constructor for :class:`~Results.Transform`.

        :param name: name of transform
        :param transform_info: dictionary of information to generate transform object
        """
        super().__init__(name)

        transform_info_keys = list(transform_info.keys())
        
        module_name = transform_info["module"]
        module = sys.modules[module_name]
        
        transform_function_name = transform_info["function"]
        transform_function = getattr(module, transform_function_name)
        
        if "time_function" in transform_info_keys:
            time_function_name = transform_info["time_function"]
        else:
            time_function_name = transform_function_name
        time_function = getattr(module, time_function_name)

        if "requires_times" in transform_info_keys:
            requires_times = transform_info["requires_times"]
        else:
            requires_times = False
        
        argument_count = transform_info["arguments"]
        
        self.module = module
        self.transform_function = transform_function
        self.time_function = time_function
        self.argument_count = argument_count
        self.requires_times = requires_times

    def getFunction(self) -> Callable:
        return self.transform_function

    def getTimeFunction(self) -> Callable:
        return self.time_function

    def getArgumentCount(self) -> int:
        return self.argument_count

    def requiresTimes(self) -> bool:
        return self.requires_times

class Coordinate(StoredObject):
    def __init__(self, name: str, coordinate_info: dict) -> None:
        """
        Constructor for :class:`~Results.Transform`.

        :param name: name of coordinate (e.g. "Cartesian")
        :param transform_info: dictionary of information to generate coordinate object
        """
        super().__init__(name)

        module_name = coordinate_info["module"]
        module = sys.modules[module_name]
        coordinate_function_name = coordinate_info["function"]
        coordinate_function = getattr(module, coordinate_function_name)

        self.coordinate_function = coordinate_function

    def getFunction(self) -> Callable:
        return self.coordinate_function


class Results:
    """
    This class stores results from an ODE simulation.
    Minimally, results for temporal variables are required to be set.
    Other results may be calculated and saved from :class:`~Function.Model` as needed.

    :ivar results: 2D dictionary of results.
        First key is indicies of free parameter for current data to present.
        Second key is name of quantity to retrieve results of.
        Value is array of quantitiy values over time.
    :ivar model: :class:`~Function.Model` to calculated results from
    :ivar free_parameter_values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
    :ivar general_equilibrium_expressions: dictionary of symbolic equilibrium expressions.
        Key is name of variable.
        Value is symbolic expression.
        This attribute is so that equilibria only need to be calculated once.
        They are reused after their initial calculation.
    """

    def __init__(
        self,
        model: Model,
        free_parameter_values: Dict[str, ndarray],
        transform_config_filepath: str = "transforms.json",
        coordinate_config_filepath: str = "coordinates.json",
        folderpath: str = None,
        results: dict = None
    ) -> None:
        """
        Constructor for :class:`~Results.Results`

        :param model: :class:`~Function.Model` to calculate results from
        :param free_parameter_values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
        :param folderpath: folder path containing relevant Results files.
            Save and load here.
        """

        self.folderpath = folderpath
        self.data_foldername = "data"
        self.stepcount = None
        self.results = {}
        if results is not None:
            for index, result in results.items():
                self.setResults(index, result)

        transform_config = loadConfig(transform_config_filepath)
        for transform_name, transform_info in transform_config.items():
            transform_obj = Transform(transform_name, transform_info)

        coordinate_config = loadConfig(coordinate_config_filepath)
        for coordinate_name, coordinate_info in coordinate_config.items():
            coordinate_obj = Coordinate(coordinate_name, coordinate_info)

        self.model = model
        self.variable_names = model.getVariables(return_type=str)
        self.function_names = model.getFunctionNames()

        self.general_function_expressions = {}
        self.free_parameter_values = free_parameter_values
        self.general_equilibrium_expressions = {}

    @staticmethod
    def updateProgressMeter(current_value: int, max_value: int, title: str) -> sg.OneLineProgressMeter:
        """
            Update progress meter.

            :param title: title to display in progress meter window
            :param current_value: present number of simulation being calculated
            :param max_value: total number of simulations to calculate
            """
        return sg.OneLineProgressMeter(
            title=title,
            orientation="horizontal",
            current_value=current_value,
            max_value=max_value,
            keep_on_top=True
        )

    def getFolderpath(self) -> str:
        """
        Get folderpath to save/load results into/from.

        :param self: :class:`~Results.Results` to retrieve folderpath from
        """
        if self.folderpath is not None:
            folderpath = self.folderpath
        else:
            folderpath = sg.PopupGetFolder(
                message="Enter Folder to Load",
                title="Load Previous Results"
            )
            self.folderpath = folderpath

        return folderpath

    def setFolderpath(self, folderpath: str) -> None:
        """
        Set folderpath to save/load results into/from.

        :param self: :class:`~Results.Results` to retrieve folderpath from
        :param folderpath: folderpath to save/load 
        """
        self.folderpath = folderpath

    def getModel(self) -> Model:
        """
        Get associated :class:`~Function.Model`.

        :param self: :class:`~Results.Results` to retrieve associated :class:`~Function.Model` from
        """
        return self.model

    def getFreeParameterIndex(self, name: str) -> int:
        """
        Get index of free parameter within collection of free-parameter names.

        :param self: :class:`~Results.Results` to retreive free-parameter names from
        :param name: name of free parameter to retreive index of
        """
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_index = free_parameter_names.index(name)
        return free_parameter_index

    def getFreeParameterNames(self) -> List[str]:
        """
        Get names of free parameters.

        :param self: :class:`~Results.Results` to retrieve free-parameter names from
        """
        return list(self.free_parameter_values.keys())

    def getFreeParameterValues(
        self,
        names: Union[str, Iterable[str]] = None,
        output_type: type = list
    ) -> Union[ndarray, Dict[str, ndarray]]:
        """
        Get values for a free parameter.

        :param self: :class:`~Results.Results` to retreive value from
        :param names: name(s) of parameter to retreive values for
        :param output_type: iterable to output as
            if :paramref:`~Results.Results.getFreeParameterValues.names` is iterable
        """
        def get(name: str) -> ndarray:
            """Base method for :meth:`~Results.Results.getFreeParameterValues`"""
            return self.free_parameter_values[name]

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=output_type,
            default_args=self.getFreeParameterNames()
        )

    def getFreeParameterSubstitutions(
        self,
        index: Union[tuple, Tuple[int, ...]]
    ) -> Dict[Symbol, float]:
        """
        Get substitutions for free parameters at index.

        :param self: :class:`~Results.Results` to retrieve substitutions from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_substitutions = {}
        for parameter_location, free_parameter_name in enumerate(free_parameter_names):
            parameter_index = index[parameter_location]
            parameter_values = self.getFreeParameterValues(names=free_parameter_name)
            parameter_value = parameter_values[parameter_index]
            parameter_symbol = Symbol(free_parameter_name)
            free_parameter_substitutions[parameter_symbol] = parameter_value
        return free_parameter_substitutions

    def getParameterSubstitutions(
        self,
        index: Union[tuple, Tuple[int, ...]] = None,
        name: str = None,
        include_nonfree: bool = True,
        include_free: bool = False,
    ) -> Dict[Symbol, float]:
        """
        Get substitutions from parameter symbol to parameter value.

        :param self: :class:`~Results.Results` to retrieve values from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param name: name of function to retrieve parameter names from.
            Returns substitutions for all parameters in model if None.
            Returns substitutions only for parameters in function if str.
        :param include_free: set True to include substitutions for free parameters.
            Set False to exclude them.
            :paramref:`~Results.Results.getParameterSubstitutions.index` must be given if set to True.
        :param include_nonfree: set True to include substitutions for non-free parameters.
            Set False to exclude them.
        """
        model = self.getModel()

        if name is None:
            parameter_names = None
        else:
            function = model.getFunctions(names=name)
            parameter_names = function.getFreeSymbols(
                species=Parameter,
                expanded=True,
                return_type=str
            )

        substitutions = {}
        if include_free:
            free_substitutions = self.getFreeParameterSubstitutions(index)
            substitutions.update(free_substitutions)
        if include_nonfree:
            free_parameter_names = self.getFreeParameterNames()
            nonfree_substitutions = model.getParameterSubstitutions(
                parameter_names,
                skip_parameters=free_parameter_names
            )
            substitutions.update(nonfree_substitutions)

        return substitutions

    def setEquilibriumExpressions(
        self,
        equilibrium_expressions: Dict[Symbol, Expr] = None
    ) -> None:
        """
        Set symbolic expressions for equilibrium variables.
        These expressions are simplified, except variables and free parameters are kept symbolic.
        Equilibria may be set manually or calculated automatically from the stored :class:`~Function.Model`.

        :param self: :class:`~Results.Results` to set equilibria for
        :param equilibrium_expressions: dictionary of equilibria for variables.
            Key is symbolic variable.
            Value is symbolic equilibrium expression of variable.
        """
        if equilibrium_expressions is None:
            free_parameter_names = self.getFreeParameterNames()
            model = self.getModel()
            solutions = model.getEquilibriumSolutions(skip_parameters=free_parameter_names)
            self.general_equilibrium_expressions = solutions
        else:
            self.general_equilibrium_expressions = equilibrium_expressions

    def getEquilibriumExpression(
        self,
        index: Union[tuple, Tuple[int, ...]],
        name: Union[Symbol, str]
    ) -> Expr:
        """
        Get equilibrium expression for a variable.

        :param self: :class:`~Results.Results` to retrieve equilibrium from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param name: name of variable to retrieve equilibrium for
        """
        if len(self.general_equilibrium_expressions.keys()) == 0:
            self.setEquilibriumExpressions()

        if isinstance(name, Symbol):
            general_expression = self.general_equilibrium_expressions[name]
        elif isinstance(name, str):
            symbol = Symbol(name)
            general_expression = self.general_equilibrium_expressions[symbol]
        else:
            raise TypeError("name must be sp.Symbol or str")

        parameter_substitutions = self.getParameterSubstitutions(
            index=index,
            include_free=True,
            include_nonfree=False
        )
        simplified_expression = general_expression.subs(parameter_substitutions)
        return simplified_expression

    def getGeneralFunctionExpression(self, name: str) -> Expr:
        """
        Get expression for function, with values for non-free parameter substituted in.

        :param self: :class:`~Results.Results` to retrieve expression from
        :param name: name of function to retrieve expression for
        """
        try:
            expression = self.general_function_expressions[name]
        except KeyError:
            function = self.getModel().getFunctions(names=name)
            expression = function.getExpression(
                expanded=False,
                substitute_dependents=True
            )
            parameter_substitutions = self.getParameterSubstitutions(
                include_nonfree=True,
                include_free=False
            )
            expression = expression.subs(parameter_substitutions)
            self.general_function_expressions[name] = expression

        return expression

    def resetResults(self) -> None:
        """
        Reset results to store a new set of them.

        :param self: :class:`~Results.Results` to reset results for
        """
        self.stepcount = None
        self.results = {}

    def getSubstitutedResults(
        self,
        index: Union[tuple, Tuple[int]],
        expression: Expr,
        name: str = None
    ) -> ndarray:
        """
        Get results from simulation for function, after substituting results from variables.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param expression: function to substitute results into
        :param name: name of function where expression was derived from
        """
        parameter_substitutions = self.getParameterSubstitutions(
            index=index,
            name=name,
            include_nonfree=True,
            include_free=True
        )
        expression_sub = expression.subs(parameter_substitutions)

        model = self.getModel()
        variables = model.getVariables(
            time_evolution_types="Temporal",
            return_type=Symbol
        )
        expression_lambda = lambdify(
            [[Symbol("t"), *variables]],
            expression_sub,
            modules=["numpy", "scipy"]
        )

        variable_names = list(map(str, variables))
        temporal_results = self.getResultsOverTime(
            index,
            quantity_names=variable_names
        )
        times = self.getResultsOverTime(
            index,
            quantity_names="t"
        )
        times = times.reshape((1, times.size))
        arguments = np.append(
            times,
            temporal_results,
            axis=0
        )
        results = expression_lambda(arguments)

        return np.array(results)

    def getFunctionResults(
        self,
        index: Union[tuple, Tuple[int]],
        name: str
    ) -> ndarray:
        """
        Get results from simulation for function.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: :meth:`~Results.Results.getResultsOverTime`
        :param name: name of function to retrieve results of
        """
        expression = self.getGeneralFunctionExpression(name)
        parameter_substitutions = self.getParameterSubstitutions(
            index,
            include_nonfree=False,
            include_free=True
        )
        expression = expression.subs(parameter_substitutions)

        if expression.is_constant():
            time_count = len(self.getResultsOverTime(index, 't'))
            constant = float(expression)
            updated_results = np.repeat(constant, time_count)
        else:
            free_symbols = expression.free_symbols
            free_symbol_names = list(map(str, free_symbols))
            expression_lambda = lambdify(
                [free_symbols],
                expression,
                modules=["numpy", "scipy"]
            )
            substitutions_results = self.getResultsOverTime(
                index=index,
                quantity_names=free_symbol_names
            )
            updated_results = expression_lambda(substitutions_results)

        return updated_results

    def getEquilibriumVariableResults(
        self,
        index: Union[tuple, Tuple[int]],
        name: str
    ) -> ndarray:
        """
        Get results from simulation for variable in equilibrium.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: :meth:`~Results.Results.getResultsOverTime`
        :param name: name of variable to retrieve results of
        """
        equilibrium_expression = self.getEquilibriumExpression(index, name)
        results = self.getSubstitutedResults(
            index,
            equilibrium_expression
        )
        return np.array(results)

    def getConstantVariableResults(
        self,
        index: Union[tuple, Tuple[int]],
        name: str
    ) -> ndarray:
        """
        Get results from simulation for constant variable.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: :meth:`~Results.Results.getResultsOverTime`
        :param name: name of variable to retrieve results of
        """
        model = self.getModel()
        variable_obj = model.getVariables(names=name)
        initial_condition = variable_obj.getInitialCondition()
        time_count = self.getResultsOverTime(index, "t").size
        results = np.repeat(initial_condition, time_count)
        return results

    def getResultFilepath(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        folderpath: str = None
    ):
        """
        Get filepath to save/load result from.

        :param self: :class:`~Results.Results` to retrieve filepath from
        :param index: :meth:`~Results.Results.getResultsOverTime`
        :param quantity_name: name of quantity to retrieve filepath for
        :param folderpath: path of folder to save result into.
            Defaults to loaded folder path.
        """
        if folderpath is None:
            folderpath = self.getFolderpath()

        result_folderpath = join(
            folderpath,
            self.data_foldername,
            *list(map(str, index))
        )
        file_extension = ".pkl"
        filename = quantity_name + file_extension
        filepath = join(result_folderpath, filename)

        return filepath

    @staticmethod
    def getCoordinateTransformOfResults(
        results: ndarray,
        coordinate_name: str
    ) -> ndarray:
        """
        Get coordinate transform of results.

        :param results: 1D ndarray of results
        :param coordinate_name: see :class:`~Results.Results.getResultsOverTime`
        """
        coordinate_obj = Coordinate.getInstances(names=coordinate_name)
        coordinate_function = coordinate_obj.getFunction()
        coordinate_results = coordinate_function(results)
        return coordinate_results

    def getTransformOfResults(
        self,
        results: ndarray,
        transform_name: str,
        is_time: bool,
        index: Tuple[tuple, Tuple[int, ...]] = None,
        inequality_filters: Iterable[Tuple[str, str, float]] = None
    ) -> ndarray:
        """
        Get math transform of results.

        :param self: :class:`~Results.Results` to retrieve transform from
        :param results: 1D ndarray if :paramref:`~Results.Results.getTransformOfResults.argument_count`==1;
            tuple of two 1D-ndarrays with same shape if :paramref:`~Results.Results.getTransformOfResults.argument_count`==2.
        :param transform_name: see :class:`~Results.Results.getResultsOverTime`
        :param is_time: set True if results are times for simulation. Set False otherwise.
        :param index: see :meth:`~Results.Results.getResultsOverTime`.
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        :param inequality_filters: see :meth:`~Results.Results.getResultsOverTime`
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        """
        if transform_name != "None":
            transform_obj: Transform = Transform.getInstances(names=transform_name)
            argument_count = transform_obj.getArgumentCount()
            transform_requires_times = transform_obj.requiresTimes()

            if is_time:
                transform_time_function = transform_obj.getTimeFunction()
                transform_results = transform_time_function(results)
            else:
                transform_function = transform_obj.getFunction()

                if transform_requires_times:
                    times = self.getResultsOverTime(
                        index=index,
                        quantity_names='t',
                        inequality_filters=inequality_filters
                    )
                    transform_results = transform_function(results, times)
                else:
                    transform_results = transform_function(results)

        return transform_results

    def getResultsOverTime(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_names: Union[str, List[str]] = None,
        inequality_filters: Iterable[Tuple[str, str, float]] = None,
        coordinate_name: str = "None",
        transform_name: str = "None",
        functional_name: str = "None",
        functional_kwargs: dict = None
    ) -> Union[float, ndarray]:
        """
        Get results for variable or function over time.
        Results are evaluated from simulation.
        Optionally filter results based on variable/function values (in terms of inequalities).
        Optionally perform a transformation on the quantity.
        Optionally condense variable values into one value.

        __Recursion Base__
            return results for single variable: names [str]

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param quantity_names: name(s) of variable/function(s) to retrieve results for
        :param coordinate_name: name of coordinate transform to perform on results.
        :param transform_name: transform to perform on results.
            "Fourier" - Fourier transform.
        :param functional_name: analysis to perform on results to reduce into one (or few) floats.
            "Frequency" - calculate frequency of results.
            "Mean" - calculate Holder mean of results.
            "Standard Deviation" - calculate standard deviation of results.
        :param functional_kwargs: dictionary of optional arguments to pass into corresponding condensing function
        :param inequality_filters: iterable of tuples indicating filters for results.
            First element of tuple is variable/function name.
            Second element of tuple is inequality sign as string.
            Third element of tuple is float.
            Example: ('t', '>', 1.0) includes only data where time (t) is greater than 1
        """
        if isinstance(quantity_names, str):
            try:
                filepath = self.getResultFilepath(index, quantity_names)
                file = open(filepath, 'rb')
                single_results = dill.load(file)
            except FileNotFoundError:
                model = self.getModel()
                if quantity_names in self.variable_names:
                    variable_obj = model.getVariables(names=quantity_names)
                    time_evolution_type = variable_obj.getTimeEvolutionType()
                    results_handles = {
                        "Equilibrium": self.getEquilibriumVariableResults,
                        "Constant": self.getConstantVariableResults,
                        "Function": self.getFunctionResults,
                    }

                    # noinspection PyArgumentList
                    single_results = results_handles[time_evolution_type](
                        index,
                        quantity_names
                    )
                elif quantity_names in self.function_names:
                    single_results = self.getFunctionResults(
                        index,
                        quantity_names
                    )
                else:
                    raise ValueError("quantity_names input must correspond to either variable or function when str")

                # noinspection PyUnboundLocalVariable
                self.saveResult(
                    index,
                    quantity_names,
                    single_results
                )

            if inequality_filters is not None:
                if len(inequality_filters) >= 1:
                    for inequality_index, inequality_filter in enumerate(inequality_filters):
                        filter_quantity_name, filter_inequality_type, filter_float = inequality_filter
                        filter_results = self.getResultsOverTime(
                            index=index,
                            quantity_names=filter_quantity_name,
                            inequality_filters=None
                        )

                        if inequality_index == 0:
                            stepcount = filter_results.shape[-1]
                            filter_intersection = np.arange(stepcount)

                        new_filter_indicies = eval(f"np.where(filter_results{filter_inequality_type:s}{filter_float:})")
                        filter_intersection = np.intersect1d(
                            filter_intersection,
                            new_filter_indicies
                        )

                    single_results = single_results[filter_intersection]

            if coordinate_name != "None":
                single_results = self.getCoordinateTransformOfResults(
                    single_results,
                    coordinate_name=coordinate_name
                )

            if transform_name != "None":
                is_time = quantity_names == 't'
                single_results = self.getTransformOfResults(
                    single_results,
                    transform_name=transform_name,
                    is_time=is_time,
                    index=index,
                    inequality_filters=inequality_filters
                )

            if functional_name != "None":
                if functional_name == "Frequency":
                    times = self.getResultsOverTime(
                        index=index,
                        quantity_names="t",
                        transform_name=transform_name,
                        inequality_filters=inequality_filters
                    )
                    frequency = oscillationFrequency(
                        single_results,
                        times,
                        **functional_kwargs
                    )
                    return frequency
                elif functional_name == "Mean":
                    return holderMean(
                        single_results,
                        **functional_kwargs
                    )
                elif functional_name == "Standard Deviation":
                    return np.std(single_results)
                else:
                    raise ValueError(f"invalid functional name ({functional_name:s})")

            return single_results
        elif isinstance(quantity_names, list):
            print("multi:", quantity_names, transform_name)
            if transform_name == "None":
                new_results = np.array([
                    self.getResultsOverTime(
                        index=index,
                        quantity_names=name
                    )
                    for name in quantity_names
                ])
                print("shape:", new_results.shape)
            else:
                pass
            return new_results
        else:
            raise TypeError("names input must be str or list")

    def getResultsOverTimePerParameter(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_names: str,
        quantity_names: Union[str, List[str]],
        **kwargs
    ) -> Tuple[tuple, ndarray]:
        """
        Get free-parameter values and "averaged" quantity values.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param parameter_names: name(s) of free parameter(s) to retrieve quantity results over.
        :param quantity_names: name(s) of quantity(s) to average results over
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        :returns: tuple of results.
            First index gives array of parameter base values.
                nth index of this array gives values for nth parameter in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.parameter_names`.
            Second index gives matrix quantity results.
                nth index of this matrix gives values for nth quantity in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.quantity_names`.
        """
        if isinstance(quantity_names, str):
            quantity_names = [quantity_names]
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        per_parameter_locations = np.array(list(map(self.getFreeParameterIndex, parameter_names)))
        per_parameter_base_values = tuple(list(self.getFreeParameterValues(names=parameter_names)))
        per_parameter_stepcounts = tuple(map(len, per_parameter_base_values))
        per_parameter_partial_indicies = list(itertools.product(*map(range, per_parameter_stepcounts)))

        default_index = np.array(index)
        times = self.getResultsOverTime(
            index=index,
            quantity_names="t",
            **kwargs
        )
        if isinstance(times, (float, int)):
            single_result_size = per_parameter_stepcounts
        elif isinstance(times, ndarray):
            single_result_size = (*per_parameter_stepcounts, *times.shape)
        else:
            raise TypeError(f"invalid type ({type(times):})")

        quantity_count = len(quantity_names)
        simulation_count_per_quantity = prod(list(per_parameter_stepcounts))
        simulation_count = quantity_count * simulation_count_per_quantity

        results = np.zeros((quantity_count, *single_result_size))
        updateProgressMeter = partial(
            self.updateProgressMeter,
            title="Calculating Simulation",
            max_value=simulation_count
        )

        simulation_index_flat = 0
        for quantity_location, quantity_name in enumerate(quantity_names):
            single_results = np.zeros(single_result_size)
            for partial_index_flat, partial_index in enumerate(per_parameter_partial_indicies):
                simulation_index_flat += 1  # quantity_location * simulation_count_per_quantity + partial_index_flat + 1

                if simulation_index_flat % 100 == 0:
                    if not updateProgressMeter(simulation_index_flat):
                        break

                new_index = default_index
                new_index[per_parameter_locations] = partial_index

                try:
                    single_result = self.getResultsOverTime(
                        index=tuple(new_index),
                        quantity_names=quantity_name,
                        **kwargs
                    )
                    single_results[partial_index] = single_result
                except KeyError:
                    single_results[partial_index] = None

            results[quantity_location] = single_results

        updateProgressMeter(simulation_count)

        return per_parameter_base_values, results

    def getStepcount(self) -> float:
        """
        Get number of steps per result.

        :param self: :class:`~Results.Results` to retrieve stepcount from
        """
        return self.stepcount

    def setStepcount(self, count: float) -> None:
        """
        Set step count per results for new set of data.

        :param self: :class:`~Results.Results` to set step count in
        :param count: new step count for results object
        """
        self.stepcount = count

    def setResults(
        self,
        index: Union[tuple, Tuple[int]],
        results: Union[ndarray, Dict[str, ndarray]],
        name: str = None
    ) -> None:
        """
        Save results from simulation.

        :param self: :class:`~Results.Results` to save results in
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param results: results to save at :paramref:`~Results.Results.setResults.index`.
            This must be a list of floats if name is specified.
            This must be a dictionary if name is not specified.
                Key is name of variable.
                Value is list of floats for variable over time.
        :param name: name of quantity to set results for
        """
        if isinstance(name, str):
            results_size = results.size
            if self.getStepcount() is None:
                self.setStepcount(results_size)

            stepcount = self.getStepcount()
            assert results_size == stepcount

            if index not in self.results.keys():
                self.results[index] = {}

            self.results[index][name] = results
        elif name is None:
            for name, result in results.items():
                self.setResults(index, result, name=name)

    def saveResult(
        self,
        index: Union[tuple, Tuple[int]],
        name: str,
        result: ndarray,
        folderpath: str = None
    ) -> None:
        """
        Save single result into file.
        :param self: :class:`~Results.Results` to retrieve save folder from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param name: name of quantity to set results for
        :param result: single array of results for quantity
        :param folderpath: path of folder to save results into.
            Defaults to loaded folder path.
        """
        if folderpath is None:
            folderpath = self.getFolderpath()

        result_filepath = self.getResultFilepath(
            index,
            name,
            folderpath=folderpath
        )

        result_folderpath = dirname(result_filepath)
        if not exists(result_folderpath):
            makedirs(result_folderpath)

        if not exists(result_filepath):
            with open(result_filepath, 'wb') as result_file:
                dill.dump(result, result_file)

    def saveResults(
        self,
        folderpath: str = None
    ) -> None:
        """
        Save results object (self) into folder.

        :param self: :class:`~Results.Results` to save into file
        :param folderpath: path of folder to save results into.
            Defaults to loaded folder path.
        """
        if folderpath is None:
            folderpath = self.getFolderpath()

        model = self.getModel()
        parameter_values = self.getFreeParameterValues(output_type=dict)

        free_parameter_info = {}
        for name, values in parameter_values.items():
            parameter = model.getParameters(names=name)
            parameter_quantity = parameter.getQuantity()
            unit = str(parameter_quantity.to_base_units().units)
            free_parameter_info[name] = {
                "index": self.getFreeParameterIndex(name),
                "values": list(map(str, values)),
                "unit": unit
            }

        function_filepath = join(folderpath, "Function.json")
        if not exists(function_filepath):
            function_file = model.saveFunctionsToFile(function_filepath)

        parameter_filepath = join(folderpath, "Parameter.json")
        if not exists(parameter_filepath):
            parameter_file = model.saveParametersToFile(parameter_filepath)

        variable_filepath = join(folderpath, "Variable.json")
        if not exists(variable_filepath):
            variable_file = model.saveVariablesToFile(variable_filepath)

        free_parameter_filepath = join(folderpath, "FreeParameter.json")
        if not exists(free_parameter_filepath):
            free_parameter_file = saveConfig(free_parameter_info, free_parameter_filepath)
