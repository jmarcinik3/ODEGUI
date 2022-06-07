from __future__ import annotations

import itertools
import os
import sys
import time
from functools import partial
from math import prod
from os.path import exists, join
from typing import Callable, Dict, Iterable, List, Tuple, Union

import h5py
import numpy as np
import PySimpleGUI as sg
from numpy import ndarray
from sympy import Expr, Symbol
from sympy.utilities.lambdify import lambdify

from Function import Model, Parameter, Variable
from macros import StoredObject, recursiveMethod
from Transforms.CustomMath import normalizeOverAxes
from YML import loadConfig, saveConfig


class FunctionOnResult:
    def __init__(self, name: str, info: dict) -> None:
        """
        Constructor for :class:`~Results.FunctionOnResult`.

        :param name: name of function
        :param info: dictionary of information to generate object
        """
        info_keys = list(info.keys())

        module_name = info["module"]
        module = sys.modules[module_name]
        function_name = info["function"]
        function = getattr(module, function_name)

        self.function = function

        if "requires_times" in info_keys:
            requires_times = info["requires_times"]
        else:
            requires_times = False
        self.requires_times = requires_times

        if "requires_parameters" in info_keys:
            requires_parameters = info["requires_parameters"]
        else:
            requires_parameters = False
        self.requires_parameters = requires_parameters

    def getFunction(self) -> Callable:
        """
        Get function, pre-substituting in times if required.

        :param self: :class:`~Results.FunctionOnResult` to retreive function from
        """
        return self.function

    def requiresTimes(self) -> bool:
        """
        Get whether function requires times as additional input argument.

        :param self: :class:`~Results.Results.FunctionOnResults` to retrieve boolean from
        """
        return self.requires_times

    def requiresParameters(self) -> bool:
        """
        Get whether function requires parameter values as additional input argument.

        :param self: :class:`~Results.Results.FunctionOnResults` to retrieve boolean from
        """
        return self.requires_parameters


class Transform(FunctionOnResult, StoredObject):
    def __init__(self, name: str, transform_info: dict) -> None:
        """
        Constructor for :class:`~Results.Transform`.

        :param name: name of transform
        :param transform_info: dictionary of information to generate transform object.
            See :class:`~Results.FunctionOnResult`.
        """
        FunctionOnResult.__init__(self, name, transform_info)
        StoredObject.__init__(self, name)

        transform_info_keys = list(transform_info.keys())

        module_name = transform_info["module"]
        module = sys.modules[module_name]

        transform_function_name = transform_info["function"]

        if "time_function" in transform_info_keys:
            time_function_name = transform_info["time_function"]
        else:
            time_function_name = transform_function_name
        time_function = getattr(module, time_function_name)

        argument_count = transform_info["arguments"]

        self.module = module
        self.time_function = time_function
        self.argument_count = argument_count

    def getTimeFunction(self) -> Callable:
        return self.time_function

    def getArgumentCount(self) -> int:
        return self.argument_count


class Envelope(FunctionOnResult, StoredObject):
    def __init__(self, name: str, envelope_info: dict) -> None:
        """
        Constructor for :class:`~Results.Envelope`.

        :param name: name of envelope (e.g. "Amplitude")
        :param envelope_info: dictionary of information to generate envelope object.
            See :class:`~Results.FunctionOnResult`.
        """
        FunctionOnResult.__init__(self, name, envelope_info)
        StoredObject.__init__(self, name)


class Functional(FunctionOnResult, StoredObject):
    def __init__(self, name: str, functional_info: dict):
        """
        Constructor for :class:`~Results.Functional`.

        :param name: name of functional
        :param functional_info: dictionary of information to generate functional object.
            See :class:`~Results.FunctionOnResult`.
        """
        FunctionOnResult.__init__(self, name, functional_info)
        StoredObject.__init__(self, name)


class Complex(FunctionOnResult, StoredObject):
    def __init__(self, name: str, complex_info: dict):
        """
        Constructor for :class:`~Results.Functional`.

        :param name: name of functional
        :param complex_info: dictionary of information to generate functional object.
            See :class:`~Results.FunctionOnResult`.
        """
        FunctionOnResult.__init__(self, name, complex_info)
        StoredObject.__init__(self, name)


class Results:
    """
    This class stores results from an ODE simulation.
    Minimally, results for temporal variables are required to be set.
    Other results may be calculated and saved from :class:`~Function.Model` as needed.

    :ivar model: :class:`~Function.Model` to calculated results from
    :ivar general_equilibrium_expressions: dictionary of symbolic equilibrium expressions.
        Key is name of variable.
        Value is symbolic expression.
        This attribute is so that equilibria only need to be calculated once.
        They are reused after their initial calculation.
    """

    def __init__(
        self,
        model: Model,
        parameter_name2values: Dict[str, ndarray],
        folderpath: str,
        simulation_type: str,
        transform_config_filepath: str = "transforms/transforms.json",
        envelope_config_filepath: str = "transforms/envelopes.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json",
        stepcount: int = None
    ) -> None:
        """
        Constructor for :class:`~Results.Results`

        :param model: :class:`~Function.Model` to calculate results from
        :param parameter_name2values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
        :param folderpath: folder path containing relevant Results files.
            Save and load here.
        """
        variable_names = model.getVariables(return_type=str)
        function_names = model.getFunctionNames()
        self.results_file_handler = ResultsFileHandler(
            folderpath=folderpath,
            variable_names=variable_names,
            function_names=function_names,
            stepcount=stepcount,
            parameter_name2values=parameter_name2values,
            simulation_type=simulation_type
        )

        transform_config = loadConfig(transform_config_filepath)
        for transform_name, transform_info in transform_config.items():
            transform_obj = Transform(transform_name, transform_info)

        envelope_config = loadConfig(envelope_config_filepath)
        for envelope_name, envelope_info in envelope_config.items():
            envelope_obj = Envelope(envelope_name, envelope_info)

        functional_config = loadConfig(functional_config_filepath)
        for functional_name, functional_info in functional_config.items():
            functional_obj = Functional(functional_name, functional_info)

        complex_config = loadConfig(complex_config_filepath)
        for complex_name, complex_info in complex_config.items():
            complex_obj = Complex(complex_name, complex_info)

        self.model = model

        self.general_function_expressions = {}
        self.general_equilibrium_expressions = {}

    def getResultsFileHandler(self) -> ResultsFileHandler:
        """
        Get object to handle saving/loading files.
        """
        return self.results_file_handler

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
            keep_on_top=True,
            key="-PER PARAMETER PROGRESS-"
        )

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
        results_file_handler = self.getResultsFileHandler()
        free_parameter_names = results_file_handler.getFreeParameterNames()
        free_parameter_index = free_parameter_names.index(name)
        return free_parameter_index

    def getFreeParameterSubstitutions(
        self,
        index: Union[tuple, Tuple[int, ...]]
    ) -> Dict[Symbol, float]:
        """
        Get substitutions for free parameters at index.

        :param self: :class:`~Results.Results` to retrieve substitutions from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        results_file_handler = self.getResultsFileHandler()
        free_parameter_names = results_file_handler.getFreeParameterNames()

        free_parameter_substitutions = {}
        for parameter_location, free_parameter_name in enumerate(free_parameter_names):
            parameter_index = index[parameter_location]
            parameter_values = results_file_handler.getFreeParameterValues(names=free_parameter_name)
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
            results_file_handler = self.getResultsFileHandler()
            free_parameter_names = results_file_handler.getFreeParameterNames()
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
            results_file_handler = self.getResultsFileHandler()
            free_parameter_names = results_file_handler.getFreeParameterNames()
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
            times = self.getResultsOverTime(index, 't')
            time_count = len(times)
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

    def getEnvelopeOfResults(
        self,
        results: ndarray,
        envelope_name: str,
        index: Tuple[tuple, Tuple[int, ...]] = None,
        inequality_filters: Iterable[Tuple[str, str, float]] = None
    ) -> ndarray:
        """
        Get envelope of results.

        :param self: :class:`~Results.Results` to retrieve envelope from
        :param results: 1D ndarray of results
        :param envelope_name: see :class:`~Results.Results.getResultsOverTime`
        :param index: see :meth:`~Results.Results.getResultsOverTime`.
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        :param inequality_filters: see :meth:`~Results.Results.getResultsOverTime`
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        """
        envelope_obj: Envelope = Envelope.getInstances(names=envelope_name)
        envelope_function = envelope_obj.getFunction()

        envelope_requires_times = envelope_obj.requiresTimes()
        if envelope_requires_times:
            times = self.getResultsOverTime(
                index=index,
                quantity_names='t',
                inequality_filters=inequality_filters
            )
            envelope_function = partial(envelope_function, times=times)

        envelope_results = envelope_function(results)

        return envelope_results

    def getFunctionalOfResults(
        self,
        results: ndarray,
        functional_name: str,
        parameters: ndarray = None
    ) -> float:
        """
        Get functional of results.

        :param self: :class:`~Results.Results` to retrieve functional from
        :param results: 1D ndarray of results
        :param functional_name: see :class:`~Results.Results.getResultsOverTime`
        :param parameters: array of parameter values for function.
            Only called if functional requires parameter values.
        """
        functional_obj: Functional = Functional.getInstances(names=functional_name)
        functional_function = functional_obj.getFunction()

        functional_requires_parameters = functional_obj.requiresParameters()
        if functional_requires_parameters:
            functional_function = partial(functional_function, parameters=parameters)

        functional_results = functional_function(results)
        return functional_results

    def getTransformOfResults(
        self,
        results: ndarray,
        transform_name: str,
        is_time: bool = False,
        index: Tuple[tuple, Tuple[int, ...]] = None,
        inequality_filters: Iterable[Tuple[str, str, float]] = None
    ) -> ndarray:
        """
        Get math transform of results.

        :param self: :class:`~Results.Results` to retrieve transform from
        :param results: 1D ndarray if transform function requires exactly one nontime argument;
            tuple of two 1D-ndarrays each with same size, if transform requires two or more nontime arguments.
        :param transform_name: see :class:`~Results.Results.getResultsOverTime`
        :param is_time: set True if result is times for simulation. Set False otherwise.
            Only called if transform function requires exactly one nontime argument.
        :param index: see :meth:`~Results.Results.getResultsOverTime`.
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        :param inequality_filters: see :meth:`~Results.Results.getResultsOverTime`
            Only called if corresponding :class:`~Results.Results.Transform` requires time as input.
        """
        assert isinstance(results, ndarray)
        transform_obj: Transform = Transform.getInstances(names=transform_name)

        dimension_count = results.ndim
        if dimension_count == 1:
            quantity_count = 1
        elif dimension_count == 2:
            quantity_count = results.shape[0]
        else:
            raise ValueError(f"results ({results.__class__:s}) must be either 1D ndarray or 2D-arrays")
        argument_count = transform_obj.getArgumentCount()
        assert argument_count == quantity_count

        if is_time and argument_count == 1:
            transform_function = transform_obj.getTimeFunction()
        else:
            transform_function = transform_obj.getFunction()

            transform_requires_times = transform_obj.requiresTimes()
            if transform_requires_times:
                times = self.getResultsOverTime(
                    index=index,
                    quantity_names='t',
                    inequality_filters=inequality_filters
                )
                transform_function = partial(transform_function, times=times)

        transform_results = transform_function(results)
        return transform_results

    @staticmethod
    def getComplexReductionOfResults(
        results: ndarray,
        complex_name: str
    ) -> float:
        """
        Get complex-reduction method performed on results.

        :param results: 1D ndarray of results
        :param complex_name: see :class:`~Results.Results.getResultsOverTime`
        """
        complex_obj: Complex = Complex.getInstances(names=complex_name)
        complex_function = complex_obj.getFunction()
        complex_results = complex_function(results)

        return complex_results

    def getResultsOverTime(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_names: Union[str, List[str]],
        inequality_filters: Iterable[Tuple[str, str, float]] = None,
        envelope_name: str = "None",
        transform_name: str = "None",
        functional_name: str = "None",
        functional_kwargs: dict = None,
        complex_name: str = "None",
        parameter_functional_names: List[str] = None,
        functional_parameter_namess: List[List[str]] = None,
        close_files: bool = True
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
        :param envelope_name: name of envelope to perform on results.
        :param transform_name: name of transform to perform on results
        :param functional_name: functional to perform on results to reduce into one (or few) floats
        :param functional_kwargs: dictionary of optional arguments to pass into corresponding condensing function
        :param complex_name: name of function to reduce complex numbers into real numbers
        :param inequality_filters: iterable of tuples indicating filters for results.
            First element of tuple is variable/function name.
            Second element of tuple is inequality sign as string.
            Third element of tuple is float.
            Example: ('t', '>', 1.0) includes only data where time (t) is greater than 1.
        :param close_files: set True to immediately close any opened files.
        """
        if parameter_functional_names is None:
            parameter_functional_names = []
        assert isinstance(parameter_functional_names, list)
        for parameter_functional_name in parameter_functional_names:
            assert isinstance(parameter_functional_name, str)
            assert parameter_functional_name != "None"

        if len(parameter_functional_names) >= 1:
            assert isinstance(functional_parameter_namess, list)
            for functional_parameter_names in functional_parameter_namess:
                assert isinstance(functional_parameter_names, list)
                for parameter_name in functional_parameter_names:
                    assert isinstance(parameter_name, str)
            assert len(functional_parameter_namess) == len(parameter_functional_names)

            parameter_names_flat = [
                parameter_name
                for parameter_names in functional_parameter_namess
                for parameter_name in parameter_names
            ]

            parameter_results, results_per_parameter = self.getResultsOverTimePerParameter(
                index=index,
                parameter_names=parameter_names_flat,
                quantity_names=quantity_names,
                inequality_filters=inequality_filters,
                envelope_name=envelope_name,
                transform_name=transform_name,
                functional_name=functional_name,
                functional_kwargs=functional_kwargs,
                complex_name=complex_name
            )
            results = results_per_parameter[0]

            previous_parameter_counts = 0
            for index, parameter_functional_name in enumerate(parameter_functional_names):
                functional_parameter_names = functional_parameter_namess[index]
                parameter_count = len(functional_parameter_names)

                parameter_values = parameter_results[previous_parameter_counts:previous_parameter_counts + parameter_count]

                if len(parameter_values) == 1:
                    parameter_values = parameter_values[0]

                results_shape = results.shape
                parameters_shape = results_shape[:parameter_count]
                residual_shape = results_shape[parameter_count:]
                parameter_value_count = np.prod(parameters_shape)

                getFunctionalOfResults = partial(
                    self.getFunctionalOfResults,
                    functional_name=parameter_functional_name,
                    parameters=parameter_values
                )

                results_shaped = results.reshape((parameter_value_count, *residual_shape))
                results = np.apply_along_axis(
                    getFunctionalOfResults,
                    0,
                    results_shaped
                )

                previous_parameter_counts += parameter_count
        else:
            if isinstance(quantity_names, str):
                try:
                    results_file_handler = self.getResultsFileHandler()
                    results_file = results_file_handler.getResultsFile(name=quantity_names)
                    single_results = results_file[quantity_names][index]

                    is_all_zero = not np.any(single_results)
                    if is_all_zero:
                        raise ValueError(f"numpy array {quantity_names:s} {index} contains only zeros... calculating values")
                except (FileNotFoundError, ValueError) as error:
                    print(error)
                    model = self.getModel()
                    results_file_handler = self.getResultsFileHandler()

                    if quantity_names in results_file_handler.getVariableNames():
                        variable_obj: Variable = model.getVariables(names=quantity_names)
                        time_evolution_type = variable_obj.getTimeEvolutionType()
                        results_handles = {
                            "Equilibrium": self.getEquilibriumVariableResults,
                            "Constant": self.getConstantVariableResults,
                            "Function": self.getFunctionResults,
                        }
                        single_results = results_handles[time_evolution_type](
                            index,
                            quantity_names
                        )
                    elif quantity_names in results_file_handler.getFunctionNames():
                        single_results = self.getFunctionResults(
                            index,
                            quantity_names
                        )
                    elif quantity_names == 't':
                        print("returning NaN for time")
                        return np.nan
                    else:
                        raise ValueError("quantity_names input must correspond to either variable or function when str")

                    results_file_handler.saveResult(
                        index=index,
                        name=quantity_names,
                        result=single_results,
                        close_files=close_files
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

                if envelope_name != "None":
                    single_results = self.getEnvelopeOfResults(
                        single_results,
                        envelope_name=envelope_name,
                        index=index,
                        inequality_filters=inequality_filters
                    )

                quantity_is_time = quantity_names == 't'
                results = single_results
            elif isinstance(quantity_names, list):
                multiple_results = np.array([
                    self.getResultsOverTime(
                        index=index,
                        quantity_names=name,
                        inequality_filters=inequality_filters,
                        envelope_name=envelope_name
                    )
                    for name in quantity_names
                ])

                quantity_is_time = False
                results = multiple_results
            else:
                raise TypeError("names input must be str or list")

            if transform_name != "None":
                results = self.getTransformOfResults(
                    results=results,
                    transform_name=transform_name,
                    is_time=quantity_is_time,
                    index=index,
                    inequality_filters=inequality_filters
                )

            if functional_name != "None":
                results = self.getFunctionalOfResults(
                    results,
                    functional_name=functional_name
                )

            if complex_name != "None":
                results = self.getComplexReductionOfResults(
                    results,
                    complex_name=complex_name
                )

        return results


class GridResults(Results):
    def __init__(
        self,
        model: Model,
        parameter_name2values: Dict[str, ndarray],
        folderpath: str,
        transform_config_filepath: str = "transforms/transforms.json",
        envelope_config_filepath: str = "transforms/envelopes.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json",
        stepcount: int = None
    ) -> None:
        Results.__init__(
            self,
            model=model,
            parameter_name2values=parameter_name2values,
            folderpath=folderpath,
            transform_config_filepath=transform_config_filepath,
            envelope_config_filepath=envelope_config_filepath,
            functional_config_filepath=functional_config_filepath,
            complex_config_filepath=complex_config_filepath,
            stepcount=stepcount,
            simulation_type="grid"
        )

    def getResultsOverTimePerParameter(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_names: Union[str, List[str]],
        quantity_names: Union[str, List[str]],
        transform_name: str = "None",
        normalize_names: Union[str, List[str]] = None,
        show_progress: bool = False,
        **kwargs
    ) -> Tuple[tuple, ndarray]:
        """
        Get free-parameter values and "averaged" quantity values.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param parameter_names: name(s) of free parameter(s) to retrieve quantity results over.
        :param quantity_names: name(s) of quantity(s) to pass into :meth:`~Results.Results.getResultsOverTime`
        :param transform_name: see :class:`~Results.Results.getResultsOverTime`
        :param normalize_names: name(s) of parameter(s) to normalize results over
        :param show_progress: show progress bar during calculation
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
        else:
            assert isinstance(quantity_names, list)
            for quantity_name in quantity_names:
                assert isinstance(quantity_name, str)

        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        else:
            assert isinstance(parameter_names, list)
            for parameter_name in parameter_names:
                assert isinstance(parameter_name, str)

        assert isinstance(transform_name, str)
        if transform_name != "None":
            quantity_names = [quantity_names]

        if normalize_names is None:
            normalize_names = []
        elif isinstance(normalize_names, str):
            normalize_names = [normalize_names]
        else:
            assert isinstance(normalize_names, list)
            for normalize_name in normalize_names:
                assert isinstance(normalize_name, str)
                assert normalize_name in parameter_names

        results_file_handler = self.getResultsFileHandler()

        per_parameter_locations = np.array(list(map(self.getFreeParameterIndex, parameter_names)))
        per_parameter_base_values = tuple(list(results_file_handler.getFreeParameterValues(names=parameter_names)))
        per_parameter_stepcounts = tuple(map(len, per_parameter_base_values))
        per_parameter_partial_indicies = list(itertools.product(*map(range, per_parameter_stepcounts)))

        default_index = np.array(index)
        sample_result = self.getResultsOverTime(
            index=index,
            quantity_names=quantity_names[0],
            transform_name=transform_name,
            **kwargs
        )

        normalize_over_axes = [
            parameter_names.index(normalize_name)
            for normalize_name in normalize_names
        ]
        if isinstance(sample_result, (float, int, np.float32)):
            single_result_size = per_parameter_stepcounts
        elif isinstance(sample_result, ndarray):
            single_result_size = (*per_parameter_stepcounts, *sample_result.shape)
            if len(normalize_over_axes) >= 1:
                normalize_over_axes.append(-1)
        else:
            raise TypeError(f"invalid type ({type(sample_result):})")
        normalize_over_axes = tuple(normalize_over_axes)

        quantity_count = len(quantity_names)
        simulation_count_per_quantity = prod(list(per_parameter_stepcounts))
        simulation_count = quantity_count * simulation_count_per_quantity
        results = np.zeros((quantity_count, *single_result_size))

        if show_progress:
            updateProgressMeter = partial(
                self.updateProgressMeter,
                title="Calculating Simulation",
                max_value=simulation_count
            )

        time_seconds_resolution = 1
        previous_time = time.time()

        for quantity_location, quantity_name in enumerate(quantity_names):
            single_results = np.zeros(single_result_size)
            for partial_index_flat, partial_index in enumerate(per_parameter_partial_indicies):
                current_time = time.time()
                if show_progress and current_time - previous_time >= time_seconds_resolution:
                    previous_time = current_time
                    simulation_index_flat = quantity_location * simulation_count_per_quantity + partial_index_flat + 1
                    if not updateProgressMeter(simulation_index_flat):
                        updateProgressMeter(simulation_count)
                        return

                new_index = default_index
                new_index[per_parameter_locations] = partial_index

                try:
                    single_result = self.getResultsOverTime(
                        index=tuple(new_index),
                        quantity_names=quantity_name,
                        transform_name=transform_name,
                        close_files=False,
                        **kwargs
                    )
                    single_results[partial_index] = single_result
                except KeyError:
                    single_results[partial_index] = None

            single_results = normalizeOverAxes(single_results, normalize_over_axes)
            results[quantity_location] = single_results

        if show_progress:
            updateProgressMeter(simulation_count)

        results_file_handler.closeResultsFiles()

        return per_parameter_base_values, results

    def saveResultsMetadata(self) -> None:
        """
        Save results object (self) into folder.

        :param self: :class:`~Results.Results` to save into file
        :param folderpath: path of folder to save results into.
            Defaults to loaded folder path.
        """
        results_file_handler = self.getResultsFileHandler()
        folderpath = results_file_handler.getFolderpath()

        model = self.getModel()
        parameter_values = results_file_handler.getFreeParameterValues(output_type=dict)

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
        parameter_filepath = join(folderpath, "Parameter.json")
        variable_filepath = join(folderpath, "Variable.json")
        free_parameter_filepath = join(folderpath, "FreeParameter.json")

        function_file = model.saveFunctionsToFile(function_filepath)
        parameter_file = model.saveParametersToFile(parameter_filepath)
        variable_file = model.saveVariablesToFile(variable_filepath)
        free_parameter_file = saveConfig(free_parameter_info, free_parameter_filepath)


class ResultsFileHandler:
    def __init__(
        self,
        folderpath: str,
        variable_names: List[str],
        function_names: List[str],
        parameter_name2values: Dict[str, ndarray],
        stepcount: int = None,
        simulation_type: str = "grid"
    ) -> None:
        """
        Constructor for :class:`~Results.ResultsFileHandler`.

        :param folderpath: root folderpath to save/load results
        :param variable_names: collection of variable names to save/load results for
        :param function_names: collection of function names to save/load results for
        :param parameter_name2values: see :class:`~Results.Results`
        :param stepcount: number of values in each simulation per variable.
            Defaults to size of time array at first index for all parameters.
        :param simulation_type: type of simulation, indicated how to organize results.
            Must be "grid", "optimization".
            "grid": filled, discrete of parameter values is simulated.
            "optimization": simulations are optimized relative to dataset, for collection of parameters.
        """
        assert isinstance(folderpath, str)
        self.folderpath = folderpath

        assert isinstance(variable_names, Iterable)
        for variable_name in variable_names:
            assert isinstance(variable_name, str)
        self.variable_names = variable_names

        assert isinstance(function_names, Iterable)
        for function_name in function_names:
            assert isinstance(function_name, str)
        self.function_names = function_names

        assert isinstance(parameter_name2values, dict)
        for parameter_name, parameter_values in parameter_name2values.items():
            assert isinstance(parameter_name, str)
            assert isinstance(parameter_values, ndarray)
        self.parameter_name2values = parameter_name2values

        assert isinstance(simulation_type, str)
        self.simulation_type = simulation_type.lower()

        if stepcount is None:
            parameter_count = len(parameter_name2values)
            first_index = tuple(list(np.zeros(parameter_count)))
            time_results_filepath = self.getResultFilepath('t', index=first_index)
            with h5py.File(time_results_filepath, 'r') as time_results_file:
                time_results_dataset = time_results_file['t']
                time_results_shape = time_results_dataset.shape
                time_result_index = tuple(np.zeros(len(time_results_shape) - 1, dtype=np.int32))
                time_result = time_results_dataset[time_result_index]

            time_result_size = time_result.size
            self.stepcount = time_result_size
        else:
            assert isinstance(stepcount, int)
            self.stepcount = stepcount

        self.name2file: Dict[str, h5py.File] = {}

    def getSimulationType(self) -> str:
        """
        Get type of simulation executed to produce data.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve type from
        """
        return self.simulation_type

    def getStepcount(self) -> float:
        """
        Get number of steps per result.

        :param self: :class:`~Results.Results` to retrieve stepcount from
        """
        return self.stepcount

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables simulated in results.

        :param self: :class:`~Results.DynamicSaveResults` to retrieve names from
        """
        return self.variable_names

    def getFunctionNames(self) -> List[str]:
        """
        Get names of functions possibly simulated in results.

        :param self: :class:`~Results.DynamicSaveResults` to retrieve names from
        """
        return self.function_names

    def getFreeParameterNames(self) -> List[str]:
        """
        Get names of free parameters.

        :param self: :class:`~Results.Results` to retrieve free-parameter names from
        """
        return list(self.parameter_name2values.keys())

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
            return self.parameter_name2values[name]

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=output_type,
            default_args=self.getFreeParameterNames()
        )

    def getFolderpath(
        self,
        index: Union[tuple, Tuple[int, ...]] = None,
    ) -> str:
        """
        Get folderpath to save/load results into/from.

        :param self: :class:`~Results.Results` to retrieve folderpath from
        :param index: :meth:`~Results.Results.getResultsOverTime` (not implemented)
        """
        if self.folderpath is not None:
            root_folderpath = self.folderpath
        else:
            root_folderpath = sg.PopupGetFolder(
                message="Enter Folder to Load",
                title="Load Previous Results"
            )
            self.folderpath = root_folderpath

        simulation_type = self.simulation_type
        if simulation_type == "grid":
            folderpath = root_folderpath
        elif simulation_type == "optimization":
            subfolder = str(index[0])
            folderpath = join(root_folderpath, subfolder)

        return folderpath

    def getResultFilepath(
        self,
        quantity_name: str,
        index: Union[tuple, Tuple[int, ...]] = None
    ):
        """
        Get filepath to save/load result from.

        :param self: :class:`~Results.Results` to retrieve filepath from
        :param index: :meth:`~Results.Results.getResultsOverTime` (not implemented)
        :param quantity_name: name of quantity to retrieve filepath for
        :param subfolder: folder relative to data folder to retrieve results file
        """
        folderpath = self.getFolderpath(index=index)
        file_extension = ".hdf5"
        filename = quantity_name + file_extension
        filepath = join(folderpath, filename)

        return filepath

    def getResultsFile(
        self,
        name: str,
        index: Union[tuple, Tuple[int]] = None
    ):
        """
        Get results file for single quantity.

        :param self: :class:`~Results.Results` to retrieve file from
        :param name: name of quantity to retrieve results for
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        :param subfolder: folder relative to data folder to retrieve results file
        """
        try:
            results_file = self.name2file[name]
        except KeyError:
            results_filepath = self.getResultFilepath(
                name,
                index=index
            )
            results_file_exists = exists(results_filepath)
            results_file = h5py.File(results_filepath, 'a')

            if not results_file_exists:
                result_stepcount = self.getStepcount()

                parameters_values = self.getFreeParameterValues()
                parameter_shape = tuple([
                    len(parameter_values)
                    for parameter_values in parameters_values
                ])

                results_shape = (*parameter_shape, result_stepcount)
                chunk_shape = (*np.ones(len(parameter_shape)), result_stepcount)

                results_file.create_dataset(
                    name,
                    results_shape,
                    chunks=chunk_shape
                )

            self.name2file[name] = results_file

        return results_file

    def closeResultsFiles(
        self,
        names: Union[str, List[str]] = None,
        index: Union[tuple, Tuple[int]] = None
    ):
        """
        Close results file for single quantity.

        :param self: :class:`~Results.Results` to retrieve file from
        :param names: name of quantity to close file for.
            Defaults to closing all open files.
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        """
        name2file = self.name2file

        if names is None:
            for file in name2file.values():
                file.close()
            self.name2file = {}
        else:
            if isinstance(names, str):
                names = [names]

            name2file_keys = list(name2file.keys())
            for name in names:
                if name in name2file_keys:
                    file = name2file[name]
                    file.close()
                    del self.name2file[name]

    def flushResultsFiles(
        self,
        names: str = None,
        index: Union[tuple, Tuple[int]] = None
    ):
        """
        Close results file for single quantity.

        :param self: :class:`~Results.Results` to retrieve file from
        :param names: name of quantity to close file for.
            Defaults to flushing all open files.
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        """
        def flush(name):
            """Base method for :class:`~Results.Results.closeResultsFile`"""
            try:
                results_file = self.name2file[name]
                results_file.flush()
            except KeyError:
                pass

        recursiveMethod(
            args=names,
            base_method=flush,
            valid_input_types=str,
            default_args=self.name2file.keys()
        )

    def deleteResultsFiles(
        self,
        names: str = None,
        index: Union[tuple, Tuple[int]] = None
    ):
        """
        Close results file for single quantity.

        :param self: :class:`~Results.Results` to retrieve file from
        :param names: name of quantity to close file for.
            Defaults to flushing all open files.
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        """

        def delete(name):
            """Base method for :class:`~Results.Results`"""
            self.closeResultsFiles(names=name)
            results_filepath = self.getResultFilepath(name, index=index)
            try:
                os.remove(results_filepath)
            except FileNotFoundError:
                pass

        variable_names = self.getVariableNames()
        function_names = self.getFunctionNames()
        quantity_names = [
            't',
            *variable_names,
            *function_names
        ]

        recursiveMethod(
            args=names,
            base_method=delete,
            valid_input_types=str,
            default_args=quantity_names
        )

    def saveResult(
        self,
        index: Union[tuple, Tuple[int]],
        name: str,
        result: ndarray,
        close_files: bool = True
    ) -> None:
        """
        Save single result into file.

        :param self: :class:`~Results.Results` to retrieve file from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param name: name of quantity to set results for
        :param result: single 1D array of results for quantity
        :param close_files: set True to immediately close any opened files;
            Set False to leave files open, and close manually later.
        """
        assert isinstance(index, tuple)
        for simulation_index in index:
            assert isinstance(simulation_index, (int, np.integer))

        stepcount = self.getStepcount()
        assert result.size == stepcount

        results_file = self.getResultsFile(name, index=index)

        simulation_type = self.getSimulationType()
        if simulation_type == "grid":
            array_index = index
        elif simulation_type == "optimization":
            array_index = index[1]

        results_file[name][array_index] = result
        if close_files:
            self.closeResultsFiles(names=name)
