from __future__ import annotations

import itertools
import os
import sys
import time
import traceback
from functools import partial
from math import prod
from os import mkdir
from os.path import isdir, isfile, join
from typing import Callable, Dict, Iterable, List, Tuple, Union

import h5py
import numpy as np
import PySimpleGUI as sg
from numpy import ndarray
from pint import Quantity
from sympy import Expr, Symbol
from sympy.utilities.lambdify import lambdify

from Function import Model, Parameter, Variable
from Layout.AxisQuantity import AxisQuantity, AxisQuantityMetadata
from macros import StoredObject, recursiveMethod
from Transforms.CustomMath import normalizeOverAxes
from Config import loadConfig, saveConfig


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

    def getFunction(self) -> Callable[[ndarray], ndarray]:
        """
        Get function, pre-substituting in times if required.

        :param self: :class:`~Results.FunctionOnResult` to retrieve function from
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


def transformOfResults(
    results: ndarray,
    transform_name: str,
    is_time: bool = False,
    times_callable: Callable[[], ndarray] = None
) -> ndarray:
    """
    Get math transform of results.

    :param self: :class:`~Results.Results` to retrieve transform from
    :param results: 1D ndarray if transform function requires exactly one nontime argument;
        tuple of two 1D-ndarrays each with same size, if transform requires two or more nontime arguments.
    :param transform_name: see :class:`~Results.Results.getResultsOverTime`
    :param is_time: set True if result is times for simulation. Set False otherwise.
        Only called if transform function requires exactly one nontime argument.
    :param times_callable: callable returning 1D ndarray of times with inequality filters applied.
        Only called if corresponding :class:`~Results.Transform` requires time as input.
    """
    assert isinstance(results, ndarray)
    transform_results = results
    transform_obj: Transform = Transform.getInstances(names=transform_name)

    dimension_count = results.ndim
    if dimension_count == 1:
        quantity_count = 1
    elif dimension_count == 2:
        quantity_count = results.shape[0]
        if quantity_count == 1:
            results = results[0]
            dimension_count = 1
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
            times = times_callable()
            transform_function = partial(transform_function, times=times)

    transform_results = transform_function(transform_results)
    return transform_results


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


def envelopeOfResults(
    results: ndarray,
    envelope_name: str,
    times_callable: Callable[[], ndarray] = None
) -> ndarray:
    """
    Get envelope of results.

    :param self: :class:`~Results.Results` to retrieve envelope from
    :param results: 1D ndarray of results
    :param envelope_name: see :class:`~Results.Results.getResultsOverTime`
    :param times_callable: callable returning 1D ndarray of times with inequality filters applied.
        Only called if corresponding :class:`~Results.Results.Envelope` requires time as input.
    """
    envelope_obj: Envelope = Envelope.getInstances(names=envelope_name)
    envelope_function = envelope_obj.getFunction()

    envelope_requires_times = envelope_obj.requiresTimes()
    if envelope_requires_times:
        times = times_callable()
        envelope_function = partial(envelope_function, times=times)

    envelope_results = envelope_function(results)
    return envelope_results


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


def functionalOfResults(
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


def complexReductionOfResults(
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


def preResultsOverTime(
    results: ndarray,
    axis_quantity_metadata: AxisQuantityMetadata,
    times_callable: Callable[[], ndarray] = None
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
    :param axis_quantity_metadata: see :class:`~Results.Results.getResultsOverTime`
    :param times_callable: callable returning 1D ndarray of times with inequality filters applied
    """
    if axis_quantity_metadata is not None:
        assert isinstance(axis_quantity_metadata, AxisQuantityMetadata)
        envelope_name = axis_quantity_metadata.getEnvelopeName()
    else:
        envelope_name = "None"
    post_results = results

    if envelope_name != "None":
        post_results = envelopeOfResults(
            post_results,
            envelope_name=envelope_name,
            times_callable=times_callable
        )

    return post_results


def postResultsOverTime(
    results: ndarray,
    axis_quantity_metadata: AxisQuantityMetadata,
    times_callable: Callable[[], ndarray] = None
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
    :param axis_quantity_metadata: see :class:`~Results.Results.getResultsOverTime`
    :param times_callable: callable returning 1D ndarray of times with inequality filters applied
    """
    if axis_quantity_metadata is not None:
        assert isinstance(axis_quantity_metadata, AxisQuantityMetadata)
        transform_name = axis_quantity_metadata.getTransformName()
        functional_name = axis_quantity_metadata.getFunctionalName()
        complex_name = axis_quantity_metadata.getComplexName()

        quantity_names = axis_quantity_metadata.getAxisQuantityNames()
        if len(quantity_names) >= 2:
            quantity_is_time = False
        elif len(quantity_names) == 1:
            quantity_name = quantity_names[0]
            quantity_is_time = quantity_name == 't'
    else:
        transform_name = "None"
        functional_name = "None"
        complex_name = "Real"
        quantity_is_time = False

    post_results = results

    if transform_name != "None":
        post_results = transformOfResults(
            post_results,
            transform_name=transform_name,
            is_time=quantity_is_time,
            times_callable=times_callable
        )

    if functional_name != "None":
        post_results = functionalOfResults(
            post_results,
            functional_name=functional_name
        )

    if complex_name != "None":
        post_results = complexReductionOfResults(
            post_results,
            complex_name=complex_name
        )

    return post_results


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
        results_file_handler: ResultsFileHandler,
        transform_config_filepath: str = "transforms/transforms.json",
        envelope_config_filepath: str = "transforms/envelopes.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json"
    ) -> None:
        """
        Constructor for :class:`~Results.Results`


        :param fit_parameter_names: see :class:`~Results.OptimizationResultsFileHandler.__init__.fit_parameter_names`.
            Only required for :class:`~Results.OptimizationResultsFileHandler`.
        :param sample_sizes: see :class:`~Results.OptimizationResultsFileHandler.sample_sizes`.
            Only required for :class:`~Results.OptimizationResultsFileHandler`.
        """
        assert isinstance(model, Model)
        self.model = model

        assert isinstance(results_file_handler, ResultsFileHandler)
        self.results_file_handler = results_file_handler

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

        self.general_function_expressions = {}
        self.general_equilibrium_expressions = {}

    def getResultsFileHandler(self) -> ResultsFileHandler:
        """
        Get object to handle saving/loading files.

        :param self: :class:`~Results.Results` to retrieve object from
        """
        return self.results_file_handler

    @staticmethod
    def updateProgressMeter(
        current_value: int,
        max_value: int,
        title: str
    ) -> sg.OneLineProgressMeter:
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

    def getFreeParameterIndicies(
        self,
        names: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        Get index of free parameter within collection of free-parameter names.

        :param self: :class:`~Results.Results` to retrieve free-parameter names from
        :param names: name(s) of free parameter(s) to retrieve index(es) of
        """
        results_file_handler = self.getResultsFileHandler()
        free_parameter_names = results_file_handler.getFreeParameterNames()

        def get(name: str) -> int:
            """Base method for :meth:`~Results.Results.getFreeParameterIndicies`"""
            free_parameter_index = free_parameter_names.index(name)
            return free_parameter_index

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            default_args=free_parameter_names
        )

    def getParameterSubstitutionsCore(
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

        results_file_handler: Union[GridResultsFileHandler, OptimizationResultsFileHandler] = self.getResultsFileHandler()
        substitutions = {}
        if include_free:
            free_substitutions = results_file_handler.getParameterSubstitutions(
                names=parameter_names,
                index=index,
            )
            substitutions.update(free_substitutions)
        if include_nonfree:
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

        parameter_substitutions = self.getParameterSubstitutionsCore(
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
            parameter_substitutions = self.getParameterSubstitutionsCore(
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
        parameter_substitutions = self.getParameterSubstitutionsCore(
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
        parameter_substitutions = self.getParameterSubstitutionsCore(
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

    def getFilteredIndicies(
        self,
        index: Union[tuple, Tuple[int, ...]],
        inequality_filters: Iterable[Tuple[str, str, float]] = None,
    ) -> ndarray:
        if inequality_filters is None:
            filter_intersection = ()
        else:
            assert isinstance(inequality_filters, Iterable)

            if len(inequality_filters) >= 1:
                for inequality_index, inequality_filter in enumerate(inequality_filters):
                    assert isinstance(inequality_filter, tuple)
                    assert len(inequality_filter) == 3
                    filter_quantity_name, filter_inequality_type, filter_float = inequality_filter
                    assert isinstance(filter_quantity_name, str)
                    assert isinstance(filter_inequality_type, str)
                    filter_float = float(filter_float)

                    filtered_results = self.getResultsOverTime(
                        index=index,
                        quantity_names=filter_quantity_name,
                        inequality_filters=None
                    )

                    if isinstance(filtered_results, ndarray):
                        if inequality_index == 0:
                            stepcount = filtered_results.shape[-1]
                            filter_intersection = np.arange(stepcount)

                        new_filter_indicies = eval(f"np.where(filtered_results{filter_inequality_type:s}{filter_float:})")
                        filter_intersection = np.intersect1d(
                            filter_intersection,
                            new_filter_indicies
                        )
                    else:
                        return ()
            else:
                filter_intersection = ()

        return filter_intersection

    def generateResultsOverTime(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        close_file: bool = True
    ) -> ndarray:
        """
        Generate and save non-calculated results from simulation.

        :param index: see :meth:`~Results.Results.loadResultsOverTime.index`
        :param quantity_name: see :meth:`~Results.Results.loadResultsOverTime.quantity_name`
        :param close_file: set True to immediately close any opened files
        """
        model = self.getModel()
        results_file_handler = self.getResultsFileHandler()

        variable_names = results_file_handler.getVariableNames()
        function_names = results_file_handler.getFunctionNames()
        quantity_is_time = quantity_name == 't'
        if quantity_name in variable_names:
            variable_obj: Variable = model.getVariables(names=quantity_name)
            time_evolution_type = variable_obj.getTimeEvolutionType()

            if time_evolution_type == "Temporal":
                print(f"returning NaN for {quantity_name:s}")
                return np.nan

            results_handles = {
                "Equilibrium": self.getEquilibriumVariableResults,
                "Constant": self.getConstantVariableResults,
                "Function": self.getFunctionResults,
            }
            single_results = results_handles[time_evolution_type](
                index,
                quantity_name
            )
        elif quantity_name in function_names:
            single_results = self.getFunctionResults(
                index,
                quantity_name
            )
        elif quantity_is_time:
            print("returning NaN for time")
            return np.nan
        else:
            raise ValueError("quantity_name input must correspond to either variable or function when str")

        results_file_handler.saveResult(
            index=index,
            quantity_name=quantity_name,
            result=single_results,
            close_file=close_file
        )

        return single_results

    def loadResultsOverTime(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        close_file: bool = True
    ) -> ndarray:
        """
        Load results from file or generate if not present in file.

        :param self: :class:`~Results.Results` to load results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param quantity_name: name of variable/function to retrieve results for
        :param close_file: set True to immediately close any opened files
        """
        try:
            results_file_handler = self.getResultsFileHandler()
            single_results = results_file_handler.loadResult(
                index=index,
                quantity_name=quantity_name,
                close_file=close_file
            )

            is_all_zero = not np.any(single_results)
            if is_all_zero:
                raise ValueError(f"numpy array {quantity_name:s} {index} contains only zeros... calculating values")
        except (FileNotFoundError, ValueError, IndexError) as error:
            print(error)
            single_results = self.generateResultsOverTime(
                index=index,
                quantity_name=quantity_name,
                close_file=close_file
            )

        return single_results

    def getResultsOverTime(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_names: Union[str, List[str]],
        axis_quantity_metadata: AxisQuantityMetadata = None,
        inequality_filters: Iterable[Tuple[str, str, float]] = None,
        functional_kwargs: dict = None,
        perform_post_multiple: bool = True,
        perform_post_functional: bool = True,
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
        :param index: see :meth:`~Results.Results.loadResultsOverTime.index`
        :param quantity_names: see :meth:`~Results.Results.loadResultsOverTime.quantity_name`.
            May alternatively be a collection of these names.
        :param axis_quantity_metadata: object containing attributes indicating how to retrieve data...
            envelope_name: name of envelope to perform on results.
            transform_name: name of transform to perform on results
            functional_name: functional to perform on results to reduce into one (or few) floats
            complex_name: name of function to reduce complex numbers into real numbers
        :param functional_kwargs: dictionary of optional arguments to pass into corresponding condensing function
        :param inequality_filters: iterable of tuples indicating filters for results.
            First element of tuple is variable/function name.
            Second element of tuple is inequality sign as string.
            Third element of tuple is float.
            Example: ('t', '>', 1.0) includes only data where time (t) is greater than 1.
        :param perform_post_multiple: set True to perform functions after possibly retrieving results for multiple quantities.
            Set False otherwise.
        :param close_files: set True to immediately close any opened files.
        """
        if axis_quantity_metadata is not None:
            assert isinstance(axis_quantity_metadata, AxisQuantityMetadata)
            envelope_name = axis_quantity_metadata.getEnvelopeName()

            if perform_post_multiple:
                transform_name = axis_quantity_metadata.getTransformName()
                functional_name = axis_quantity_metadata.getFunctionalName()
                complex_name = axis_quantity_metadata.getComplexName()
            else:
                transform_name = "None"
                functional_name = "None"
                complex_name = "Real"
            if perform_post_functional:
                parameter_functional_names = axis_quantity_metadata.getFunctionalFunctionalNames(include_none=False)
                for parameter_functional_name in parameter_functional_names:
                    assert isinstance(parameter_functional_name, str)

                functional_parameter_namess = axis_quantity_metadata.getFunctionalParameterNamess(include_none=False)
                for functional_parameter_names in functional_parameter_namess:
                    assert isinstance(functional_parameter_names, list)
                    for parameter_name in functional_parameter_names:
                        assert isinstance(parameter_name, str)
            else:
                parameter_functional_names = []
                functional_parameter_namess = []
        else:
            envelope_name = "None"
            transform_name = "None"
            functional_name = "None"
            complex_name = "Real"
            parameter_functional_names = []
            functional_parameter_namess = []

        functional_parameter_namess_count = len(functional_parameter_namess)
        parameter_functional_names_count = len(parameter_functional_names)
        if functional_parameter_namess_count >= 1 or parameter_functional_names_count >= 1:
            assert functional_parameter_namess_count == parameter_functional_names_count

            parameter_names_flat = [
                parameter_name
                for parameter_names in functional_parameter_namess
                for parameter_name in parameter_names
            ]

            self: GridResults = self
            parameter_results, results_per_parameter = self.getResultsOverTimePerParameter(
                index=index,
                parameter_names=parameter_names_flat,
                quantity_names=quantity_names,
                axis_quantity_metadata=axis_quantity_metadata,
                inequality_filters=inequality_filters,
                functional_kwargs=functional_kwargs,
                perform_post_functional=False
            )
            results: ndarray = results_per_parameter[0]

            previous_parameter_counts = 0
            for parameter_functional_index, parameter_functional_name in enumerate(parameter_functional_names):
                functional_parameter_names = functional_parameter_namess[parameter_functional_index]
                parameter_count = len(functional_parameter_names)

                parameter_values = parameter_results[previous_parameter_counts:previous_parameter_counts + parameter_count]

                if len(parameter_values) == 1:
                    parameter_values = parameter_values[0]

                results_shape = results.shape
                parameters_shape = results_shape[:parameter_count]
                residual_shape = results_shape[parameter_count:]
                parameter_value_count = np.prod(parameters_shape)

                getFunctionalOfResults = partial(
                    functionalOfResults,
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
            filtered_indicies = self.getFilteredIndicies(
                index=index,
                inequality_filters=inequality_filters
            )

            def filteredTimesCallable() -> ndarray:
                times = self.getResultsOverTime(
                    index=index,
                    quantity_names='t'
                )
                times = times[filtered_indicies]
                return times

            if isinstance(quantity_names, str):
                single_results = self.loadResultsOverTime(
                    index=index,
                    quantity_name=quantity_names,
                    close_file=close_files
                )
                if not isinstance(single_results, ndarray):
                    return single_results
                single_results = single_results[filtered_indicies]

                if envelope_name != "None":
                    single_results = preResultsOverTime(
                        single_results,
                        axis_quantity_metadata=axis_quantity_metadata,
                        times_callable=filteredTimesCallable
                    )

                results = single_results
            elif isinstance(quantity_names, list):
                getResultsOverTime = partial(
                    self.getResultsOverTime,
                    index=index,
                    inequality_filters=inequality_filters,
                    axis_quantity_metadata=axis_quantity_metadata,
                    perform_post_multiple=False,
                    perform_post_functional=False
                )
                multiple_results = np.array([
                    getResultsOverTime(quantity_names=name)
                    for name in quantity_names
                ])

                results = multiple_results
            else:
                raise TypeError("names input must be str or list")

            if perform_post_multiple:
                if transform_name != "None" or functional_name != "None" or complex_name != "None":
                    results = postResultsOverTime(
                        results,
                        axis_quantity_metadata=axis_quantity_metadata,
                        times_callable=filteredTimesCallable
                    )

        return results


class GridResults(Results):
    def __init__(
        self,
        model: Model,
        folderpath: str,
        free_parameter_name2values: Dict[str, ndarray],
        stepcount: int = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Results.GridResults`

        :param model: :class:`~Function.Model` to calculate results from
        :param folderpath: folder path containing relevant Results files.
            Save and load here.
        :param stepcount: number of times per individual simulation.
            Uses files in :paramref:`Results.GridResults.folderpath` when None.
        :param kwargs: additional arguments to pass into :class:`~Results.Results`
        """
        assert isinstance(model, Model)
        variable_names = model.getVariables(return_type=str)
        function_names = model.getFunctionNames()

        results_file_handler = GridResultsFileHandler(
            folderpath=folderpath,
            variable_names=variable_names,
            function_names=function_names,
            stepcount=stepcount,
            free_parameter_name2values=free_parameter_name2values
        )

        Results.__init__(
            self,
            model=model,
            results_file_handler=results_file_handler,
            **kwargs
        )

    def getResultsOverTimePerParameter(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_names: Union[str, List[str]],
        quantity_names: Union[str, List[str], List[List[str]]],
        axis_quantity_metadata: AxisQuantityMetadata,
        normalize_names: Union[str, List[str]] = None,
        show_progress: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, ndarray], ndarray]:
        """
        Get free-parameter values and "averaged" quantity values.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param parameter_names: name(s) of free parameter(s) to retrieve quantity results over.
        :param quantity_names: name(s) of quantity(s) to pass into :meth:`~Results.Results.getResultsOverTime`
        :param axis_quantity_metadata: see :class:`~Results.Results.getResultsOverTime`
        :param normalize_names: name(s) of parameter(s) to normalize results over
        :param show_progress: show progress bar during calculation
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        :returns: tuple of results.
            First index gives dictionary of parameter base values.
                Key is name of parameter.
                Value is array of parameter values.
            Second index gives matrix quantity results.
                nth index of this matrix gives values for nth quantity in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.quantity_names`.
        """
        if isinstance(quantity_names, str):
            quantity_names = [quantity_names]
        else:
            assert isinstance(quantity_names, list)
            has_list_in_list = False
            for quantity_name in quantity_names:
                if isinstance(quantity_name, list):
                    has_list_in_list = True
                    for sub_quantity_name in quantity_name:
                        assert isinstance(sub_quantity_name, quantity_name)
                else:
                    assert isinstance(quantity_name, str)
            if not has_list_in_list:
                quantity_names = [quantity_names]

        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        else:
            assert isinstance(parameter_names, list)
            for parameter_name in parameter_names:
                assert isinstance(parameter_name, str)

        if normalize_names is None:
            assert isinstance(axis_quantity_metadata, AxisQuantityMetadata)
            normalize_names = axis_quantity_metadata.getNormalizeNames()
        elif isinstance(normalize_names, str):
            normalize_names = [normalize_names]
        assert isinstance(normalize_names, list)
        for normalize_name in normalize_names:
            assert isinstance(normalize_name, str)

        default_index = np.array(index)
        sample_result = self.getResultsOverTime(
            index=index,
            quantity_names=quantity_names[0],
            axis_quantity_metadata=axis_quantity_metadata,
            **kwargs
        )

        results_file_handler: GridResultsFileHandler = self.getResultsFileHandler()
        per_parameter_base_values = tuple(results_file_handler.getFreeParameterName2Values(names=parameter_names))
        per_parameter_stepcounts = tuple(map(len, per_parameter_base_values))

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
        simulation_count_per_quantity = np.prod(np.array(per_parameter_stepcounts))
        simulation_count = quantity_count * simulation_count_per_quantity
        results = np.zeros((quantity_count, *single_result_size))

        if show_progress:
            updateProgressMeter = partial(
                self.updateProgressMeter,
                title="Calculating Simulation",
                max_value=simulation_count
            )

        time_resolution_seconds = 1
        previous_time = time.time()

        per_parameter_locations = np.array(self.getFreeParameterIndicies(names=parameter_names))
        per_parameter_partial_indicies = tuple(list(itertools.product(*map(range, per_parameter_stepcounts))))
        for quantity_location, quantity_name in enumerate(quantity_names):
            single_results = np.zeros(single_result_size)
            for partial_index_flat, partial_index in enumerate(per_parameter_partial_indicies):
                current_time = time.time()
                if show_progress and current_time - previous_time >= time_resolution_seconds:
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
                        axis_quantity_metadata=axis_quantity_metadata,
                        close_files=False,
                        **kwargs
                    )
                    single_results[partial_index] = single_result
                except (KeyError, AssertionError):
                    print('Per-parameter results error:', partial_index, traceback.print_exc())
                    single_results[partial_index] = None

            single_results = normalizeOverAxes(single_results, normalize_over_axes)
            results[quantity_location] = single_results

        if show_progress:
            updateProgressMeter(simulation_count)

        results_file_handler.closeResultsFiles()

        parameter_name2basevalues = dict(zip(parameter_names, per_parameter_base_values))
        return parameter_name2basevalues, results

    def saveResultsMetadata(self) -> None:
        """
        Save metadata into folder, to recreate results object.

        :param self: :class:`~Results.GridResults` to save metadata from
        """
        results_file_handler: GridResultsFileHandler = self.getResultsFileHandler()
        folderpath = results_file_handler.getFolderpath()
        model = self.getModel()

        parameter_values = results_file_handler.getFreeParameterName2Values(output_type=dict)
        free_parameter_info = {}
        for parameter_name, parameter_values in parameter_values.items():
            parameter = model.getParameters(names=parameter_name)
            parameter_quantity = parameter.getQuantity()
            unit = str(parameter_quantity.to_base_units().units)
            free_parameter_info[parameter_name] = {
                "index": self.getFreeParameterIndicies(parameter_name),
                "values": list(map(str, parameter_values)),
                "unit": unit
            }
        free_parameter_filepath = join(folderpath, "FreeParameter.json")
        free_parameter_file = saveConfig(free_parameter_info, free_parameter_filepath)

        function_filepath = join(folderpath, "Function.json")
        parameter_filepath = join(folderpath, "Parameter.json")
        variable_filepath = join(folderpath, "Variable.json")

        function_file = model.saveFunctionsToFile(function_filepath)
        parameter_file = model.saveParametersToFile(parameter_filepath)
        variable_file = model.saveVariablesToFile(variable_filepath)


class OptimizationResults(Results):
    def __init__(
        self,
        model: Model,
        folderpath: str,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        fitdata_filepath: str,
        fit_axis_quantity_metadata: AxisQuantityMetadata,
        sample_sizes: Tuple[int, ...] = None,
        stepcount: int = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Results.OptimizationResults`

        :param model: :class:`~Function.Model` to calculate results from
        :param folderpath: folder path containing relevant Results files.
            Save and load here.
        :param stepcount: number of times per individual simulation.
            Uses files in :paramref:`Results.GridResults.folderpath` when None.
        :param free_parameter_name2quantity: dictionary for free parameter quantities.
            Key is name of parameter.
            Value is quantity object with default value of parameter.
        :param sample_sizes: see :class:`~Results.OptimizationResultsFileHandler.sample_sizes`
        :param kwargs: additional arguments to pass into :class:`~Results.Results`
        """
        assert isinstance(model, Model)
        variable_names = model.getVariables(return_type=str)
        function_names = model.getFunctionNames()

        assert isinstance(fitdata_filepath, str)
        self.fitdata_filepath = fitdata_filepath

        assert isinstance(fit_axis_quantity_metadata, AxisQuantityMetadata)
        self.fit_axis_quantity_metadata = fit_axis_quantity_metadata

        if sample_sizes is None:
            fit_data: ndarray = np.load(fitdata_filepath)
            fit_data_count = fit_data.shape[-1]
            sample_sizes = list(np.logspace(
                0,
                np.log10(fit_data_count),
                3,
                endpoint=True,
                dtype=np.int32
            ))
            sample_sizes = tuple(list(map(int, sample_sizes)))
        else:
            assert isinstance(sample_sizes, tuple)

        results_file_handler = OptimizationResultsFileHandler(
            folderpath=folderpath,
            variable_names=variable_names,
            function_names=function_names,
            stepcount=stepcount,
            free_parameter_name2quantity=free_parameter_name2quantity,
            fit_parameter_names=fit_parameter_names,
            sample_sizes=sample_sizes
        )

        Results.__init__(
            self,
            model=model,
            results_file_handler=results_file_handler,
            **kwargs
        )

    def getFitdataFilepath(self) -> str:
        """
        Get filepath for file containing fitdata.

        :param self: :class:`~Results.OptimizationResults` to retrive filepath from
        """
        return self.fitdata_filepath

    def getFitAxisQuantityMetadata(self) -> AxisQuantityMetadata:
        """
        Get axis-quantity to fit simulation to data.

        :param self: :class:`~Results.OptimizationResults` to retrieve axis-quantity from
        """
        return self.fit_axis_quantity_metadata

    def getResultsOverTimePerParameter(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_names: Union[str, List[str]],
        quantity_names: Union[str, List[str], List[List[str]]],
        axis_quantity_metadata: AxisQuantityMetadata,
        normalize_names: Union[str, List[str]] = None,
        show_progress: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, ndarray], ndarray]:
        """
        Get free-parameter values and "averaged" quantity values.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param parameter_names: name(s) of free parameter(s) to retrieve quantity results over.
        :param quantity_names: name(s) of quantity(s) to pass into :meth:`~Results.Results.getResultsOverTime`
        :param axis_quantity_metadata: see :class:`~Results.Results.getResultsOverTime`
        :param normalize_names: name(s) of parameter(s) to normalize results over
        :param show_progress: show progress bar during calculation
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        :returns: tuple of results.
            First index gives dictionary of parameter base values.
                Key is name of parameter.
                Value is array of parameter values.
            Second element gives matrix quantity results.
                nth index of this matrix gives values for nth quantity in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.quantity_names`.
        """
        if isinstance(quantity_names, str):
            quantity_names = [quantity_names]
        else:
            assert isinstance(quantity_names, list)
            has_list_in_list = False
            for quantity_name in quantity_names:
                if isinstance(quantity_name, list):
                    has_list_in_list = True
                    for sub_quantity_name in quantity_name:
                        assert isinstance(sub_quantity_name, quantity_name)
                else:
                    assert isinstance(quantity_name, str)
            if not has_list_in_list:
                quantity_names = [quantity_names]

        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        else:
            assert isinstance(parameter_names, list)
            for parameter_name in parameter_names:
                assert isinstance(parameter_name, str)

        if normalize_names is None:
            assert isinstance(axis_quantity_metadata, AxisQuantityMetadata)
            normalize_names = axis_quantity_metadata.getNormalizeNames()
        elif isinstance(normalize_names, str):
            normalize_names = [normalize_names]
        assert isinstance(normalize_names, list)
        for normalize_name in normalize_names:
            assert isinstance(normalize_name, str)

        sample_result = self.getResultsOverTime(
            index=index,
            quantity_names=quantity_names[0],
            axis_quantity_metadata=axis_quantity_metadata,
            **kwargs
        )

        results_file_handler: OptimizationResultsFileHandler = self.getResultsFileHandler()

        all_fit_parameter_names = results_file_handler.getFitParameterNames()
        all_free_parameter_names = results_file_handler.getFreeParameterNames()
        fit_parameter_names = []
        free_parameter_names = []
        for parameter_name in parameter_names:
            assert parameter_name in all_fit_parameter_names

            if parameter_name in all_fit_parameter_names:
                fit_parameter_names.append(parameter_name)
            elif parameter_name in all_free_parameter_names:
                free_parameter_names.append(parameter_name)
            else:
                raise ValueError(f"parameter {parameter_name:s} must be in fit or free parameters")

        parameters_values: Tuple[ndarray] = results_file_handler.getFreeParameterName2Values(
            names=parameter_names,
            index=index,
            output_type=tuple
        )
        has_fit_parameter = len(fit_parameter_names) >= 1
        has_free_parameter = len(free_parameter_names) >= 1
        assert has_fit_parameter or has_free_parameter

        per_parameter_shape = parameters_values[0].shape
        for parameter_values in parameters_values[1:]:
            parameter_shape = parameter_values.shape
            assert parameter_shape == per_parameter_shape

        total_group_size, total_sample_size = per_parameter_shape
        default_group_index, default_sample_index = results_file_handler.generateArrayIndex(index)

        normalize_over_axes = [
            parameter_names.index(normalize_name)
            for normalize_name in normalize_names
        ]
        if has_fit_parameter and has_free_parameter:
            single_result_base_size = (total_group_size, total_sample_size)
            parameter_indicies = np.array(list(itertools.product(*map(range, per_parameter_shape))))
        elif has_free_parameter:
            single_result_base_size = (total_group_size, )
            parameter_indicies = np.array([
                [group_index, default_sample_index]
                for group_index in range(total_group_size)
            ])
        elif has_fit_parameter:
            single_result_base_size = (total_sample_size, )
            parameter_indicies = np.array([
                [default_group_index, sample_index]
                for sample_index in range(total_sample_size)
            ])

        group_indicies, sample_indicies = parameter_indicies.T
        parameters_results = tuple([
            parameter_values[group_indicies, sample_indicies]
            for parameter_values in parameters_values
        ])

        if isinstance(sample_result, (float, int, np.float32)):
            single_result_size = single_result_base_size
        elif isinstance(sample_result, ndarray):
            single_result_size = (*single_result_base_size, *sample_result.shape)
            if len(normalize_over_axes) >= 1:
                normalize_over_axes.append(-1)
        else:
            raise TypeError(f"invalid type ({type(sample_result):})")
        normalize_over_axes = tuple(normalize_over_axes)

        quantity_count = len(quantity_names)
        simulation_count_per_quantity = np.prod(np.array(single_result_base_size))
        simulation_count = quantity_count * simulation_count_per_quantity
        results = np.zeros((quantity_count, *single_result_size))

        if show_progress:
            updateProgressMeter = partial(
                self.updateProgressMeter,
                title="Calculating Simulation",
                max_value=simulation_count
            )

        time_resolution_seconds = 1
        previous_time = time.time()

        for quantity_location, quantity_name in enumerate(quantity_names):
            single_results = np.zeros(single_result_size)
            for array_index_flat, array_index in enumerate(parameter_indicies):
                current_time = time.time()
                if show_progress and current_time - previous_time >= time_resolution_seconds:
                    simulation_index_flat = quantity_location * simulation_count_per_quantity + array_index_flat + 1
                    if not updateProgressMeter(simulation_index_flat):
                        updateProgressMeter(simulation_count)
                        return
                    previous_time = current_time

                try:
                    parameter_index = (total_sample_size, *array_index)
                    single_result = self.getResultsOverTime(
                        index=parameter_index,
                        quantity_names=quantity_name,
                        axis_quantity_metadata=axis_quantity_metadata,
                        close_files=False,
                        **kwargs
                    )

                    if has_fit_parameter and has_free_parameter:
                        results_index: ndarray = array_index
                    elif has_free_parameter:
                        results_index: np.int32 = array_index[0]
                    elif has_fit_parameter:
                        results_index: np.int32 = array_index[1]
                    single_results[results_index] = single_result
                except (KeyError, AssertionError):
                    print('Per-parameter results error:', array_index, traceback.print_exc())
                    single_results[array_index] = None

            single_results = normalizeOverAxes(single_results, normalize_over_axes)
            results[quantity_location] = single_results

        if show_progress:
            updateProgressMeter(simulation_count)

        results_file_handler.closeResultsFiles()
        results_file_handler.closeParametersFiles()

        parameter_name2result = dict(zip(parameter_names, parameters_results))
        return parameter_name2result, results

    def saveResultsMetadata(self) -> None:
        """
        Save metadata into folder, to recreate results object.

        :param self: :class:`~Results.OptimizationResults` to save metadata from
        """
        results_file_handler: OptimizationResultsFileHandler = self.getResultsFileHandler()
        folderpath = results_file_handler.getFolderpath()
        model = self.getModel()

        function_filepath = join(folderpath, "Function.json")
        parameter_filepath = join(folderpath, "Parameter.json")
        variable_filepath = join(folderpath, "Variable.json")

        function_file = model.saveFunctionsToFile(function_filepath)
        parameter_file = model.saveParametersToFile(parameter_filepath)
        variable_file = model.saveVariablesToFile(variable_filepath)

        parameter_name2quantity = results_file_handler.getFreeParameterName2Quantity(output_type=dict)
        free_parameter_info = {}
        for parameter_name, parameter_quantity in parameter_name2quantity.items():
            parameter_quantity_si = parameter_quantity.to_base_units()

            boundstep = 5
            initial_guess = parameter_quantity_si.magnitude
            unit = str(parameter_quantity_si.units)
            bounds = [initial_guess / boundstep, initial_guess * boundstep]

            free_parameter_info[parameter_name] = {
                "index": self.getFreeParameterIndicies(parameter_name),
                "initial_guess": initial_guess,
                "bounds": bounds,
                "unit": unit
            }
        free_parameter_filepath = join(folderpath, "FreeParameter.json")
        free_parameter_file = saveConfig(free_parameter_info, free_parameter_filepath)

        fit_axis_quantity_metadata = self.getFitAxisQuantityMetadata()
        fit_axis_quantity_filepath = join(folderpath, "FitAxisQuantity.json")
        fit_parameter_names = results_file_handler.getFitParameterNames()
        if False:
            fit_axis_quantity_file = fit_axis_quantity_metadata.saveMetadataToFile(
                fit_axis_quantity_filepath,
                fit_parameter_names=fit_parameter_names
            )

    def saveFitData(self) -> None:
        """
        Save fit data to folder with simulated values.

        :param self: :class:`~Results.OptimizationResults` to retrieve fit-data from
        """
        results_file_handler: OptimizationResultsFileHandler = self.getResultsFileHandler()
        folderpath = results_file_handler.getFolderpath()
        save_filepath = join(folderpath, "FitData.npy")

        fit_data_filepath = self.getFitdataFilepath()
        fit_data = np.load(fit_data_filepath)

        np.save(save_filepath, fit_data)
        self.fitdata_filepath = save_filepath


class ResultsFileHandler:
    def __init__(
        self,
        folderpath: str,
        variable_names: List[str],
        function_names: List[str],
        free_parameter_names: List[str],
        stepcount: int,
        base_index: Tuple[int, ...]
    ) -> None:
        """
        Constructor for :class:`~Results.ResultsFileHandler`.

        :param folderpath: root folderpath to save/load results
        :param variable_names: collection of variable names to save/load results for
        :param function_names: collection of function names to save/load results for
        :param stepcount: number of values in each simulation per variable
        :param base_index: parameter index for first set of parameters
        """
        results_file_handler_classes = (
            GridResultsFileHandler,
            OptimizationResultsFileHandler
        )
        assert isinstance(self, results_file_handler_classes)

        assert isinstance(folderpath, str)
        self.root_folderpath = folderpath

        assert isinstance(variable_names, Iterable)
        for variable_name in variable_names:
            assert isinstance(variable_name, str)
        self.variable_names = variable_names

        assert isinstance(function_names, Iterable)
        for function_name in function_names:
            assert isinstance(function_name, str)
        self.function_names = function_names

        assert isinstance(free_parameter_names, Iterable)
        for free_parameter_name in free_parameter_names:
            assert isinstance(function_name, str)
        self.free_parameter_names = free_parameter_names

        assert isinstance(stepcount, int) or stepcount is None
        self.stepcount = stepcount

        assert isinstance(base_index, tuple)
        for base_subindex in base_index:
            if isinstance(base_subindex, np.integer):
                base_subindex = int(base_subindex)
            assert isinstance(base_subindex, int)
        self.base_index = base_index

        self.name2file: Dict[str, h5py.File] = {}

    def getStepcount(self) -> float:
        """
        Get number of steps per result.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve stepcount from
        """
        stepcount = self.stepcount

        if stepcount is None:
            first_index = self.getBaseIndex()
            time_results_filepath = self.getResultFilepath('t', index=first_index)
            assert isfile(time_results_filepath)

            with h5py.File(time_results_filepath, 'r') as time_results_file:
                time_results_dataset = time_results_file['t']
                time_results_dimension = time_results_dataset.ndim
                first_array_index = tuple(list(np.zeros(time_results_dimension - 1, dtype=int)))
                time_result = time_results_dataset[first_array_index]

            stepcount = time_result.size
            self.stepcount = stepcount

        return self.stepcount

    def getBaseIndex(self) -> int:
        """
        Get index for first set of simulations.
        """
        return self.base_index

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables simulated in results.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve names from
        """
        return self.variable_names

    def getFunctionNames(self) -> List[str]:
        """
        Get names of functions possibly simulated in results.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve names from
        """
        return self.function_names

    def getFreeParameterNames(self) -> List[str]:
        """
        Get names of free parameters.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve names from
        """
        return self.free_parameter_names

    def getFolderpath(
        self,
        index: Union[tuple, Tuple[int, ...]] = None,
    ) -> str:
        """
        Get folderpath to save/load results into/from.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve folderpath from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        root_folderpath = self.root_folderpath
        if root_folderpath is None:
            root_folderpath = sg.PopupGetFolder(
                message="Enter Folder to Load",
                title="Load Previous Results"
            )
            self.root_folderpath = root_folderpath

        self: Union[GridResultsFileHandler, OptimizationResultsFileHandler] = self
        folderpath = self.generateFolderpath(
            root_folderpath,
            index=index
        )

        if not isdir(folderpath):
            mkdir(folderpath)

        return folderpath

    def getResultFilepath(
        self,
        quantity_name: str,
        index: Union[tuple, Tuple[int, ...]] = None
    ):
        """
        Get filepath to save/load result from.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve filepath from
        :param quantity_name: name of quantity to retrieve filepath for
        :param index: :meth:`~Results.Results.getResultsOverTime`
        """
        folderpath = self.getFolderpath(index=index)
        file_extension = ".hdf5"
        filename = quantity_name + file_extension
        filepath = join(folderpath, filename)

        return filepath

    def getResultsFile(
        self,
        quantity_name: str,
        index: Union[tuple, Tuple[int]] = None
    ) -> h5py.File:
        """
        Get results file for single quantity.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
        :param quantity_name: name of quantity to retrieve results for
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        self: Union[GridResultsFileHandler, OptimizationResultsFileHandler] = self

        try:
            results_file = self.name2file[quantity_name]
        except KeyError:
            results_filepath = self.getResultFilepath(
                quantity_name,
                index=index
            )
            results_file_exists = isfile(results_filepath)
            results_file = h5py.File(results_filepath, 'a')

            if not results_file_exists:
                self.createResultsDataset(
                    quantity_name=quantity_name,
                    results_file=results_file,
                    index=index
                )
            else:
                results_dataset_names = list(results_file.keys())
                #print("ds_names:", quantity_name, results_dataset_names)
                if quantity_name not in results_dataset_names:
                    self.createResultsDataset(
                        quantity_name=quantity_name,
                        results_file=results_file
                    )

            self.name2file[quantity_name] = results_file

        return results_file

    def closeResultsFiles(
        self,
        names: Union[str, List[str]] = None,
        index: Union[tuple, Tuple[int]] = None
    ):
        """
        Close results file for single quantity.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
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

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
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

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
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
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        result: ndarray,
        close_file: bool = True
    ) -> None:
        """
        Save single result into file.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param quantity_name: name of quantity to set results for
        :param result: single 1D array of results for quantity
        :param close_file: set True to immediately close any opened files;
            Set False to leave files open, and close manually later.
        """
        self: Union[GridResultsFileHandler, OptimizationResultsFileHandler] = self

        assert isinstance(index, tuple)
        for sub_index in index:
            assert isinstance(sub_index, (int, np.integer))

        stepcount = self.getStepcount()
        assert result.size == stepcount

        results_file = self.getResultsFile(quantity_name, index=index)
        array_index = self.generateArrayIndex(index)

        if isinstance(self, OptimizationResultsFileHandler):
            results_dataset = results_file[quantity_name]
            results_dataset_shape = results_dataset.shape
            results_group_size, results_sample_size, results_time_size = results_dataset_shape
            
            group_index, sample_index = array_index

            new_dataset_group_size = max(results_group_size, group_index + 1)
            per_simulation_shape = (results_sample_size, results_time_size)
            new_dataset_shape = (new_dataset_group_size, *per_simulation_shape)
            results_dataset.resize(new_dataset_shape)

        results_file[quantity_name][array_index] = result

        if close_file:
            self.closeResultsFiles(names=quantity_name)


class GridResultsFileHandler(ResultsFileHandler):
    def __init__(
        self,
        free_parameter_name2values: Dict[str, ndarray],
        **kwargs
    ):
        """
        Constructor for :class:`~Results.GridResultsFileHandler`.

        :param free_parameter_name2values: see :class:`~Results.Results`
        :param kwargs: additional arguments to pass into :class:`~Results.ResultsFileHandler`
        """
        assert isinstance(free_parameter_name2values, dict)
        for parameter_name, parameter_values in free_parameter_name2values.items():
            assert isinstance(parameter_name, str)
            assert isinstance(parameter_values, ndarray)
        self.free_parameter_name2values = free_parameter_name2values

        free_parameter_names = list(free_parameter_name2values.keys())
        free_parameter_count = len(free_parameter_names)
        base_index = tuple(list(np.zeros(free_parameter_count, dtype=int)))

        ResultsFileHandler.__init__(
            self,
            free_parameter_names=free_parameter_names,
            base_index=base_index,
            **kwargs
        )

    @staticmethod
    def generateFolderpath(
        root_folderpath: str,
        index: Union[tuple, Tuple[int, ...]] = None
    ) -> str:
        """
        Generate folderpath for grid-results file handler.

        :param self: :class:`~Results.GridResultsFileHandler` to retrieve folderpath from
        :param root_folderpath: root folderpath for results
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        """
        folderpath = root_folderpath
        return folderpath

    @staticmethod
    def generateArrayIndex(index: Union[tuple, Tuple[int, ...]]) -> Union[tuple, Tuple[int, ...]]:
        """
        Generate array index from parameter index.

        :param self: :class:`~Results.GridResultsFileHandler` to retrieve index from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        return index

    def createResultsDataset(
        self,
        quantity_name: str,
        results_file: h5py.File,
        index: Tuple[int, ...] = None
    ):
        """
        Creates dataset with given name in given HDF5 file.

        :param self: :class:`~Results.GridResultsFileHandler` to create dataset for
        :param quantity_name: name of quantity to retrieve results for
        :param results_file: file to create dataset in
        :param index: not implemented
        """
        result_stepcount = self.getStepcount()

        parameters_values = self.getFreeParameterName2Values()
        parameter_shape = tuple([
            len(parameter_values)
            for parameter_values in parameters_values
        ])

        results_shape = (*parameter_shape, result_stepcount)
        chunk_shape = (*np.ones(len(parameter_shape)), result_stepcount)

        results_file.create_dataset(
            quantity_name,
            results_shape,
            chunks=chunk_shape
        )

    def loadResult(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        close_file: bool = True
    ) -> ndarray:
        """
        Load pre-calculated results from file.

        :param self: :class:`~Results.GridResultsFileHandler` to load results from
        :param index: see :meth:`~Results.Results.loadResultsOverTime.index`
        :param quantity_name: name of quantity to load results for
        :param close_file: set True to immediately close results file
        """
        results_file = self.getResultsFile(
            quantity_name=quantity_name,
            index=index
        )
        single_results = results_file[quantity_name][index]

        if close_file:
            self.closeResultsFiles(names=quantity_name)

        return single_results

    def getFreeParameterName2Values(
        self,
        names: Union[str, Iterable[str]] = None,
        output_type: type = list,
        index: tuple = None
    ) -> Union[ndarray, Dict[str, ndarray]]:
        """
        Get value(s) of free parameter(s).

        :param self: :class:`~Results.GridResultsFileHandler` to retrieve values from
        :param names: name(s) of parameter(s) to retrieve values for
        :param output_type: iterable to output as
            if :paramref:`~Results.ResultsFileHandler.getFreeParameterName2Values.names` is iterable
        :param index: not implemented
        """
        parameter_name2values = self.free_parameter_name2values

        def get(name: str) -> ndarray:
            """Base method for :meth:`~Results.GridResultsFileHandler.getFreeParameterName2Values`"""
            return parameter_name2values[name]

        free_parameter_names = self.getFreeParameterNames()
        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=output_type,
            default_args=free_parameter_names
        )

    def getParameterSubstitutions(
        self,
        index: Union[tuple, Tuple[int, ...]],
        names: Union[str, List[str]]
    ) -> Dict[Symbol, float]:
        """
        Get substitutions for free parameters at index.

        :param self: :class:`~Results.GridResultsFileHandler` to retrieve substitutions from
        :param names: name(s) of free parameter(s) to retrieve values for
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        if names is None:
            names = self.getFreeParameterNames()

        free_parameter_substitutions = {}
        for parameter_location, free_parameter_name in enumerate(names):
            parameter_index = index[parameter_location]
            parameter_values = self.getFreeParameterName2Values(names=free_parameter_name)
            parameter_value = parameter_values[parameter_index]
            parameter_symbol = Symbol(free_parameter_name)
            free_parameter_substitutions[parameter_symbol] = parameter_value

        return free_parameter_substitutions


class OptimizationResultsFileHandler(ResultsFileHandler):
    def __init__(
        self,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        sample_sizes: Tuple[int, ...],
        **kwargs
    ):
        """
        Constructor for :class:`~Results.OptimizationResultsFileHandler`.

        :param kwargs: additional arguments to pass into :class:`~Results.ResultsFileHandler`
        """
        assert isinstance(free_parameter_name2quantity, dict)
        for free_parameter_name, free_parameter_quantity in free_parameter_name2quantity.items():
            assert isinstance(free_parameter_name, str)
            assert isinstance(free_parameter_quantity, Quantity)
        self.free_parameter_name2quantity = free_parameter_name2quantity

        assert isinstance(fit_parameter_names, list)
        for fit_parameter_name in fit_parameter_names:
            assert isinstance(fit_parameter_name, str)
        self.fit_parameter_names = fit_parameter_names

        assert isinstance(sample_sizes, tuple)
        for sample_size in sample_sizes:
            assert isinstance(sample_size, int)
        self.sample_sizes = sample_sizes

        default_sample_size = sample_sizes[0]
        base_index = (default_sample_size, 0)

        free_parameter_names = list(free_parameter_name2quantity.keys())
        ResultsFileHandler.__init__(
            self,
            free_parameter_names=free_parameter_names,
            base_index=base_index,
            **kwargs
        )

        self.size2file: Dict[int, h5py.File] = {}

    @staticmethod
    def generateFolderpath(
        root_folderpath: str,
        index: Tuple[int, int, int] = None
    ) -> str:
        """
        Generate folderpath for grid-results file handler.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve folderpath from
        :param root_folderpath: root folderpath for results
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        if index is None:
            folderpath = root_folderpath
        else:
            assert isinstance(index, tuple)
            assert len(index) == 3
            for sub_index in index:
                assert isinstance(sub_index, (int, np.int32))

            subfolder = str(index[0])
            folderpath = join(root_folderpath, subfolder)

        return folderpath

    @staticmethod
    def generateArrayIndex(index: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Generate array index from parameter index.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve array-index from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :returns: 2-tuple of non-negative integers.
            First integer is group index; second integer is sample index.
        """
        return index[1:]

    @staticmethod
    def generateSampleSizeFromIndex(index: Tuple[int, int, int]) -> int:
        """
        Generate sample size from parameter index.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve sample size from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        return index[0]

    def getSampleSizes(
        self,
        index: int = None
    ) -> Tuple[int, ...]:
        """
        Get samples sizes of optimization fits.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve sizes from
        :param index: index to retrieve sample size at.
            Defaults to returning tuple of all sample sizes.
        """
        sample_sizes = self.sample_sizes
        if index is not None:
            sample_sizes = sample_sizes[index]

        return self.sample_sizes

    def createResultsDataset(
        self,
        quantity_name: str,
        results_file: h5py.File,
        index: Tuple[int, int, int]
    ):
        """
        Creates dataset with given name in given HDF5 file.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to create dataset for
        :param quantity_name: name of quantity to create dataset for
        :param results_file: file to create dataset in
        :param index: index for set of parameters
        """
        assert isinstance(quantity_name, str)
        assert isinstance(results_file, h5py.File)
        
        sample_size = self.generateSampleSizeFromIndex(index)
        
        result_stepcount = self.getStepcount()

        results_shape = (1, sample_size, result_stepcount)
        chunk_shape = (1, 1, result_stepcount)
        max_shape = (None, sample_size, result_stepcount)

        results_file.create_dataset(
            quantity_name,
            results_shape,
            chunks=chunk_shape,
            maxshape=max_shape
        )

    def createParameterDatasets(
        self,
        parameters_file: h5py.File,
        sample_size: int,
        parameter_names: Union[str, List[str]] = None
    ) -> None:
        """
        Creates dataset with given name in given HDF5 file.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to create dataset for
        :param parameters_file: file to create datasets in
        :param parameter_names: name(s) of parameter(s) to create dataset for.
            Default to all parameters.
        """
        assert isinstance(parameters_file, h5py.File)
        assert isinstance(sample_size, int)

        all_parameter_names = self.getParameterNames()
        if parameter_names is None:
            parameter_names = all_parameter_names
        elif isinstance(parameter_names, str):
            assert parameter_names in all_parameter_names
            parameter_names = [parameter_names]
        else:
            assert isinstance(parameter_names, list)
            for parameter_name in parameter_names:
                assert isinstance(parameter_name, str)
                assert parameter_name in all_parameter_names

        parameters_shape = (1, sample_size)
        chunk_shape = (1, 1)
        max_shape = (None, sample_size)

        createParameterDataset = partial(
            parameters_file.create_dataset,
            shape=parameters_shape,
            chunks=chunk_shape,
            maxshape=max_shape
        )
        dataset_names = list(parameters_file.keys())
        for parameter_name in parameter_names:
            if parameter_name not in dataset_names:
                createParameterDataset(parameter_name)

    def loadResult(
        self,
        index: Union[tuple, Tuple[int, ...]],
        quantity_name: str,
        close_file: bool = True
    ) -> ndarray:
        """
        Load pre-calculated results from file.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to load results from
        :param index: see :meth:`~Results.Results.loadResultsOverTime.index`
        :param quantity_name: name of quantity to load results for
        :param close_file: set True to immediately close results file
        """
        results_file = self.getResultsFile(
            quantity_name=quantity_name,
            index=index
        )
        array_index = self.generateArrayIndex(index)
        single_results = results_file[quantity_name][array_index]

        if close_file:
            self.closeResultsFiles(names=quantity_name)

        return single_results

    def getParameterFilepath(
        self,
        index: Tuple[int, int, int]
    ) -> str:
        """
        Get filepath to save/load result from.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve filepath from
        :param index: :meth:`~Results.Results.getResultsOverTime`
        """
        folderpath = self.getFolderpath(index=index)
        file_extension = ".hdf5"
        filename = "Parameter" + file_extension
        filepath = join(folderpath, filename)

        return filepath

    def getParametersFile(
        self,
        index: Tuple[int, int, int]
    ) -> h5py.File:
        """
        Get results file for single quantity.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve file from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        """
        parameters_filepath = self.getParameterFilepath(index=index)
        parameters_file_exists = isfile(parameters_filepath)
        parameters_file = h5py.File(parameters_filepath, 'a')

        if not parameters_file_exists:
            sample_size = self.generateSampleSizeFromIndex(index)
            self.createParameterDatasets(
                parameters_file=parameters_file,
                sample_size=sample_size
            )

        return parameters_file

    def closeParametersFiles(
        self,
        sample_sizes: Union[int, List[int]] = None
    ):
        """
        Close parameters file for single sample size.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve file from
        :param sample_sizes: sample size(s) to close parameter file for.
            Defaults to closing all open parameter files.
        """
        size2file = self.size2file

        if sample_sizes is None:
            for file in size2file.values():
                file.close()
            self.size2file = {}
        else:
            if isinstance(sample_sizes, int):
                sample_sizes = [sample_sizes]

            size2file_keys = list(size2file.keys())
            for sample_size in sample_sizes:
                if sample_size in size2file_keys:
                    file = size2file[sample_size]
                    file.close()
                    del self.size2file[sample_size]

    def getFreeParameterName2Values(
        self,
        index: Tuple[int, int, int],
        names: Union[str, Iterable[str]] = None,
        output_type: type = list,
        close_files: bool = True
    ) -> Union[ndarray, Dict[str, ndarray]]:
        """
        Get value(s) of free parameter(s).

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve values from
        :param index: index(es) of parameter(s) to retrieve values for
        :param names: name(s) of parameter(s) to retrieve values for
        :param output_type: iterable to output as
            if :paramref:`~Results.ResultsFileHandler.getFreeParameterName2Values.names` is iterable
        :param close_files: set True to immediately close any opened parameter files
        """
        parameters_file = self.getParametersFile(index=index)

        def get(name: str) -> ndarray:
            """Base method for :meth:`~Results.OptimizationResultsFileHandler.getFreeParameterName2Values`"""
            return parameters_file[name][:]

        free_parameter_names = self.getFreeParameterNames()
        parameters_values = recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=output_type,
            default_args=free_parameter_names
        )

        if close_files:
            sample_size = self.generateSampleSizeFromIndex(index=index)
            self.closeParametersFiles(sample_sizes=sample_size)

        return parameters_values

    def saveParametersSet(
        self,
        index: Tuple[int, int, int],
        parameter_values: ndarray
    ) -> None:
        """
        Save parameter values into file.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve file from
        :param index: see :meth:`~Results.Results.getResultsOverTime`
        :param parameter_values: single 1D array of parameters for ODE solution (with order matching fit followed by free parameters)
        """
        assert isinstance(index, tuple)
        assert len(index) == 3
        for sub_index in index:
            assert isinstance(sub_index, int)

        assert isinstance(parameter_values, ndarray)
        assert parameter_values.ndim == 1

        parameter_names = self.getParameterNames()
        parameter_count = len(parameter_names)
        assert parameter_values.size == parameter_count

        parameter_indicies = range(parameter_count)
        array_index = self.generateArrayIndex(index)
        group_index, sample_index = array_index
        with self.getParametersFile(index=index) as parameters_file:
            for parameter_index in parameter_indicies:
                parameter_name = parameter_names[parameter_index]
                parameter_dataset = parameters_file[parameter_name]

                parameter_dataset_shape = parameter_dataset.shape
                dataset_group_size, dataset_sample_size = parameter_dataset_shape

                group_minimum_size = group_index + 1
                if dataset_group_size < group_minimum_size:
                    parameter_dataset.resize((group_minimum_size, dataset_sample_size))

                parameter_value = parameter_values[parameter_index]
                parameters_file[parameter_name][array_index] = parameter_value

    def getFitParameterNames(self) -> List[str]:
        """
        Get names of fit parameters.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve names from
        """
        return self.fit_parameter_names

    def getParameterNames(self) -> List[str]:
        """
        Get names of fit followed by free parameters.

        :param self: :class:`~Results.OptimizationResultsFileHandler` to retrieve names from
        """
        fit_parameter_names = self.getFitParameterNames()
        free_parameter_names = self.getFreeParameterNames()
        parameter_names = [*fit_parameter_names, *free_parameter_names]
        return parameter_names

    def getFreeParameterName2Quantity(
        self,
        names: Union[str, Iterable[str]] = None,
        output_type: type = dict
    ) -> Union[Quantity, Dict[str, Quantity]]:
        """
        Get values for a free parameter.

        :param self: :class:`~Results.ResultsFileHandler` to retrieve value from
        :param names: name(s) of parameter to retrieve values for
        :param output_type: iterable to output as
            if :paramref:`~Results.ResultsFileHandler.getFreeParameterName2Values.names` is iterable
        """
        parameter_name2quantity = self.free_parameter_name2quantity

        def get(name: str) -> ndarray:
            """Base method for :meth:`~Results.ResultsFileHandler.getFreeParameterValues`"""
            parameter_quantity = parameter_name2quantity[name]
            return parameter_quantity

        free_parameter_names = self.getFreeParameterNames()
        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=output_type,
            default_args=free_parameter_names
        )

    def getParameterSubstitutions(
        self,
        index: Tuple[int, ...],
        names: Union[str, List[str]] = None
    ) -> Dict[str, float]:
        """
        Get dictionary of parameter values.

        :param self: :class:`~Results.GridResultsFileHandler` to retrieve values from
        :param names: name(s) of parameter(s) to retrieve values for.
            Default to all varying parameters.
        :param index: see :meth:`~Results.Results.getResultsOverTime` (not implemented)
        :returns: dictionary.
            Key is name of parameter.
            Value is float value for parameter.
        """
        if names is None:
            names = self.getParameterNames()
        array_index = self.generateArrayIndex(index)

        parameter_name2value = {}
        with self.getParametersFile(index=index) as parameters_file:
            for parameter_name in names:
                parameter_dataset = parameters_file[parameter_name]
                parameter_value = parameter_dataset[array_index]
                parameter_name2value[parameter_name] = parameter_value

        return parameter_name2value
