from functools import partial
import math
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import PySimpleGUI as sg
from numpy import exp, ndarray
from scipy.integrate import odeint
from sympy import Expr, Symbol
from sympy.core import function
from sympy.utilities.lambdify import lambdify

from Layout.AxisQuantity import AxisQuantityMetadata
from Results import (GridResultsFileHandler, OptimizationResults,
                     OptimizationResultsFileHandler, ResultsFileHandler,
                     postResultsOverTime, preResultsOverTime)
from Transforms.CustomMath import normalizedRmsError


def RK4step(
    ydot: function,
    y0: ndarray,
    t0: float,
    dt: float
) -> ndarray:
    """
    Perform one step of RK4 on ODE.

    :param ydot: derivative vector for ODE
    :param y0: initial value vector for variables
    :param t0: initial time
    :param dt: time step
    """
    array = np.array
    k1 = array(ydot(y0, t0))
    k2 = array(ydot(y0 + k1 * dt / 2, t0 + dt / 2))
    k3 = array(ydot(y0 + k2 * dt / 2, t0 + dt / 2))
    k4 = array(ydot(y0 + k3 * dt, t0 + dt))

    dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y0 + dy


def ODEstep(
    ydot: function,
    y0: ndarray,
    t: List[float]
) -> float:
    """
    Get variable vector for ODE after performing one step.

    :param ydot: derivative vector for ODE
    :param y0: initial value vector for variables
    :param t: list of initial time followed by final time
    """
    ynext: float = odeint(ydot, y0, t)[1]
    return ynext


def solveODE(
    ydot: function,
    y0: ndarray,
    t: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Solve ODE IVP numerically

    :param ydot: derivative vector
    :param y0: initial condition of derivative vector
    :param t: collection of times to solve ODE at
    :returns: 2D ndarray of floats.
        First index corresponds to time step.
        Second index corresponds to simulated variable.
    """
    y: ndarray = odeint(ydot, y0, t, tfirst=True)
    return t, y


def formatResultsAsDictionary(
    t: ndarray,
    y: ndarray,
    names: List[str]
) -> Dict[str, ndarray]:
    """
    Reformat results array from simulation as dictionary.

    :param t: collection of times that simulation was run at
    :param y: results from simulation.
        Same format type as output from :meth:`~Simulation.solveODE`.
    :param names: name(s) of variable(s).
        Index corresponds to those in :paramref:`~Simulation.formatResultsAsDictionary.y`.
    :returns: Dictionary of results from simulation.
        Key is name of variable as string.
        Value is array of floats for variable.
        Array is ordered such that each float corresponds to the time step at the same index.
        Time steps are stored at key 't'.
    """
    name_count = len(names)
    if len(y[0]) != name_count:
        raise ValueError("y and names must have equal length")
    results = dict(zip(names, y.T))
    results['t'] = t
    return results


def symbolicToCallableDerivative(
    symbolic_derivatives: List[Expr],
    variable_names: List[str],
    parameter_name2value: Dict[str, float] = None,
    include_time: bool = True
) -> Callable:
    """
    Convert collection of symbolic derivatives into vectorized callable function.

    :param symbolic_derivatives: collection of symbolic derivatives to convert into callable
    :param variable_names: collection of names for variables in ODE
    :param parameter_name2value: dictionary indicating parameter values to substitute into derivatives.
        Key is name of parameter.
        Value is float value for parameter.
        Only needed if symbolic derivatives have extraneous quantities, outside of given variables.
    :param include_time: set True to include time as argument in callable; set False otherwise.
        Time is first argument of callable, followed by given variables.
    """
    derivatives = [
        derivative.subs(parameter_name2value)
        for derivative in symbolic_derivatives
    ]

    if include_time:
        variables = (Symbol('t'), tuple(variable_names))
    else:
        variables = tuple(variable_names)

    ydot = lambdify(
        variables,
        derivatives,
        modules=["math"]
    )
    return ydot


class RunSimulation:
    def __init__(
        self,
        variable_names: List[str],
        general_derivative_vector: List[Expr],
        initial_value_vector: ndarray,
        times: ndarray
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.RunSimulation`.
        Run simulation for a single set of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        :param variable_names: names of temporal variables in model.
            This gives the order for arguments in the lambdified derivative vector.
        :param general_derivative_vector: partially-simplified, symbolic derivative vector.
            Simplified as much as possible, except leaving free parameters and variables as symbolic.
        :param initial_value_vector: initial condition vector for derivative vector
        :param times: vector of time steps to solve ODE at
        """
        assert isinstance(general_derivative_vector, Iterable)
        for general_derivative in general_derivative_vector:
            assert isinstance(general_derivative, Expr)
        self.general_derivative_vector = general_derivative_vector

        assert isinstance(variable_names, Iterable)
        for variable_name in variable_names:
            assert isinstance(variable_name, str)
        self.variable_names = variable_names

        assert isinstance(initial_value_vector, ndarray)
        self.initial_value_vector = initial_value_vector
        assert isinstance(times, ndarray)
        self.times = times

    def getGeneralDerivativeVector(self) -> List[Expr]:
        """
        Get collection of symbolic derivatives.

        :params self: :class:`~Simulation.RunSimulation` to retrieve derivatives from
        """
        return self.general_derivative_vector

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables in simulation.

        :param self: :class:`~Simulation.RunSimulation` to retrieve names from
        """
        return self.variable_names

    def getInitialValueVector(self) -> ndarray:
        """
        Get array of initial values for ODE.

        :param self: :class:`~Simulation.RunSimulation` to retrieve values from
        """
        return self.initial_value_vector

    def getTimes(self) -> ndarray:
        """
        Get array of times to output simulation at.

        :param self: :class:`~Simulation.RunSimulation` to retrieve values from
        """
        return self.times


class RunGridSimulation(RunSimulation):
    def __init__(
        self,
        variable_names: List[str],
        general_derivative_vector: List[Expr],
        initial_value_vector: ndarray,
        times: ndarray,
        parameter_name2value: Dict[str, float],
        results_file_handler: ResultsFileHandler,
        index: Union[tuple, Tuple[int]]
    ):
        """
        Constructor for :class:`~Simulation.RunGridSimulation`.
        Run simulation for a single set of free-parameter values.

        :param variable_names: see :class:`~Simulation.RunSimulation.variable_names`
        :param general_derivative_vector: see :ref:`~Simulation.RunSimulation.general_derivative_vector`
        :param initial_value_vector: see :ref:`~Simulation.RunSimulation.initial_value_vector`
        :param times: see :ref:`~Simulation.RunSimulation.times`
        :param results_file_handler: object to handle saving results
        :param parameter_name2value: dictionary of free-parameter values.
            Key is name of each parameter.
            Value is value of corresponding parameter.
            Defaults to empty dictionary.
        :param index: index for free-parameter values.
            Results are saved at this index within the saved array.
        """
        RunSimulation.__init__(
            self,
            variable_names=variable_names,
            general_derivative_vector=general_derivative_vector,
            initial_value_vector=initial_value_vector,
            times=times
        )

        assert isinstance(parameter_name2value, dict)
        for parameter_name, parameter_value in parameter_name2value.items():
            assert isinstance(parameter_name, str)
            assert isinstance(parameter_value, (int, float))
        self.parameter_name2value = parameter_name2value

        assert isinstance(results_file_handler, ResultsFileHandler)
        assert isinstance(index, tuple)
        for parameter_index in index:
            assert isinstance(parameter_index, int)

        assert isinstance(results_file_handler, GridResultsFileHandler)
        self.saveResult = partial(
            results_file_handler.saveResult,
            index=index,
            close_file=False
        )

    def getParameterName2Value(self) -> Dict[str, float]:
        """
        Get dictionary from parameter name to parameter value.

        :params self: :class:`~Simulation.RunGridSimulation` to retrieve dictionary from
        """
        return self.parameter_name2value

    def solveODE(self) -> Dict[str, ndarray]:
        """
        Get solution for each variable in ODE.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve ODE information from
        :returns: solution for ODE separated by variable.
            See :meth:`~Simulation.formatResultsAsDictionary`.
        """
        general_derivative_vector = self.getGeneralDerivativeVector()
        variable_names = self.getVariableNames()
        parameter_name2value = self.getParameterName2Value()
        ydot = symbolicToCallableDerivative(
            symbolic_derivatives=general_derivative_vector,
            variable_names=variable_names,
            parameter_name2value=parameter_name2value
        )

        initial_value_vector = self.getInitialValueVector()
        times = self.getTimes()

        results = solveODE(
            ydot,
            initial_value_vector,
            times
        )
        results = formatResultsAsDictionary(*results, variable_names)
        return results

    def saveODEResults(self, results: Dict[str, ndarray]) -> None:
        """
        Save ODE results into file.

        :param self: :class:`~Simulation.RunGridSimulation` to retrieve save function, including index, from
        :param results: results to save into file.
            See :meth:`~Simulation.formatResultsAsDictionary`.
        """
        saveResult = self.saveResult
        for variable_name, result in results.items():
            saveResult(
                quantity_name=variable_name,
                result=result
            )

    def runSimulation(self) -> None:
        """
        Run simulation for a single set of free-parameter values.
        Save results to file(s).
        """
        try:
            results = self.solveODE()
            self.saveODEResults(results)
        except OverflowError:
            parameter_name2value = self.getParameterName2Value()
            print('overflow:', parameter_name2value)


class RunOptimizationSimulation(RunSimulation):
    def __init__(
        self,
        variable_names: List[str],
        general_derivative_vector: List[Expr],
        initial_value_vector: ndarray,
        times: ndarray,
        fit_parameter_names: List[str],
        free_parameter_names: List[str],
        output_axis_quantity_metadata: AxisQuantityMetadata,
        results_obj: OptimizationResults,
        cost_function: Callable[[ndarray, ndarray], float] = normalizedRmsError
    ) -> None:
        """
        Constructor for :class:`~Simulation.RunOptimizationSimulation`.
        Run simulation to find best fitting function for collection of parameters.

        :param variable_names: see :class:`~Simulation.RunSimulation.variable_names`
        :param general_derivative_vector: see :ref:`~Simulation.RunSimulation.general_derivative_vector`
        :param initial_value_vector: see :ref:`~Simulation.RunSimulation.initial_value_vector`
        :param times: see :ref:`~Simulation.RunSimulation.times`
        :param fit_parameter_names: collection for names for parameters acting as independent parameter in fit data
        :param free_parameter_names: collection of names for parameters to fit to ODE
        :param cost_function: see :class:`~ParameterFit.MonteCarloMinimization.cost_function`
        :param output_axis_quantity_metadata: metadata object indicating what functions to perform on ODE solution, to correspond to output data
        :param results_file_handler: object to handle saving results
        """
        RunSimulation.__init__(
            self,
            variable_names=variable_names,
            general_derivative_vector=general_derivative_vector,
            initial_value_vector=initial_value_vector,
            times=times
        )

        assert isinstance(fit_parameter_names, list)
        for fit_parameter_name in fit_parameter_names:
            assert isinstance(fit_parameter_name, str)
        self.fit_parameter_names = fit_parameter_names
        assert isinstance(free_parameter_names, list)
        for free_parameter_name in free_parameter_names:
            assert isinstance(free_parameter_name, str)
        self.free_parameter_names = free_parameter_names
        self.parameter_names = [*fit_parameter_names, *free_parameter_names]

        assert isinstance(results_obj, OptimizationResults)
        self.results_obj = results_obj

        assert isinstance(output_axis_quantity_metadata, AxisQuantityMetadata)
        self.output_axis_quantity_metadata = output_axis_quantity_metadata

        assert isinstance(cost_function, Callable)
        self.cost_function = cost_function

        self.results = {}
        self.iteration_step = -1

    def getResultsObject(self) -> OptimizationResults:
        """
        Get object to handle results for simulation.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve object from
        """
        return self.results_obj

    def getResultsFileHandler(self) -> OptimizationResultsFileHandler:
        """
        Get object to handle saving/loading files.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve object from
        """
        results_obj = self.getResultsObject()
        results_file_handler = results_obj.getResultsFileHandler()
        return results_file_handler

    def getFreeParameterNames(self) -> List[str]:
        """
        Get names of free parameters for simulation.
        These parameters are fit during optimization.

        :params self: :class:`~Simulation.RunOptimizationSimulation` to retrieve names from
        """
        return self.free_parameter_names

    def getFitParameterNames(self) -> List[str]:
        """
        Get names of fit parameters for simulation.
        These parameters are dependent variables for data.

        :params self: :class:`~Simulation.RunOptimizationSimulation` to retrieve names from
        """
        return self.fit_parameter_names

    def getParameterNames(self) -> List[str]:
        """
        Get names of parameters for simulation.
        Ordered with fit-parameters names first, followed by free-parameter names.

        :params self: :class:`~Simulation.RunOptimizationSimulation` to retrieve names from
        """
        return self.parameter_names

    def getIterationStep(self) -> int:
        """
        Get most-recent iteration number for steps in minimization, starting at zero.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve number from
        """
        return self.iteration_step

    def succeedIterationStep(self) -> None:
        """
        Add one to most-recent iteration number.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to succeed iteration in
        """
        self.iteration_step += 1

    def resetIterationStep(self) -> None:
        """
        Reset iteration number to begin new set of fits.
        Set to -1 so that first result had iteration step of 0.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to reset iteration in
        """
        self.iteration_step = -1
        results_file_handler = self.getResultsFileHandler()
        results_file_handler.closeResultsFiles()

    def getOutputAxisQuantityMetadata(self) -> AxisQuantityMetadata:
        """
        Get object to retrieve metadata for simulated output vector.
        """
        return self.output_axis_quantity_metadata

    def getCostFunction(self) -> Callable[[ndarray, ndarray], float]:
        """
        Get cost function to compare experimental data to simulated data.
        Takes as input: output experimental data and output simulated data.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve function from
        """
        return self.cost_function

    def getCost(
        self,
        data: ndarray,
        simulated_data: ndarray
    ) -> float:
        """
        Get cost of given data vs. given simulated array.
        Then close all results files.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve cost form
        :param data: data to compare simulated results to
        :param simulated_data: results to compare to original/experimental data
        :param
        """
        assert data.shape == simulated_data.shape

        cost_function = self.getCostFunction()
        cost = cost_function(data, simulated_data)

        results_file_handler = self.getResultsFileHandler()
        results_file_handler.closeResultsFiles()

        return cost

    def saveODEResults(
        self,
        results: Dict[str, ndarray],
        parameter_values: ndarray,
        sample_size: int
    ) -> None:
        """
        Save ODE results into file.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve save function from
        :param results: results to save into file.
            See :meth:`~Simulation.formatResultsAsDictionary`.
        :param parameter_values: 1D array of parameter values, fit parameters followed by free parameters
        :param sample_size: number of sample points in data that simulation is fit to
        """
        iteration_step = self.getIterationStep()
        group_index = math.floor(iteration_step / sample_size)
        sample_index = iteration_step % sample_size
        parameter_index = (sample_size, group_index, sample_index)
        results_file_handler = self.getResultsFileHandler()

        results_file_handler.saveParametersSet(
            index=parameter_index,
            parameter_values=parameter_values
        )

        saveResult = partial(
            results_file_handler.saveResult,
            close_file=False
        )
        for variable_name, result in results.items():
            saveResult(
                index=parameter_index,
                quantity_name=variable_name,
                result=result
            )

    def solveODE(
        self,
        parameter_values: ndarray,
    ) -> Dict[str, ndarray]:
        """
        Get solution for each variable in ODE.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve ODE information from
        :param parameter_values: values for substitute in for parameters in ODE
        :returns: solution for ODE organized by variable.
            See :meth:`~Simulation.formatResultsAsDictionary`.
        """
        general_derivative_vector = self.getGeneralDerivativeVector()
        variable_names = self.getVariableNames()
        parameter_names = self.getParameterNames()
        parameter_name2value = dict(zip(parameter_names, parameter_values))
        ydot = symbolicToCallableDerivative(
            symbolic_derivatives=general_derivative_vector,
            variable_names=variable_names,
            parameter_name2value=parameter_name2value
        )

        initial_value_vector = self.getInitialValueVector()
        times = self.getTimes()

        results = solveODE(
            ydot,
            initial_value_vector,
            times
        )
        results = formatResultsAsDictionary(*results, variable_names)
        return results

    def solveAndSaveODE(
        self,
        parameter_values: ndarray,
        sample_size: int
    ) -> Dict[str, ndarray]:
        """
        Get solution for each variable in ODE, then save solution.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve ODE information from
        :param parameter_values: values for substitute in for parameters in ODE
        :param sample_size: see :meth:`~Simulation.RunOptimizationSimulation.sample_size`
        :returns: solution for ODE organized by variable (or numpy nan if solution diverges).
            See :meth:`~Simulation.formatResultsAsDictionary`.
        """
        self.succeedIterationStep()
        try:
            results = self.solveODE(parameter_values)
            self.saveODEResults(
                results,
                sample_size=sample_size,
                parameter_values=parameter_values
            )
            return results
        except OverflowError:
            parameter_names = self.getParameterNames()
            parameter_name2value = dict(zip(parameter_names, parameter_values))
            print('overflow:', parameter_name2value)
            return np.nan

    def getResultsFit(
        self,
        fit_parameter_values: ndarray,
        free_parameter_values: ndarray
    ) -> ndarray:
        """
        Get simulated output vector by solving ODE.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve fitting criteria from
        :param fit_parameter_values: parameters that are independent variables for data
        :param free_parameter_values: parameters to fit to given experimental data
        """
        def updateProgressMeter(current_value: int) -> sg.OneLineProgressMeter:
            """
            Update progress meter.

            :param current_value: present number of simulation being calculated
            """
            iteration_count = self.getIterationStep()

            return sg.OneLineProgressMeter(
                title=f"{iteration_count:d} Iterations",
                orientation="horizontal",
                current_value=current_value,
                max_value=sample_size,
                keep_on_top=True
            )

        fit_parameter_dimension = fit_parameter_values.ndim
        assert fit_parameter_dimension == 2
        sample_size, fit_parameter_count = fit_parameter_values.shape

        if fit_parameter_dimension == 2:
            output_axis_quantity_metadata = self.getOutputAxisQuantityMetadata()
            is_functional = output_axis_quantity_metadata.isFunctional()
            assert is_functional

            show_progress = sample_size >= 2
            post_results = np.zeros(sample_size)
            for sample_index in range(sample_size):
                if show_progress and not updateProgressMeter(sample_index):
                    break

                fit_parameter_value = fit_parameter_values[sample_index]
                parameter_values = np.concatenate((fit_parameter_value, free_parameter_values))

                results = self.solveAndSaveODE(
                    parameter_values,
                    sample_size
                )

                quantity_names = output_axis_quantity_metadata.getAxisQuantityNames(include_none=False)
                quantity_count = len(quantity_names)
                assert quantity_count == 1
                quantity_name = quantity_names[0]

                if isinstance(results, dict):
                    post_result = results[quantity_name]
                    post_result = preResultsOverTime(
                        post_result,
                        axis_quantity_metadata=output_axis_quantity_metadata
                    )
                    post_result = postResultsOverTime(
                        post_result,
                        axis_quantity_metadata=output_axis_quantity_metadata
                    )
                    post_results[sample_index] = post_result
                else:
                    assert np.isnan(results)
                    post_results[sample_index] = results

            if show_progress:
                updateProgressMeter(sample_size)

        return post_results
