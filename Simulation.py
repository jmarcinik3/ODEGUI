from functools import partial
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
from numpy import exp, ndarray
from scipy.integrate import odeint
from sympy import Expr, Symbol
from sympy.core import function
from sympy.utilities.lambdify import lambdify

from Results import ResultsFileHandler


def RK4step(ydot: function, y0: ndarray, t0: float, dt: float) -> ndarray:
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


def ODEstep(ydot: function, y0: ndarray, t: List[float]) -> float:
    """
    Get variable vector for ODE after performing one step.

    :param ydot: derivative vector for ODE
    :param y0: initial value vector for variables
    :param t: list of initial time followed by final time
    """
    ynext: float = odeint(ydot, y0, t)[1]
    return ynext


def solveODE(ydot: function, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
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


def formatResultsAsDictionary(t: ndarray, y: ndarray, names: List[str]) -> Dict[str, ndarray]:
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

        assert isinstance(results_file_handler, ResultsFileHandler)
        file_handler_type = results_file_handler.getSimulationType()
        assert file_handler_type == "grid"
        self.saveResult = partial(
            results_file_handler.saveResult,
            index=index,
            close_files=False
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
                name=variable_name,
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
        results_file_handler: ResultsFileHandler,
        parameter_names: Dict[str, float],
        cost_function: Callable[[ndarray], float],
        sample_size: int
    ) -> None:
        """
        Constructor for :class:`~Simulation.RunOptimizationSimulation`.
        Run simulation to find best fitting function for collection of parameters.

        :param variable_names: see :class:`~Simulation.RunSimulation.variable_names`
        :param general_derivative_vector: see :ref:`~Simulation.RunSimulation.general_derivative_vector`
        :param initial_value_vector: see :ref:`~Simulation.RunSimulation.initial_value_vector`
        :param times: see :ref:`~Simulation.RunSimulation.times`
        :param results_file_handler: object to handle saving results
        :param parameter_names: collection of names for parameters to fit to ODE
        :param cost_function: function to determine cost for fit of each set of parameters.
            Takes as input array of parameter values.
            Outputs float of cost.
        :param sample_size: size of data sample that results are fit to
        """
        RunSimulation.__init__(
            self,
            variable_names=variable_names,
            general_derivative_vector=general_derivative_vector,
            initial_value_vector=initial_value_vector,
            times=times
        )

        assert isinstance(parameter_names, Iterable)
        for parameter_name in parameter_names:
            assert isinstance(parameter_name, str)
        self.parameter_names = parameter_names

        assert isinstance(results_file_handler, ResultsFileHandler)
        file_handler_type = results_file_handler.getSimulationType()
        assert file_handler_type == "optimization"
        self.saveResult = partial(
            results_file_handler.saveResult,
            close_files=False
        )

        assert isinstance(cost_function, Callable)
        self.cost_function = cost_function

        assert isinstance(sample_size, int)
        self.sample_size = sample_size

        self.results = {}
        self.iteration_step = -1

    def getParameterNames(self) -> List[str]:
        """
        Get name(s) of free parameters for simulation.

        :params self: :class:`~Simulation.RunOptimizationSimulation` to retrieve names from
        """
        return self.parameter_names

    def getSampleSize(self) -> int:
        """
        Get sample size of data to fit to.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve size from
        """
        return self.sample_size

    def getIterationStep(self) -> int:
        """
        Get most-recent iteration number for steps in minimization.
        """
        return self.iteration_step

    def succeedIterationStep(self) -> None:
        """
        Add one to most-recent iteration number.
        """
        self.iteration_step += 1

    def getCostFunction(self) -> Callable[[ndarray], float]:
        """
        Get function to calculate cost for set of parameters.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve function from
        :returns: callable cost function.
            Takes as input array of parameter values.
            Outputs cost as float.
        """
        return self.cost_function

    def saveODEResults(self) -> None:
        """
        Save ODE results into file.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve save function from
        :param results: results to save into file.
            See :meth:`~Simulation.formatResultsAsDictionary`.
        """
        sample_size = self.getSampleSize()
        iteration_step = self.getIterationStep()
        simulation_index = (sample_size, iteration_step)

        saveResult = self.saveResult
        for variable_name, result in self.results.items():
            saveResult(
                index=simulation_index,
                name=variable_name,
                result=result
            )

    def getCost(self, parameter_values: ndarray) -> float:
        results = self.solveODE(parameter_values)
        self.succeedIterationStep()
        self.saveODEResults(results)

    def solveODE(self, parameter_values: ndarray) -> Dict[str, ndarray]:
        """
        Get solution for each variable in ODE.

        :param self: :class:`~Simulation.RunOptimizationSimulation` to retrieve ODE information from
        :param parameter_values: values for substitute in for parameters in ODE
        :returns: solution for ODE separated by variable.
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
