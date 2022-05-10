import time
from functools import partial
from multiprocessing import Process
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy import exp, ndarray
from scipy.integrate import odeint
from sympy import Expr, Symbol
from sympy.core import function
# noinspection PyUnresolvedReferences
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
    # noinspection PyTypeChecker
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
    # noinspection PyTypeChecker
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


class RunSimulation(Process):
    def __init__(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_name2value: Dict[str, float],
        variable_names: List[str],
        general_derivative_vector: List[Expr],
        y0: ndarray,
        times: ndarray,
        results_file_handler: ResultsFileHandler
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.RunSimulation`.
        Run simulation for a single set of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        :param index: index for free-parameter values.
            Results are saved at this index.
        :param parameter_name2value: dictionary of free-parameter values.
            Key is name of each parameter.
            Value is value of corresponding parameter.
            Defaults to empty dictionary.
        :param variable_names: names of temporal variables in model.
            This gives the order for arguments in the lambdified derivative vector.
        :param general_derivative_vector: partially-simplified, symbolic derivative vector.
            Simplified as much as possible, except leave free parameters and variables as symbolic.
        :param y0: initial condition vector for derivative vector
        :param times: vector of time steps to solve ODE at
        :param results_file_handler: object to handle saving results
        """
        Process.__init__(self)

        derivatives = [
            derivative.subs(parameter_name2value)
            for derivative in general_derivative_vector
        ]
        ydot = lambdify(
            (Symbol('t'), tuple(variable_names)),
            derivatives,
            modules=["math"]
        )
        self.ydot = ydot

        self.saveResult = partial(
            results_file_handler.saveResult,
            index=index,
            close_files=False
        )

        self.results = {}
        self.variable_names = variable_names
        self.y0 = y0
        self.times = times

    def run(self) -> None:
        s3 = time.time()
        self.runSimulation()
        s4 = time.time()
        print("individual:", f"{s4-s3:.3f}")

    def solveODE(self) -> Dict[str, ndarray]:
        results = solveODE(
            self.ydot,
            self.y0,
            self.times
        )
        self.results = formatResultsAsDictionary(*results, self.variable_names)

    def saveODEResults(self) -> None:
        saveResult = self.saveResult
        for variable_name, result in self.results.items():
            saveResult(
                name=variable_name,
                result=result
            )

    def runSimulation(self) -> None:
        """
        Run simulation for a single set of free-parameter values.
        Save results from :class:`~Layout.SimulationWindow.SimulationWindowRunner` to file(s).
        """
        try:
            self.solveODE()
            self.saveODEResults()
        except OverflowError:
            print('overflow:', self.parameter_values)
