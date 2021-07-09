from typing import Dict, List, Tuple

import numpy as np
# noinspection PyUnresolvedReferences
from numpy import exp
from numpy import ndarray
from scipy.integrate import odeint
from sympy.core import function


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


def ODEstep(ydot: function, y0: ndarray, t: List[float, float]) -> float:
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
