from typing import Dict, List, Tuple

import numpy as np
# noinspection PyUnresolvedReferences
from numpy import exp
from numpy import ndarray
from scipy.integrate import odeint
from sympy.core import function


def RK4step(derivative, y0, t0, dt):
    k1 = np.array(derivative(y0, t0))
    k2 = np.array(derivative(y0 + k1 * dt / 2, t0 + dt / 2))
    k3 = np.array(derivative(y0 + k2 * dt / 2, t0 + dt / 2))
    k4 = np.array(derivative(y0 + k3 * dt, t0 + dt))

    dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # print(k1,k2,k3,k4,dy)
    return y0 + dy


def ODEstep(ydot, y0, t):
    """
    __Purpose__
        Get collection of values for function(s) after performing one step
    __Input__
        ydot [lambda]: ODE to solve, collection of derivatives
        y0 [float]: initial value of variables
        t [list of float]: [initial time, final time] where final_time=initial_time+delta_time
    __Return__
        list of float
    """
    ynext = odeint(ydot, y0, t)[1]
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
    :param y: results from simulation. Same format type as output from :meth:`~Simulation.solveODE`
    :param names: name(s) of variable(s). Index corresponds to those in :paramref:`~Simulation.formatResultsAsDictionary.y`
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
