"""
This file contains classes relating to the simulation window.
Free parameters are parameters for which multiple values are simulated.
The simulation must be run once by hitting the button before the window functions.
"""
from __future__ import annotations

import os
import tkinter as tk
import traceback
from functools import partial
from itertools import product
from os.path import basename, join
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
# noinspection PyPep8Naming
# noinspection PyPep8Naming
from zipfile import ZipFile

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import yaml
from Function import Model
from macros import StoredObject, recursiveMethod
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import ndarray
from pint import Quantity, Unit
from Results import Results
from Simulation import formatResultsAsDictionary, solveODE
from sympy import Expr, Symbol
from sympy.core import function
from sympy.utilities.lambdify import lambdify
from YML import config_file_types, getDimensions, getStates

from Layout.Layout import Element, Layout, Row, Tab, TabbedWindow, TabGroup, WindowRunner, getKeys, storeElement

cc_pre = "CANVAS CHOICE"
ccs_pre = ' '.join((cc_pre, "SPECIE"))
fps_pre = "FREE_PARAMETER SLIDER"


def drawFigure(canvas: tk.Canvas, figure: Figure) -> FigureCanvasTkAgg:
    """
    Draw figure on canvas.

    :param canvas: canvas to draw figure on
    :param figure: figure containing data to draw on canvas
    """
    figure_canvas = FigureCanvasTkAgg(figure, canvas)
    figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas.draw()
    return figure_canvas


# noinspection PyUnboundLocalVariable
def getFigure(
        results: Dict[str, ndarray],
        scale_factor: Dict[str, float] = None,
        plot_type: str = "xy",
        segment_count: int = 100,
        clim: Tuple[Optional[float], Optional[float]] = (None, None),
        autoscalec_on: bool = True,
        colormap: str = None,
        colorbar_kwargs: Dict[str, Any] = None,
        axes_kwargs: Dict[str, Any] = None,
        plot_kwargs: Dict[str, Any] = None,
        inset_parameters: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Figure:
    """
    Get matplotlib figure from data.

    :param results: dictionary of results.
        Key is name of axis.
        Value is values to to plot along axis.
    :param scale_factor: dictionary of scale factor for each axis.
        Key is name of axis.
        Value is scale factor.
        Result along corresponding axis is divided by this factor.
        Defaults to 1 for each axis not given.
    :param clim: axis bounds for colorbar.
        First element gives lower limit.
        Defaults to minium of :paramref:`~SimulationWindow.getFigure.c`.
        Second element gives upper limit.
        Defaults to maximum of :paramref:`~SimulationWindow.getFigure.c`.
    :param autoscalec_on: set True to autoscale colorbar axis.
        Set False otherwise.
        Overrides :paramref:`~SimulationWindow.getFigure.clim` when set True.
    :param colormap: colormap to use for colorbar.
    :param plot_type: type of plot to display.
        This could include a single curve, multiple curves, scatter plot.
    :param segment_count: number of segments for colorbar.
        Only called when a single line is multicolored.
    :param inset_parameters: 2-level dictionary of parameters to show on inset plot.
        Key is name of parameter.
        Subkeys are "range" and "value"
            "range": value is tuple (minimum, maximum) for parameter
            "value": value is value for parameter within range
    :param colorbar_kwargs: additional arguments to pass into :class:`matplotlib.pyplot.colorbar`
    :param axes_kwargs: additional arguments to pass into :class:`matplotlib.axes.Axes'
    :param plot_kwargs: additional arguments to pass into axes plot method
    """

    def getLimits(data: ndarray, axis_name: str) -> Tuple[float, float]:
        """
        Get axis limits

        :param data: data to retrieve limits for.
        :param axis_name: name of axis to check for limits in axes_kwargs
        :returns: Tuple of (minimum, maximum) range for plot axis.
            Defaults to (data.min(), data.max())
        """
        try:
            limits = list(axes_kwargs[f"{axis_name:s}lim"])
            if limits[0] is None:
                limits[0] = data.min()
            if limits[1] is None:
                limits[1] = data.max()
            limits = tuple(limits)
        except KeyError:
            limits = (data.min(), data.max())

        return limits

    if plot_kwargs is None:
        plot_kwargs = {}
    if axes_kwargs is None:
        axes_kwargs = {}

    axis_names = results.keys()

    if scale_factor is None:
        scale_factor = {}
    for axis_name in axis_names:
        if axis_name not in scale_factor.keys():
            scale_factor[axis_name] = 1

    figure = plt.figure()

    if 'x' in plot_type:
        x = results['x'] / scale_factor['x']
        xsize = x.size
        xshape = x.shape
        axes_kwargs["xlim"] = getLimits(x, 'x')

    if 'y' in plot_type:
        y = results['y'] / scale_factor['y']
        ysize = y.size
        yshape = y.shape
        axes_kwargs["ylim"] = getLimits(y, 'y')

    if 'z' in plot_type:
        z = results['z'] / scale_factor['z']
        zsize = z.size
        zshape = z.shape
        axes_kwargs["zlim"] = getLimits(z, 'z')
        axes = figure.add_subplot(projection="3d")
    else:
        axes = figure.add_subplot()

    axes.set(**axes_kwargs)

    if 'c' in plot_type or 't' in plot_type:
        c = results['c'] / scale_factor['c']
        csize = c.size
        cshape = c.shape

        if colorbar_kwargs is None:
            colorbar_kwargs = {}

        if not autoscalec_on:
            vmin = c.min() if clim[0] is None else clim[0]
            vmax = c.max() if clim[1] is None else clim[1]
        else:
            vmin, vmax = c.min(), c.max()

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # noinspection PyUnresolvedReferences
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        figure.colorbar(cmap, **colorbar_kwargs)
        try:
            c_colors = cmap.to_rgba(c)
        except ValueError:
            pass

    if inset_parameters is not None:
        free_parameter_ranges = {name: inset_parameters[name]["range"] for name in inset_parameters.keys()}
        inset_axes, inset_axes_plot = getParameterInsetAxes(axes, free_parameter_ranges)
        free_parameter_values = tuple(inset_parameters[name]["value"] for name in inset_parameters.keys())
        inset_axes_plot(*free_parameter_values)

    x_length = len(x)
    if 'z' not in plot_type:
        if plot_type in ["xy", "nxy", "xny"]:
            assert xshape == yshape
            axes.plot(x, y, **plot_kwargs)
        elif plot_type == "cnxny":
            assert cshape == (xsize, ysize)
            axes.contourf(
                x, y, c.T, 
                levels=segment_count, 
                cmap=colormap, 
                norm=norm, 
                **plot_kwargs
            )
        elif plot_type in ["txy", "cnxy", "cxny"]:
            assert cshape == xshape == yshape

            segment_count = max(min(segment_count, cshape[-1]), 1)
            segment_length = round(x_length / segment_count)

            if segment_length == 0:
                axes.scatter(x, y, color=c_colors)
            else:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(segments, cmap=colormap, norm=norm)
                line_collection.set_array(c)
                axes.add_collection(line_collection, **plot_kwargs)
        elif plot_type in ["ncxy", "ncxny", "ncnxy"]:
            if plot_type == "ncxy":
                assert xshape == yshape and xshape[0] == csize
                x_lines, y_lines = x, y
            elif plot_type == "ncxny":
                assert xshape == (csize, ysize)
                x_lines = x
                y_lines = np.tile(y, (csize, 1))
            elif plot_type == "ncnxy":
                assert yshape == (csize, xsize)
                x_lines = np.tile(x, (csize, 1))
                y_lines = y

            for line_index in range(csize):
                axes.plot(
                    x_lines[line_index],
                    y_lines[line_index], 
                    color=c_colors[line_index], 
                    **plot_kwargs
                )
        elif plot_type in ["tnxy", "txny"]:
            if plot_type == "txny":
                assert cshape == xshape and cshape[0] == ysize
                x_lines = x
                y_lines = np.tile(y, (cshape[-1], 1)).T
                line_count = ysize
            elif plot_type == "tnxy":
                assert cshape == yshape and cshape[0] == xsize
                x_lines = np.tile(x, (cshape[-1], 1)).T
                y_lines = y
                line_count = xsize

            for line_index in range(line_count):
                points = np.array([x_lines[line_index], y_lines[line_index]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(segments, cmap=colormap, norm=norm)
                line_collection.set_array(c[line_index])
                axes.add_collection(line_collection, **plot_kwargs)
    elif 'z' in plot_type:
        if plot_type == "xyz":
            assert xshape == yshape == zshape
            axes.plot3D(x, y, z, **plot_kwargs)
        elif plot_type == "txyz":
            assert cshape == xshape == yshape == zshape

            segment_count = max(min(segment_count, cshape[-1]), 1)
            segment_length = round(x_length / segment_count)

            if segment_length == 0:
                axes.scatter3D(x, y, z, color=c_colors)
            else:
                points = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = Line3DCollection(segments, cmap=colormap, norm=norm)
                line_collection.set_array(c)
                axes.add_collection(line_collection, **plot_kwargs)
        elif plot_type in ["nxyz", "xnyz", "xynz"]:
            if plot_type == "nxyz":
                assert yshape == zshape and yshape[0] == xsize
                x_rots = np.tile(x, (yshape[1], 1)).T
                y_rots = y
                z_lines = z
                rot_size = xsize
            elif plot_type == "xnyz":
                assert xshape == zshape and xshape[0] == ysize
                x_rots = x
                y_rots = np.tile(y, (xshape[1], 1)).T
                z_lines = z
                rot_size = ysize
            elif plot_type == "xynz":
                assert xshape == yshape and xshape[0] == zsize
                x_rots = x
                y_rots = y
                z_lines = np.tile(z, (xshape[1], 1)).T
                rot_size = zsize

            for line_index in range(rot_size):
                axes.plot3D(
                    x_rots[line_index], 
                    y_rots[line_index], 
                    z_lines[line_index], 
                    **plot_kwargs
                )
        elif plot_type in ["tnxyz", "txnyz", "txynz"]:
            if plot_type == "tnxyz":
                assert cshape == yshape == zshape and yshape[0] == xsize
                x_rots, y_rots, zs = y, z, x
                zdir = 'x'
            elif plot_type == "txnyz":
                assert cshape == xshape == zshape and xshape[0] == ysize
                x_rots, y_rots, zs = x, z, y
                zdir = 'y'
            elif plot_type == "txynz":
                assert cshape == xshape == yshape and xshape[0] == zsize
                x_rots, y_rots, zs = x, y, z
                zdir = 'z'

            for line_index in range(zs.size):
                points = np.array([x_rots[line_index], y_rots[line_index]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(segments, cmap=colormap, norm=norm)
                line_collection.set_array(c[line_index])
                axes.add_collection3d(
                    line_collection, 
                    zs=zs[line_index], 
                    zdir=zdir, 
                    **plot_kwargs
                )
        elif plot_type in ["ncxyz", "ncnxyz", "ncxnyz", "ncxynz"]:
            if plot_type == "ncxyz":
                assert xshape == yshape == zshape and xshape[0] == csize
                x_rots, y_rots, z_lines = x, y, z
                extra_dimension = False
            elif plot_type == "ncnxyz":
                assert yshape[0:2] == (csize, xsize) and yshape == zshape
                try:
                    x_rots = np.transpose(np.tile(x, (csize, yshape[2], 1)), axes=(0, 2, 1))
                    rot_size = xsize
                    extra_dimension = True
                except IndexError:
                    x_rots = np.tile(x, (csize, 1))
                    extra_dimension = False
                y_rots, z_lines = y, z
            elif plot_type == "ncxnyz":
                assert xshape[0:2] == (csize, ysize) and xshape == zshape
                x_rots, z_lines = x, z
                try:
                    y_rots = np.transpose(np.tile(y, (csize, xshape[2], 1)), axes=(0, 2, 1))
                    rot_size = ysize
                    extra_dimension = True
                except IndexError:
                    y_rots = np.tile(y, (csize, 1))
                    extra_dimension = False
            elif plot_type == "ncxynz":
                assert xshape[0:2] == (csize, zsize) and xshape == yshape
                x_rots, y_rots = x, y
                try:
                    z_lines = np.transpose(np.tile(z, (csize, xshape[2], 1)), axes=(0, 2, 1))
                    rot_size = zsize
                    extra_dimension = True
                except IndexError:
                    z_lines = np.tile(z, (csize, 1))
                    extra_dimension = False

            for c_index in range(csize):
                c_color = c_colors[c_index]
                if extra_dimension:
                    for z_index in range(rot_size):
                        line_index = (c_index, z_index)
                        axes.plot3D(
                            x_rots[line_index],
                            y_rots[line_index],
                            z_lines[line_index],
                            color=c_color,
                            **plot_kwargs
                        )
                else:
                    axes.plot3D(
                        x_rots[c_index],
                        y_rots[c_index],
                        z_lines[c_index],
                        color=c_color,
                        **plot_kwargs
                    )
        elif plot_type in ["tnxnyz", "tnxynz", "txnynz"]:
            if plot_type == "tnxnyz":
                assert cshape[0:2] == (xsize, ysize) and cshape == zshape
                yrot_size = xsize
                x_rots = z
                y_rots = np.tile(x, (cshape[2], 1)).T
                zs, zdir = y, 'y'
                c_rots = c
            elif plot_type == "tnxynz":
                assert cshape[0:2] == (xsize, zsize) and cshape == yshape
                yrot_size = zsize
                x_rots = np.transpose(y, axes=(1, 0, 2))
                y_rots = np.tile(z, (cshape[2], 1)).T
                zs, zdir = x, 'x'
                c_rots = np.transpose(c, axes=(1, 0, 2))
            elif plot_type == "txnynz":
                assert cshape[0:2] == (ysize, zsize) and cshape == xshape
                yrot_size = ysize
                x_rots = x
                y_rots = np.tile(y, (cshape[2], 1)).T
                zs, zdir = z, 'z'
                c_rots = c

            for y_index in range(yrot_size):
                for z_index in range(zs.size):
                    if plot_type == "tnxnyz":
                        points = np.array([y_rots[y_index], x_rots[y_index, z_index]]).T.reshape(-1, 1, 2)
                    else:
                        points = np.array([x_rots[y_index, z_index], y_rots[y_index]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    line_collection = LineCollection(segments, cmap=colormap, norm=norm)
                    line_collection.set_array(c_rots[y_index, z_index])
                    axes.add_collection3d(line_collection, zs=zs[z_index], zdir=zdir, **plot_kwargs)
        elif plot_type in ["cnxnyz", "cnxynz", "cxnynz", "cnxnynz", "ncnxnyz", "ncnxynz", "ncxnynz"]:
            if plot_type == "cnxnyz":
                assert cshape == zshape == (xsize, ysize)
                c_shaped = c.reshape(csize)
                x_shaped = np.tile(x, (ysize, 1)).T
                y_shaped = np.tile(y, (xsize, 1))
                z_shaped = z
            elif plot_type == "cnxynz":
                assert cshape == yshape == (xsize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = np.tile(x, (zsize, 1)).T
                y_shaped = y
                z_shaped = np.tile(z, (xsize, 1))
            elif plot_type == "cxnynz":
                assert cshape == xshape == (ysize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = x
                y_shaped = np.transpose(np.tile(y, (zsize, 1)))
                z_shaped = np.tile(z, (ysize, 1))
            elif plot_type == "cnxnynz":
                assert cshape == (xsize, ysize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = np.transpose(np.tile(x, (ysize, zsize, 1)), axes=(2, 0, 1))
                y_shaped = np.transpose(np.tile(y, (zsize, xsize, 1)), axes=(1, 2, 0))
                z_shaped = np.tile(z, (xsize, ysize, 1))
            elif plot_type == "ncnxnyz":
                assert zshape == (csize, xsize, ysize)
                c_shaped = np.transpose(np.tile(c, (xsize, ysize, 1)), axes=(2, 0, 1)).reshape(zsize)
                x_shaped = np.transpose(np.tile(x, (ysize, csize, 1)), axes=(1, 2, 0))
                y_shaped = np.tile(y, (csize, xsize, 1))
                z_shaped = z
            elif plot_type == "ncnxynz":
                assert yshape == (csize, xsize, zsize)
                c_shaped = np.transpose(np.tile(c, (xsize, zsize, 1)), axes=(2, 0, 1)).reshape(ysize)
                x_shaped = np.transpose(np.tile(x, (zsize, csize, 1)), axes=(1, 2, 0))
                y_shaped = y
                z_shaped = np.tile(z, (csize, xsize, 1))
            elif plot_type == "ncxnynz":
                assert xshape == (csize, ysize, zsize)
                c_shaped = np.transpose(np.tile(c, (ysize, zsize, 1)), axes=(2, 0, 1)).reshape(xsize)
                x_shaped = x
                y_shaped = np.transpose(np.tile(y, (zsize, csize, 1)), axes=(1, 2, 0))
                z_shaped = np.tile(z, (csize, ysize, 1))

            axes.scatter3D(x_shaped, y_shaped, z_shaped, color=cmap.to_rgba(c_shaped))

        axes.set_xlim3d(*axes_kwargs["xlim"])
        axes.set_ylim3d(*axes_kwargs["ylim"])
        axes.set_zlim3d(*axes_kwargs["zlim"])

    return figure


def clearFigure(figure_canvas: FigureCanvasTkAgg) -> None:
    """
    Clear figure on figure-canvas aggregate.

    :param figure_canvas: figure-canvas aggregate to clear canvas on
    """
    if isinstance(figure_canvas, FigureCanvasTkAgg):
        figure_canvas.get_tk_widget().forget()
    plt.close("all")


def getParameterInsetAxes(
        axes: Axes,
        parameter_values: Dict[str, Tuple[float, float]],
        color: str = "black",
        marker: str = 'o',
        markersize: float = 2,
        linestyle: str = '',
        clip_on: bool = False,
        labelpad: int = 2,
        fontsize: int = 8
) -> Tuple[Axes, partial]:
    """
    Get inset axes object and partially-completed plot for inset axes.
    This inset plot displays values for up to two free parameters during a parameter sweep.
    
    :param axes: axes to nest inset plot inside into
    :param parameter_values: dictionary of parameter values.
        Key is name of parameter.
        Value is 2-tuple of (minimum, maximum) values for parameter.
    :param color: marker color for inset plot
    :param marker: marker type for inset plot
    :param markersize: marker size inset plot
    :param linestyle: linestyle for inset plot
    :param clip_on: set True to allow points to extend outside inset plot. Set False otherwise.
    :param labelpad: padding for axis label(s)
    :param fontsize: fontsize for axis label(s)
    :returns: Tuple of (inset axes object, plot function taking single point of free parameters as input).
    """
    parameter_names = list(parameter_values.keys())
    parameter_count = len(parameter_names)
    axins_kwargs = {
        "color": color,
        "marker": marker,
        "markersize": markersize,
        "linestyle": linestyle,
        "clip_on": clip_on
    }
    label_kwargs = {
        "labelpad": labelpad,
        "fontsize": fontsize
    }

    location = [0.88, 0.88, 0.1, 0.1]  # [1, 1.035, 0.1, 0.1]
    axins = axes.inset_axes(location)
    axins.set_xticks(())
    axins.set_yticks(())
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    if parameter_count == 2:
        axins.set_ylabel(parameter_names[1], **label_kwargs)
        ylim = list(parameter_values.values())[1]
        axins_plot = partial(axins.plot, **axins_kwargs)
    elif parameter_count == 1:
        axins.spines["left"].set_visible(False)
        axins.spines["bottom"].set_position(("axes", 0.5))
        ylim = -1, 1
        axins_plot = lambda x: axins.plot(x, 0, **axins_kwargs)
    else:
        raise ValueError("must have exactly one or two parameters in inset plot")

    xlim = list(parameter_values.values())[0]
    axins.set_xlabel(parameter_names[0], **label_kwargs)
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.set_facecolor("none")

    return axins, axins_plot


def calculateResolution(
    minimum: float, 
    maximum: float, 
    step_count: int
) -> float:
    """
    Calculate resolution from minimum, maximum, and step count.

    :param minimum: minimum value
    :param maximum: maximum value
    :param step_count: number of values from minimum to maximum, inclusive
    """
    range_values = maximum - minimum
    distinct_values = step_count - 1
    if distinct_values == 0:
        return 0
    elif distinct_values >= 1:
        return range_values / distinct_values
    else:
        raise ValueError("count must be int at least 1")


def getUnitConversionFactor(
    old_units: Union[Unit, Quantity], 
    new_units: Union[Unit, Quantity] = None
) -> float:
    """
    Get unit conversion factor.

    :param old_units: units or quantity with units to convert from
    :param new_units: units or quantity with units to convert to
    """
    old_quantity = 1.0 * old_units
    if new_units is None:
        conversion_factor = old_quantity.to_base_units()
    else:
        conversion_factor = old_quantity.to_reduced_units(new_units)
    return conversion_factor.magnitude


class ParameterSlider(Element, StoredObject):
    """
    Slider to choose value for free parameter.
    This contains
        #. Four labels. One for parameter name. One for minimum parameter value. One for maximum parameter value.
        One for number of distinct parameter values
        #. Slider. This allows the user to choose which parameter value to plot a simulation for.
    
    :ivar name: name of parameter
    :ivar minimum: minimum value of parameter
    :ivar maximum: maximum value of parameter
    :ivar stepcount: number of disticnt parameter values
    :ivar units: units of parameter
    """

    def __init__(self, name: str, window: SimulationWindow, values: Tuple[float, float, int, Unit]) -> None:
        """
        Constuctor for :class:`~Layout.SimulationWindow.ParameterSlider`.
        
        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to initialize
        :param name: name of parameter
        :param window: window in which slider object will be displayed
        :param values: tuple of info giving parameter values.
            First value is minimum value of parameter.
            Second value is maximum value of parameter.
            Third value is number of distinct parameter values.
            Fourth value is units of parameter.
        """
        Element.__init__(self, window)
        StoredObject.__init__(self, name)

        self.minimum = values[0]
        self.maximum = values[1]
        self.stepcount = values[2]
        self.units = values[3]

    def getName(self) -> str:
        """
        Get name of parameter.
        
        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve name of
        """
        return self.name

    def getMinimum(self) -> float:
        """
        Get minimum value of parameter.
        
        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve minimum value of
        """
        return self.minimum

    def getMaximum(self) -> float:
        """
        Get maximum value of parameter.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve maximum value of
        """
        return self.maximum

    def getStepCount(self) -> int:
        """
        Get number of distinct parameter values.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve number of values for
        """
        return self.stepcount

    def getResolution(self) -> float:
        """
        Get resolution of parameter values.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve resolution of
        """
        return calculateResolution(self.getMinimum(), self.getMaximum(), self.getStepCount())

    def getNameLabel(self) -> Union[sg.Image, sg.Text]:
        """
        Get label to display parameter name.
        
        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve label for
        """
        return getTexImage(
            name=self.getName(),
            size=self.getDimensions(name="parameter_slider_name_label")
        )

    def getMinimumLabel(self) -> sg.Text:
        """
        Get label to display minimum parameter value.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve label for
        """
        return sg.Text(
            text=formatValue(self.getMinimum()),
            size=self.getDimensions(name="parameter_slider_minimum_label")
        )

    def getMaximumLabel(self) -> sg.Text:
        """
        Get label to display maximum parameter value.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve label for
        """
        return sg.Text(
            text=formatValue(self.getMaximum()),
            size=self.getDimensions(name="parameter_slider_maximum_label")
        )

    def getStepCountLabel(self) -> sg.Text:
        """
        Get label to display number of distinct parameter values.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve label for
        """
        return sg.Text(
            text=self.getStepCount(),
            size=self.getDimensions(name="parameter_slider_stepcount_label")
        )

    @storeElement
    def getSlider(self) -> sg.Slider:
        """
        Get slider to take user for value of parameter.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve slider from
        """
        minimum = self.getMinimum()
        maximum = self.getMaximum()

        return sg.Slider(
            range=(minimum, maximum),
            default_value=minimum,
            resolution=self.getResolution(),
            orientation="horizontal",
            enable_events=True,
            size=self.getDimensions(name="parameter_slider_slider"),
            border_width=0,
            pad=(0, 0),
            key=f"-{fps_pre:s} {self.getName():s}-"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for slider object.

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve slider object from
        """
        name_label = self.getNameLabel()
        minimum_label = self.getMinimumLabel()
        maximum_label = self.getMaximumLabel()
        stepcount_label = self.getStepCountLabel()
        slider = self.getSlider()

        row = Row(
            window=self.getWindowObject(),
            elements=[name_label, minimum_label, slider, maximum_label, stepcount_label]
        )
        layout = Layout(rows=row)
        return layout.getLayout()


class SimulationTab(Tab):
    """
    This class contains the layout for the simulation tab in the simulation window.
        #. Input fields to set time steps.
        This allows the user to set the minimum, maximum, and number of steps for time in the simulation.
        #. Run button. This allows the user to run the simulation.
        This is particularly useful if the user wishes to run the simulation again with more precision.
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.SimulationTab`.
        
        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        super().__init__(name, window)

    @storeElement
    def getInitialTimeInputElement(self) -> sg.InputText:
        """
        Get element to take user input for initial time.
        This allows user to choose what time to start the simulation at.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve element from
        """
        return sg.InputText(
            default_text=getStates("time", "initial"),
            tooltip="Enter initial time (seconds)",
            size=self.getDimensions(name="initial_time_input_field"),
            key="-INITIAL TIME-"
        )

    @storeElement
    def getTimeStepCountInputElement(self):
        """
        Get element to take user input for number of time steps.
        This allows user to choose how many time steps to save in simulation results.

        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve element from
        """
        return sg.InputText(
            default_text=getStates("time", "step_count"),
            tooltip="Enter number of time-steps",
            size=self.getDimensions(name="timestep_count_input_field"),
            key="-STEPCOUNT TIME-"
        )

    @storeElement
    def getFinalTimeInputElement(self) -> sg.InputText:
        """
        Get element to take user input for final time.
        This allows user to choose what time to end the simulation at.

        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve element from
        """
        return sg.InputText(
            default_text=getStates("time", "final"),
            tooltip="Enter final time (seconds)",
            size=self.getDimensions(name="final_time_input_field"),
            key="-FINAL TIME-"
        )

    @storeElement
    def getRunButton(self) -> sg.Button:
        """
        Get element allowing user to start simulation.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve element from
        """
        text = "Run Simulation"

        return sg.Button(
            button_text=text,
            key=f"-{text.upper():s}-"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for simulation tab.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve layout from
        """
        time_inputs = [
            self.getInitialTimeInputElement(),
            self.getFinalTimeInputElement(),
            self.getTimeStepCountInputElement()
        ]
        run_button = self.getRunButton()

        window = self.getWindowObject()
        layout = Layout()
        layout.addRows(Row(window=window, elements=time_inputs))
        layout.addRows(Row(window=window, elements=run_button))
        return layout.getLayout()


class ColorbarTab(Tab, StoredObject):
    """
    This class contains the layout for the aesthetics tab in the simulation window.
        #. Header row to identify purpose for each column
        #. Input fields to set lower and upper limit for colorbar
        #. Checkbox to choose whether colorbar is autoscaled or manually scaled
        #. Combobox to choose colobar scale type (e.g. linear, logarithmic)
        #. Spin to set scale factor for colorbar
        #. Combobox to choose colorbar colormap
        #. Spin to set segment count for colormap
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.ColorbarTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the purpose of each input column.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve row for
        """
        window_object = self.getWindowObject()

        top_row = Row(window=window_object)
        texts = ["Limits", "Scale"]
        dimension_keys = [f"axis_header_row_{string:s}" for string in ["limits", "scale"]]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="center"
            )
            top_row.addElements(text_element)

        bottom_row = Row(window=window_object)
        texts = ["Title", "Lower", "Upper", "Auto", "Factor"]
        dimension_keys = [
            f"axis_header_row_{string:s}"
            for string in
            ["element_name", "element_title", "lower_limit", "upper_limit", "autoscale", "scale_factor"]
        ]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="left"
            )
            bottom_row.addElements(text_element)
        
        return [top_row, bottom_row]

    @storeElement
    def getLimitInputElements(self) -> Tuple[sg.InputText, sg.InputText]:
        """
        Get elements that allow user to input colorbar limits.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve elements for
        """
        lower_limit = sg.InputText(
            default_text='',
            tooltip="Enter lower limit for colorbar.",
            size=self.getDimensions(name="axis_lower_limit_input_field"),
            key="-COLORBAR LOWER LIMIT-"
        )
        upper_limit = sg.InputText(
            default_text='',
            tooltip="Enter upper limit for colorbar.",
            size=self.getDimensions(name="axis_upper_limit_input_field"),
            key="-COLORBAR UPPER LIMIT-"
        )
        return lower_limit, upper_limit

    @storeElement
    def getTitleInputElement(self) -> sg.InputText:
        """
        Get element that allows user to input colorbar label.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        return sg.InputText(
            default_text='',
            tooltip="Enter label for colorbar.",
            size=self.getDimensions(name="colorbar_title_input_field"),
            key="-COLORBAR TITLE-"
        )

    @storeElement
    def getAutoscaleElement(self) -> sg.Checkbox:
        """
        Get element that allows user to determine whether colorbar is autoscaled.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        return sg.Checkbox(
            text='',
            tooltip="Choose boolean for colorbar."
                "When set True, colorbar will be autoscaled and limit inputs will be ignored."
                "When set False, limits inputs will be used if available.",
            default=True,
            size=self.getDimensions(name="autoscale_toggle_checkbox"),
            key="-COLORBAR AUTOSCALE-"
        )

    @storeElement
    def getScaleFactorInputElement(self) -> sg.Spin:
        """
        Get element that allows user to input colorbar scale factor.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        values = [f"1e{int(exponent):d}" for exponent in np.linspace(-24, 24, 49)]

        return sg.Spin(
            values=values,
            initial_value="1e0",
            tooltip="Choose scale factor for colorbar. Data is divided by this factor.",
            size=self.getDimensions(name="scale_factor_spin"),
            key="-COLORBAR SCALE FACTOR-"
        )

    @storeElement
    def getScaleTypeInputElement(self) -> sg.InputCombo:
        """
        Get element that allows user to choose axis scale.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        return sg.InputCombo(
            values=["Linear", "Logarithmic"],
            default_value="Linear",
            tooltip="Choose scale type for colorbar.",
            size=self.getDimensions(name="scale_type_combobox"),
            key="-COLORBAR SCALE TYPE-"
        )

    @storeElement
    def getColormapInputElement(self) -> sg.InputCombo:
        """
        Get elements that allows user to choose colormap.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        cmaps = plt.colormaps()
        default_cmap = "hsv"

        return sg.InputCombo(
            values=cmaps,
            default_value=default_cmap,
            tooltip="Choose colormap for colorbar.",
            size=self.getDimensions(name="colorbar_colormap_combobox"),
            key="-COLORBAR COLORMAP-"
        )

    @storeElement
    def getSegmentCountElement(self) -> sg.Spin:
        """
        Get element that allows user to input colorbar segment count.

        :param self: :class:`~Layout.SimulationWindow.ColorbarTab` to retrieve element from
        """
        values = [f"{int(hundred):d}00" for hundred in np.linspace(1, 9, 9)]

        return sg.Spin(
            values=values,
            initial_value=values[0],
            tooltip="Choose segment count for colorbar segments.",
            size=self.getDimensions(name="colorbar_segment_count_spin"),
            key="-COLORBAR SEGMENT COUNT-"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for tab.

        :param self: :class:`~Layout.SimulationWindow.AestheticsTab` to retrieve layout for
        """
        header_rows = self.getHeaderRows()
        row_elements = [
            self.getTitleInputElement(),
            *self.getLimitInputElements(),
            self.getAutoscaleElement(),
            self.getScaleFactorInputElement(),
            self.getColormapInputElement(),
            self.getSegmentCountElement()
        ]

        layout = Layout(rows=header_rows)
        layout.addRows(Row(window=self.getWindowObject(), elements=row_elements))
        return layout.getLayout()


class AxisTab(Tab, StoredObject):
    """
    This class contains the layout for the aesthetics tab in the simulation window.
        #. Header row to identify functions for input
        #. Axis name label for each axis to identify which axis input affects
        #. Input fields to set lower and upper limit for each axis. Two fields per axis
        #. Checkbox to choose whether each axis is autoscaled or manually determined
        #. Combobox to choose each axis scale type (e.g. linear, logarithmic)
        #. Spin to set scale factor for each axis
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve row for
        """
        window_object = self.getWindowObject()

        top_row = Row(window=window_object)
        texts = ["", "Limits", "Scale"]
        dimension_keys = [f"axis_header_row_{string:s}" for string in ["element", "limits", "scale"]]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="center"
            )
            top_row.addElements(text_element)

        bottom_row = Row(window=window_object)
        texts = ["Element", "Title", "Lower", "Upper", "Auto", "Factor", "Type"]
        dimension_keys = [
            f"axis_header_row_{string:s}"
            for string in
            ["element_name", "element_title", "lower_limit", "upper_limit", "autoscale", "scale_factor", "scale_type"]
        ]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="left"
            )
            bottom_row.addElements(text_element)

        return [top_row, bottom_row]

    @storeElement
    def getLimitInputElements(self, name: str) -> Tuple[sg.InputText, sg.InputText]:
        """
        Get elements that allow user to input axis limits.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve elements for
        :param name: name of axis
        """
        lower_limit = sg.InputText(
            default_text='',
            tooltip=f"Enter lower limit for {name:s}-axis",
            size=self.getDimensions(name="axis_lower_limit_input_field"),
            key=f"-LOWER LIMIT {name.upper():s}_AXIS-"
        )
        upper_limit = sg.InputText(
            default_text='',
            tooltip=f"Enter upper limit for {name:s}-axis",
            size=self.getDimensions(name="axis_upper_limit_input_field"),
            key=f"-UPPER LIMIT {name.upper():s}_AXIS-"
        )

        return lower_limit, upper_limit

    @storeElement
    def getTitleInputElement(self, name: str) -> sg.InputText:
        """
        Get element that allows user to input axis labels.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve element from
        :param name: name of axis
        """
        return sg.InputText(
            default_text='',
            tooltip=f"Enter label for {name:s}-axis",
            size=self.getDimensions(name="axis_row_title_input_field"),
            key=f"-TITLE {name.upper():s}_AXIS-"
        )

    @storeElement
    def getAutoscaleElement(self, name: str) -> sg.Checkbox:
        """
        Get element that allows user to determine whether axis is autoscaled.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve element from
        :param name: name of axis
        """
        return sg.Checkbox(
            text='',
            tooltip=f"Choose boolean for {name:s}-axis."
                f"When set True, {name:s}-axis will be autoscaled and limit inputs will be ignored."
                f"When set False, limits inputs will be used if available.",
            default=True,
            size=self.getDimensions(name="autoscale_toggle_checkbox"),
            key=f"-AUTOSCALE {name.upper():s}_AXIS-"
        )

    @storeElement
    def getScaleFactorInputElement(self, name: str) -> sg.Spin:
        """
        Get element that allows user to input scale factor.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve element from
        :param name: name of axis
        """
        values = [f"1e{int(exponent):d}" for exponent in np.linspace(-24, 24, 49)]

        return sg.Spin(
            values=values,
            initial_value="1e0",
            tooltip=f"Choose scale factor for {name:s}-axis. Data is divided by this factor.",
            size=self.getDimensions(name="scale_factor_spin"),
            key=f"-SCALE FACTOR {name.upper():s}_AXIS-"
        )

    @storeElement
    def getScaleTypeInputElement(self, name: str) -> sg.InputCombo:
        """
        Get element that allows user to choose axis scale.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve element from
        :param name: name of axis
        """
        return sg.InputCombo(
            values=["Linear", "Logarithmic"],
            default_value="Linear",
            tooltip=f"Choose scale type for {name:s}-axis",
            size=self.getDimensions(name="scale_type_combobox"),
            key=f"-SCALE TYPE {name.upper():s}_AXIS-"
        )

    @storeElement
    def getAxisLabelElement(self, name: str) -> sg.Text:
        """
        Get label to indicate which axis the row affects.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve element from
        :param name: name of axis
        """
        return sg.Text(
            text=name,
            size=self.getDimensions(name="axis_row_label")
        )

    def getInputRow(self, name: str, is_cartesian: bool = False) -> Row:
        """
        Get row that allows user input for a single axis.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve row for
        :param name: name of axis
        :param is_cartesian: set True if input is for a Cartesian axis.
            Set False otherwise.
        """
        name_label = self.getAxisLabelElement(name)
        title_input = self.getTitleInputElement(name)

        row = Row(window=self.getWindowObject(), elements=[name_label, title_input])
        if is_cartesian:
            lowerlimit_input, upperlimit_input = self.getLimitInputElements(name)
            autoscale_input = self.getAutoscaleElement(name)
            scale_factor_input = self.getScaleFactorInputElement(name)
            scale_type_input = self.getScaleTypeInputElement(name)
            row.addElements([lowerlimit_input, upperlimit_input, autoscale_input, scale_factor_input, scale_type_input])
        return row

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for tab.

        :param self: :class:`~Layout.SimulationWindow.AxisTab` to retrieve layout for
        """
        header_rows = self.getHeaderRows()
        axis_input_rows = [
            self.getInputRow("plot"),
            self.getInputRow('x', is_cartesian=True),
            self.getInputRow('y', is_cartesian=True),
            self.getInputRow('z', is_cartesian=True),
        ]

        layout = Layout(rows=header_rows)
        layout.addRows(axis_input_rows)
        return layout.getLayout()


class AestheticsTabGroup(TabGroup):
    """
    This class contains
        #. :class:`~Layout.SimulationWindow.AxisTab`
        #. :class:`~Layout.SimulationWindow.ColorbarTab`
    """

    def __init__(self, name: str, window: SimulationWindow):
        """
        Constructor for :class:`~Layout.SimulationWindow.AestheticTabGroup`.

        :param name: name of tabgroup
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tabgroup is stored in.
        """
        tabs = [
            AxisTab("Axis", window),
            ColorbarTab("Colorbar", window)
        ]
        super().__init__(tabs, name=name)


class PlottingTab(Tab):
    """
    This class contains the layout for the plotting tab in the simulation window.
        #. Header row to identify functions for input
        #. Axis name label for each axis to identify which axis input affects
        #. Combobox to input quantity species for each axis
        #. Combobox to input quantity for each axis
        #. Combobox to input transform type for each axis

    :ivar getPlotChoices: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotChoices`
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.PlottingTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in
        """
        super().__init__(name, window)

        self.getPlotChoices = window.getPlotChoices

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve row for
        """
        rows = [Row(window=self.getWindowObject()) for _ in range(2)]
        texts = ["Axis", "Species", "Quantity", "Condensor"]
        dimension_keys = [
            f"axis_header_row_{string:s}"
            for string in ["element_name", "quantity_species", "quantity_name", "condensor_name"]
        ]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="left"
            )
            rows[0].addElements(text_element)

        text_element = sg.Text(
            text="Transform",
            size=self.getDimensions(name="axis_header_row_transform_type"),
            justification="left"
        )
        rows[1].addElements(text_element)

        return rows

    @storeElement
    def getTransformElement(self) -> sg.InputCombo:
        """
        Get element to take user input for tranform.
        This allows user to choose which transform to perform on plot quantities.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve element from
        """
        transform_types = ["None", "Fourier"]

        return sg.InputCombo(
            values=transform_types,
            default_value=transform_types[0],
            tooltip="Choose transform to perform on plot quantities",
            enable_events=True,
            size=self.getDimensions(name="transform_type_combobox"),
            key=f"{cc_pre:s} TRANSFORM"
        )

    @storeElement
    def getAxisCondensorElement(self, name: str) -> sg.InputCombo:
        """
        Get element to take user input for an axis condensor.
        This allows user to choose which type of condensor to calculate for a plot quantity (e.g. frequency, amplitude).

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve element from
        :param name: name of axis to retrieve element from
        """
        axis_condensor = ["None", "Mean", "Frequency", "Standard Deviation"]

        return sg.InputCombo(
            values=axis_condensor,
            default_value=axis_condensor[0],
            tooltip=f"Choose condensor for quantity on {name:s}-axis of plot",
            enable_events=True,
            size=self.getDimensions(name="axis_condensor_combobox"),
            key=f"-{cc_pre:s} CONDENSOR {name.upper():s}_AXIS-"
        )

    @storeElement
    def getAxisQuantityElement(self, name: str, **kwargs) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity.
        This allows user to choose which quantity to plot on the axis.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve element from
        :param name: name of axis to retrieve element from
        :param kwargs: additional arguments to pass into :class:`~PySimpleGUI.InputCombo`
        """
        sg_kwargs = {
            "tooltip": f"Choose quantity to display on {name:s}-axis of plot",
            "enable_events": True,
            "size": self.getDimensions(name="axis_quantity_combobox"),
            "key": f"-{cc_pre:s} QUANTITY {name.upper():s}_AXIS-"
        }
        sg_kwargs.update(kwargs)

        if "values" not in sg_kwargs:
            sg_kwargs["values"] = self.getPlotChoices(species="Variable")
        if "default_value" not in sg_kwargs:
            sg_kwargs["default_value"] = sg_kwargs["values"][0]
        elem = sg.InputCombo(**sg_kwargs)
        return elem

    @storeElement
    def getAxisQuantitySpeciesElement(
        self, 
        name: str, 
        include_none: bool = False
    ) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity type.
        This allows user to choose which type of quantity to plot on the axis.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve element from
        :param name: name of axis to retrieve element from
        :param include_none: set True to include "None" as choice for quantity type.
            Set False otherwise.
        """
        axis_quantity_species = []
        if include_none:
            axis_quantity_species.append("None")
        axis_quantity_species.extend(["Variable", "Function"])
        if len(self.getPlotChoices(species="Parameter")) >= 1:
            axis_quantity_species.append("Parameter")

        return sg.InputCombo(
            values=axis_quantity_species,
            default_value=axis_quantity_species[0],
            tooltip=f"Choose quantity type to display on {name:s}-axis of plot",
            enable_events=True,
            size=self.getDimensions(name="axis_quantity_species_combobox"),
            key=f"-{ccs_pre:s} {name.upper():s}_AXIS-"
        )

    def getAxisLabelElement(self, name: str) -> sg.Text:
        """
        Get label to indicate which axis the row affects.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve element from
        :param name: name of axis
        """
        return sg.Text(
            text=name,
            size=self.getDimensions(name="axis_row_label")
        )

    def getAxisInputRow(
        self, 
        name: str, 
        include_none: bool = True
    ) -> Row:
        """
        Get row that allows user input for a single axis.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve row for
        :param name: name of axis
        :param include_none: set True to include "None" as choice for quantity specie.
            Set False to exclude "None" as a choice.
        """
        row = Row(window=self.getWindowObject())

        name_label = self.getAxisLabelElement(name)
        axis_specie_combobox = self.getAxisQuantitySpeciesElement(name, include_none=include_none)
        row.addElements([name_label, axis_specie_combobox])

        if include_none:
            axis_quantity_combobox = self.getAxisQuantityElement(name, values=[''], disabled=True)
        else:
            axis_quantity_combobox = self.getAxisQuantityElement(name)
        row.addElements(axis_quantity_combobox)

        axis_condensor_combobox = self.getAxisCondensorElement(name)
        row.addElements(axis_condensor_combobox)

        return row

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for plotting tab.

        :param self: :class:`~Layout.SimulationWindow.PlottingTab` to retrieve layout from
        """
        header_rows = self.getHeaderRows()
        axis_input_rows = [
            self.getAxisInputRow('x', include_none=False),
            self.getAxisInputRow('y', include_none=False),
            self.getAxisInputRow('z', include_none=True),
            self.getAxisInputRow('c', include_none=True)
        ]
        transform_combobox = self.getTransformElement()

        layout = Layout()
        layout.addRows(header_rows[0])
        layout.addRows(axis_input_rows)
        layout.addRows(header_rows[1])
        layout.addRows(Row(window=self.getWindowObject(), elements=transform_combobox))
        return layout.getLayout()


class AnalysisTabGroup(TabGroup):
    """
    This class contains the layout for the analysis tabgroup in the simulation window.
        #. :class:`~Layout.SimulationWindow.FrequencyTab`
        #. :class:`~Layout.SimulationWindow.MeanTab`
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AnalysisTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab group is stored in.
        """
        tabs = [
            FrequencyTab("Frequency", window),
            MeanTab("Holder Mean", window)
        ]
        super().__init__(tabs, name=name)


class FrequencyTab(Tab, StoredObject):
    """
    This class contains the layout for the frequency tab in the analysis tabgroup.
        #. Header row
        #. Combobox to choose calculation method
        #. Combobox to choose condensor method
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.FrequencyTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRow(self) -> Row:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.MeanTab` to retrieve row for
        """
        row = Row(window=self.getWindowObject())
        texts = ["Method", "Condensor"]
        dimension_keys = [f"frequency_header_row_{string:s}" for string in ["method_name", "condensor_name"]]

        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="left"
            )
            row.addElements(text_element)
        return row

    @storeElement
    def getMethodElement(self) -> sg.InputCombo:
        """
        Get element to take user input for calculation method.
        This allows user to choose which method to calculate frequency with.

        :param self: :class:`~Layout.SimulationWindow.FrequencyTab` to retrieve element from
        """
        frequency_method = [
            "Separation of Maxima",
            "Separation of Minima",
            "Separation of Extrema",
            "Separation of Fourier Maxima",
            "Argument of Autocorrelation Maximum"
        ]
        
        return sg.InputCombo(
            values=frequency_method,
            default_value=frequency_method[0],
            tooltip="Choose method to calculate frequency",
            enable_events=True,
            size=self.getDimensions(name="frequency_method_combobox"),
            key=f"-ANALYSIS METHOD FREQUENCY-"
        )

    @storeElement
    def getCondensorElement(self) -> sg.InputCombo:
        """
        Get element to take user input for condensor.
        This allows user to choose which condensor to reduce frequency with.

        :param self: :class:`~Layout.SimulationWindow.FrequencyTab` to retrieve element from
        """
        axis_condensor = ["Average", "Maximum", "Minimum", "Initial", "Final"]

        return sg.InputCombo(
            values=axis_condensor,
            default_value=axis_condensor[0],
            tooltip="Choose method to condense frequency",
            enable_events=True,
            size=self.getDimensions(name="frequency_condensor_combobox"),
            key="ANALYSIS CONDENSOR FREQUENCY"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for frequency tab.

        :param self: :class:`~Layout.SimulationWindow.FrequencyTab` to retrieve layout from
        """
        header_row = self.getHeaderRow()
        method_combobox = self.getMethodElement()
        condensor_combobox = self.getCondensorElement()

        layout = Layout()
        layout.addRows(header_row)
        layout.addRows(Row(window=self.getWindowObject(), elements=[method_combobox, condensor_combobox]))
        return layout.getLayout()


class MeanTab(Tab, StoredObject):
    """
    This class contains the layout for the mean tab in the analysis tabgroup.
        #. Header row
        #. Spinner to select order for Holder mean
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.MeanTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRow(self) -> Row:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.MeanTab` to retrieve row for
        """
        row = Row(window=self.getWindowObject())
        texts = ["Order"]
        dimension_keys = [f"mean_header_row_{string:s}" for string in ["order"]]
        
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="left"
            )
            row.addElements(text_element)
        return row

    @storeElement
    def getOrderElement(self) -> sg.Spin:
        """
        Get element to take user input for order.
        This allows user to choose order of Holder mean.

        :param self: :class:`~Layout.SimulationWindow.MeanTab` to retrieve element from
        """
        values = ['-inf', '-1', '0', '1', '2', 'inf']

        return sg.Spin(
            values=values,
            initial_value='1',
            size=self.getDimensions(name="mean_order_spin"),
            key=f"-ANALYSIS ORDER MEAN-"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for frequency tab.

        :param self: :class:`~Layout.SimulationWindow.FrequencyTab` to retrieve layout from
        """
        header_row = self.getHeaderRow()
        order_spin = self.getOrderElement()

        layout = Layout()
        layout.addRows(header_row)
        layout.addRows(Row(window=self.getWindowObject(), elements=order_spin))
        return layout.getLayout()


class SimulationWindow(TabbedWindow):
    """
    This class contains the layout for the simulation window.
        #. Simulation tab. This tab allows the user to run the simulation and display desired results.
        #. Aesthetics tab. This tab allows the user to set plot aesthetic for the displayed figure.
        #. Analysis tabgroup. This tabgroup allows the user to determine how to execute analysis for plot.
    
    :ivar plot_choices: name(s) of variable(s) and/or function(s) that the user may choose to plot
    :ivar free_parameters: name(s) of parameter(s) that the user may choose multiple values for in model
    """

    def __init__(
            self,
            name: str,
            runner: SimulationWindowRunner,
            free_parameter_values: Dict[str, Tuple[float, float, int, Quantity]],
            plot_choices: Dict[str, List[str]]
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.SimulationWindow`.
        
        :param name: title of window
        :param runner: :class:`~Layout.SimulationWindow.SimulationWindowRunner` that window is stored in
        :param free_parameter_values: dictionary of free-parameter values.
            Key is name of free parameter.
            Value is tuple of (minimum, maximum, stepcount, Quantity) for free parameter.
            Leave as empty dictionary if there exist zero free parameters.
        :param plot_choices: collection of quantities that may be plotted along each axis.
        """
        dimensions = {
            "window": getDimensions(
                ["simulation_window", "window"]
            ),
            "parameter_slider_name_label": getDimensions(
                ["simulation_window", "parameter_slider", "name_label"]
            ),
            "parameter_slider_minimum_label": getDimensions(
                ["simulation_window", "parameter_slider", "minimum_label"]
            ),
            "parameter_slider_maximum_label": getDimensions(
                ["simulation_window", "parameter_slider", "maximum_label"]
            ),
            "parameter_slider_stepcount_label": getDimensions(
                ["simulation_window", "parameter_slider", "stepcount_label"]
            ),
            "parameter_slider_slider": getDimensions(
                ["simulation_window", "parameter_slider", "slider"]
            ),
            "initial_time_input_field": getDimensions(
                ["simulation_window", "simulation_tab", "initial_time_input_field"]
            ),
            "timestep_count_input_field": getDimensions(
                ["simulation_window", "simulation_tab", "final_time_input_field"]
            ),
            "final_time_input_field": getDimensions(
                ["simulation_window", "simulation_tab", "time_stepcount_input_field"]
            ),
            "axis_header_row_element": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "top_header_row", "element"]
            ),
            "axis_header_row_limits": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "top_header_row", "limits"]
            ),
            "axis_header_row_scale": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "top_header_row", "scale"]
            ),
            "axis_header_row_element_name": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "element_name"]
            ),
            "axis_header_row_element_title": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "element_title"]
            ),
            "axis_header_row_lower_limit": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "lower_limit"]
            ),
            "axis_header_row_upper_limit": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "upper_limit"]
            ),
            "axis_header_row_autoscale": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "autoscale"]
            ),
            "axis_header_row_scale_factor": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "scale_factor"]
            ),
            "axis_header_row_scale_type": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "bottom_header_row", "scale_type"]
            ),
            "axis_row_label": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "name_label"]
            ),
            "axis_lower_limit_input_field": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "lower_limit_input_field"]
            ),
            "axis_upper_limit_input_field": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "upper_limit_input_field"]
            ),
            "axis_row_title_input_field": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "title_input_field"]
            ),
            "autoscale_toggle_checkbox": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "autoscale_toggle_checkbox"]
            ),
            "scale_factor_spin": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "scale_factor_spin"]
            ),
            "scale_type_combobox": getDimensions(
                ["simulation_window", "aesthetics_tab", "axis_tab", "axis_row", "scale_type_combobox"]
            ),
            "colorbar_title_input_field": getDimensions(
                ["simulation_window", "aesthetics_tab", "colorbar_tab", "title_input_field"]
            ),
            "colorbar_colormap_combobox": getDimensions(
                ["simulation_window", "aesthetics_tab", "colorbar_tab", "colormap_combobox"]
            ),
            "colorbar_segment_count_spin": getDimensions(
                ["simulation_window", "aesthetics_tab", "colorbar_tab", "segment_count_spin"]
            ),
            "axis_header_row_quantity_species": getDimensions(
                ["simulation_window", "plotting_tab", "header_row", "quantity_species"]
            ),
            "axis_header_row_quantity_name": getDimensions(
                ["simulation_window", "plotting_tab", "header_row", "quantity_name"]
            ),
            "axis_header_row_transform_type": getDimensions(
                ["simulation_window", "plotting_tab", "header_row", "transform_type"]
            ),
            "axis_header_row_condensor_name": getDimensions(
                ["simulation_window", "plotting_tab", "header_row", "condensor_name"]
            ),
            "axis_quantity_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_quantity_combobox"]
            ),
            "axis_quantity_species_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_quantity_species_combobox"]
            ),
            "axis_condensor_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_condensor_combobox"]
            ),
            "transform_type_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "transform_type_combobox"]
            ),
            "frequency_header_row_method_name": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "header_row", "method_name"]
            ),
            "frequency_header_row_condensor_name": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "header_row", "condensor_name"]
            ),
            "frequency_method_combobox": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "method_combobox"]
            ),
            "frequency_condensor_combobox": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "condensor_combobox"]
            ),
            "mean_header_row_order": getDimensions(
                ["simulation_window", "analysis_tab", "mean_tab", "header_row", "order"]
            ),
            "mean_order_spin": getDimensions(["simulation_window", "analysis_tab", "mean_tab", "order_spin"])
        }
        super().__init__(name, runner, dimensions=dimensions)
        self.plot_choices = plot_choices
        self.free_parameter_values = free_parameter_values

        simulation_tab_name = "Simulation"
        plotting_tab_name = "Plotting"
        simulation_tab = SimulationTab(simulation_tab_name, self)
        plotting_tab = PlottingTab(plotting_tab_name, self)
        self.getSimulationTab = partial(self.getTabs, names=simulation_tab_name)
        self.getPlottingTab = partial(self.getTabs, names=plotting_tab_name)

        self.addTabs(simulation_tab)
        self.addTabs(AestheticsTabGroup("Aesthetics", self).getAsTab())
        self.addTabs(plotting_tab)
        self.addTabs(AnalysisTabGroup("Analysis", self).getAsTab())

    def getFreeParameterNames(
        self, 
        indicies: Union[int, List[int]] = None
    ) -> Union[str, List[str]]:
        """
        Get stored name(s) for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve names from
        :param indicies: location to retrieve parameter name from in collection of names
        """

        free_parameter_names = list(self.free_parameter_values.keys())

        def get(index: int) -> str:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindow.getFreeParameterNames`"""
            return free_parameter_names[index]

        return recursiveMethod(
            args=indicies,
            base_method=get,
            valid_input_types=int,
            output_type=list,
            default_args=range(len(free_parameter_names))
        )

    def getFreeParameterValues(
            self,
            name: str = None
    ) -> Union[Dict[str, Tuple[float, float, int, Quantity]], Tuple[float, float, int, Quantity]]:
        """
        Get stored values for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve names from
        :param name: name of parameter to retrieve values of
        :returns: Tuple of (minimum, maximum, stepcount) if name is str.
            Dict of equivalent tuples if name is None; all parameter tuples are returned.
        """
        if isinstance(name, str):
            free_parameter_values = self.getFreeParameterValues()
            return free_parameter_values[name]
        elif name is None:
            return self.free_parameter_values
        else:
            raise TypeError("name must be str")

    def getPlotChoices(
        self, 
        species: Union[str, List[str]] = None
    ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get stored names for quantities the user may choose to plot.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve choices from
        :param species: specie(s) of quantities (e.g. "Variable", "Function", "Parameter")
        """
        if isinstance(species, str):
            plot_choices = self.getPlotChoices()
            return plot_choices[species]
        elif isinstance(species, list):
            plot_choices = []
            extend_choices = plot_choices.extend
            for specie in species:
                extend_choices(self.getPlotChoices(species=specie))
            return unique(plot_choices)
        elif species is None:
            return self.plot_choices
        else:
            raise TypeError("species must by str or list")

    @storeElement
    def getMenu(self) -> sg.Menu:
        """
        Get menu displayed in top toolbar of window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve menu from
        """
        free_parameter_name_entries = [name + "::Save Animated Figure" for name in self.getFreeParameterNames()]
        menu_definition = [
            [
                "Save",
                [
                    "Model",
                    [
                        "Results::Save",
                        "Parameters::Save",
                        "Functions::Save",
                        "Time-Evolution Types::Save"
                    ],
                    "Figure",
                    [
                        "Static::Save Figure",
                        "Animated",
                        free_parameter_name_entries
                    ]
                ]
            ]
        ]
        
        return sg.Menu(
            menu_definition=menu_definition,
            key="-TOOLBAR MENU-"
        )

    @storeElement
    def getCanvas(self) -> sg.Canvas:
        """
        Get canvas where figures will be plotted.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve canvas from
        """
        return sg.Canvas(
            key="-CANVAS-"
        )

    def getParameterSliderRows(self) -> List[Row]:
        """
        Get all parameter slider objects for window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve sliders from
        """
        slider_rows = []
        for free_parameter_name, free_parameter_value in self.getFreeParameterValues().items():
            default_unit = free_parameter_value[3].units
            unit_conversion_factor = getUnitConversionFactor(default_unit)
            minimum_value = float(free_parameter_value[0]) * unit_conversion_factor
            maximum_value = float(free_parameter_value[1]) * unit_conversion_factor
            value_stepcount = int(free_parameter_value[2])
            parameter_slider = ParameterSlider(
                name=free_parameter_name,
                window=self,
                values=(minimum_value, maximum_value, value_stepcount, default_unit)
            )
            slider_rows.append(Row(window=self, elements=parameter_slider.getElement()))
        
        return slider_rows

    @storeElement
    def getUpdatePlotButton(self) -> sg.Button:
        """
        Get button that allows user to update the plot with new aesthetics.

        :param self: :class:`~Layout.SimulationWindow.AestheticsTab` to retrieve element from
        """
        text = "Update Plot"
        
        return sg.Button(
            button_text=text,
            tooltip="Click to update plot with new axis settings.",
            key=f"{text.upper():s}"
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for simulation window.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve layout from
        """
        menu = self.getMenu()
        canvas = self.getCanvas()
        exit_button = sg.Exit()
        update_plot_button = self.getUpdatePlotButton()
        tabgroup = TabGroup(self.getTabs()).getTabGroup()
        slider_rows = self.getParameterSliderRows()

        prefix_layout = Layout(rows=Row(window=self, elements=menu))

        left_layout = Layout()
        left_layout.addRows(Row(window=self, elements=tabgroup))
        left_layout.addRows(slider_rows)
        left_layout.addRows(Row(window=self, elements=[update_plot_button, exit_button]))

        right_layout = Layout()
        right_layout.addRows(Row(window=self, elements=canvas))
        # noinspection PyTypeChecker
        return prefix_layout.getLayout() + [[sg.Column(left_layout.getLayout()), sg.Column(right_layout.getLayout())]]


class SimulationWindowRunner(WindowRunner):
    """
    This class runs the simulation and displays results.
    This window allows the user to...
        #. Choose which variable/function to plot on each x-axis and y-axis
        #. Choose which value of free parameters to assume for present plot
    
    :ivar values: present values for all elements in window
    :ivar figure_canvas: object storing (1) canvas on which figure is plotted and (2) figure containing plot data
    :ivar model: :class:`~Function.Model` to simulate
    :ivar general_derivative_vector: partially-simplified, symbolic derivative vector.
        Simplified as much as possible, except leave free parameters and variables as symbolic.
    :ivar results: object to store results from most recent simulation.
        This attribute greatly speeds up grabbing previously-calculated results.

    :ivar getPlotChoices: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotChoices`
    :ivar getFreeParameterNames: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getFreeParameterNames`
    :ivar resetResults: pointer to :meth:`~Results.Results.resetResults`
    :ivar setResults: pointer to :meth:`~Results.Results.setResults`
    """

    def __init__(self, name: str, model: Model = None, results: Results = None, **kwargs) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param name: title of window
        :param model: :class:`~Function.Model` to simulate.
            Only called and required
            if :paramref:`~Layout.SimulationWindow.SimulationWindowRunner.__init__.results` is None.
        :param results: initial results to start simulation with
        :param **kwargs: additional arguments to pass into :class:`~Layout.SimulationWindow.SimulationWindow`
        """
        window_object = SimulationWindow(name, self, **kwargs)
        super().__init__(window_object)
        window_object.getWindow()

        self.getPlotChoices = window_object.getPlotChoices
        self.getFreeParameterNames = window_object.getFreeParameterNames

        self.axis_names = ['c', 'x', 'y', 'z']
        self.timelike_species = ["Variable", "Function"]
        self.parameterlike_species = ["Parameter"]
        self.values = None
        self.figure_canvas = None
        self.general_derivative_vector = None

        if results is None:
            self.model = model
            free_parameter_values = {
                free_parameter_name: self.getFreeParameterValues(free_parameter_name)
                for free_parameter_name in self.getFreeParameterNames()
            }
            results_object = Results(self.getModel(), free_parameter_values)
        else:
            self.model = results.getModel()
            results_object = results

        self.results = results_object
        self.resetResults = results_object.resetResults
        self.setResults = results_object.setResults

    def getModel(self) -> Model:
        """
        Get model to simulate.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        return self.model

    def getGeneralDerivativeVector(self) -> function:
        """
        Get function handle for derivative vector.
        Solve for and save derivative vector if it has not been called previously.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve derivative vector from
        """
        general_derivative_vector = self.general_derivative_vector
        if general_derivative_vector is None:
            model = self.getModel()
            temporal_variable_names = model.getVariables(
                time_evolution_types="Temporal",
                return_type=str
            )

            self.general_derivative_vector, equilibrium_substitutions = model.getDerivativeVector(
                names=temporal_variable_names,
                skip_parameters=self.getFreeParameterNames()
            )
            self.results.setEquilibriumExpressions(equilibrium_expressions=equilibrium_substitutions)
        
        return self.general_derivative_vector

    def getResultsObject(self) -> Results:
        """
        Get stored :class:`~Results.Results`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve object from
        """
        return self.results

    def getAxisNames(self) -> List[str]:
        """
        Get names for axes.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve names from
        :returns: List of axis names (e.g. ['x', 'y', 'c'])
        """
        return self.axis_names

    def getFreeParameterIndex(self, name: str) -> int:
        """
        Get index of free parameter in collection of free-parameter names.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve free parameter names from
        :param name: name of free parameter to retrieve index of
        """
        free_parameter_names = self.getFreeParameterNames()
        index = free_parameter_names.index(name)
        return index

    def getFreeParameterValues(self, name: str) -> ndarray:
        """
        Get possible values for free-parameter slider.
        This corresponds to the values the parameter is simulated at.
        Uses present state of :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve values from
        :param name: name of parameter to retrieve values
        """
        slider_min = self.getSliderMinimum(name)
        slider_max = self.getSliderMaximum(name)
        slider_resolution = self.getSliderResolution(name)
        step_count = round((slider_max - slider_min) / slider_resolution + 1)
        return np.linspace(slider_min, slider_max, step_count)

    def getClosestSliderIndex(
        self, 
        names: Union[str, List[str]] = None
    ) -> Union[int, Tuple[int, ...]]:
        """
        Get location/index of slider closest to value of free parameter.
        Location is discretized from zero to the number of free-parameter values.
        Uses present state of :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve slider from.
        :param names: name(s) of parameter(s) associated with slider.
            Defaults to names of all free parameters.
        :returns: Slider index for free parameter if names is str.
            Tuple of slider indicies for all given free parameters if names is list or tuple.
        """

        def get(name: str) -> int:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindowRunner.getClosestSliderIndex`"""
            slider_value = self.getSliderValue(name)
            free_parameter_values = self.getFreeParameterValues(name)
            closest_index = min(
                range(len(free_parameter_values)),
                key=lambda i: abs(free_parameter_values[i] - slider_value)
            )
            return closest_index

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=tuple,
            default_args=self.getFreeParameterNames()
        )

    def getLikeSpecies(self, like: str):
        """
        Get collection of quantity species that may be treated as over-time.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve species from
        :param like: species type to retrieve collection of species of.
            Can be "timelike" or "parameterlike".
        """
        if like == "timelike":
            return self.timelike_species
        elif like == "parameterlike":
            return self.parameterlike_species
        else:
            raise ValueError("like must be 'timelike' or 'parameterlike'")

    def getAxisQuantity(self, axis_name: str) -> Tuple[str, str, str]:
        """
        Get selected quantity name, specie, and condensor for desired axis.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve quantity name from
        :param axis_name: name of axis to retrieve quantity name from
        :returns: Tuple of quantity info.
            First index gives quantity name.
            Second index gives quantity type.
            Third index gives condensor name.
        """

        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()
        # noinspection PyTypeChecker
        plotting_tab: PlottingTab = window_object.getPlottingTab()

        quantity_name_key = getKeys(plotting_tab.getAxisQuantityElement(axis_name))
        quantity_name = self.getValue(quantity_name_key)
        specie_key = getKeys(plotting_tab.getAxisQuantitySpeciesElement(axis_name))
        specie = self.getValue(specie_key)

        condensor_key = getKeys(plotting_tab.getAxisCondensorElement(axis_name))
        try:
            condensor = self.getValue(condensor_key)
        except KeyError:
            condensor = "None"

        quantity = (quantity_name, specie, condensor)
        return quantity

    def getFrequencyMethod(self) -> str:
        """
        Get selected method to calculate frequency.
        Uses present state of window runner.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve method from
        """
        # noinspection PyTypeChecker
        frequency_tab: FrequencyTab = FrequencyTab.getInstances()[0]
        method_key = getKeys(frequency_tab.getMethodElement())
        method = self.getValue(method_key)
        return method

    def getFrequencyCondensor(self) -> str:
        """
        Get selected condensor to calculate frequency.
        Uses present state of window runner.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve condensor from
        """
        # noinspection PyTypeChecker
        frequency_tab: FrequencyTab = FrequencyTab.getInstances()[0]
        condensor_key = getKeys(frequency_tab.getCondensorElement())
        condensor = self.getValue(condensor_key)
        return condensor

    def getMeanOrder(self) -> float:
        """
        Get selected order for Holder mean.
        Uses present state of window runner.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve order from
        """
        # noinspection PyTypeChecker
        mean_tab: MeanTab = MeanTab.getInstances()[0]
        order_key = getKeys(mean_tab.getOrderElement())
        order = float(self.getValue(order_key))
        return order

    def getInputTimes(self) -> Tuple[float, float, int]:
        """
        Get info about time steps to run simulation.
        Uses present state of :class:`~Layout.SimulationWindow.SimulationWindowRunner` to determine info.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve time info from
        :returns: Tuple of time info.
            First element is initial time for simulation.
            Second element is final time for simulation.
            Third element is number of time steps for simulation.
        """
        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()
        # noinspection PyTypeChecker
        simulation_tab_object: SimulationTab = window_object.getSimulationTab()

        initial_time_key = getKeys(simulation_tab_object.getInitialTimeInputElement())
        final_time_key = getKeys(simulation_tab_object.getFinalTimeInputElement())
        timestep_count_key = getKeys(simulation_tab_object.getTimeStepCountInputElement())

        initial_time = float(eval(self.getValue(initial_time_key)))
        timestep_count = int(eval(self.getValue(timestep_count_key)))
        final_time = float(eval(self.getValue(final_time_key)))
        return initial_time, final_time, timestep_count

    def getSliderValue(self, name: str) -> float:
        """
        Get present value of parameter slider.
        Uses present state of :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve slider from
        :param name: name of parameter associated with slider
        """
        slider_key = getKeys(ParameterSlider.getInstances(names=name).getSlider())
        slider_value = self.getValue(slider_key)
        return slider_value

    @staticmethod
    def getSliderMinimum(name: str) -> float:
        """
        Get minimum value of parameter slider.
        Uses present state of window.

        :param name: name of parameter associated with slider
        """
        slider = ParameterSlider.getInstances(names=name).getSlider()
        slider_minimum = vars(slider)["Range"][0]
        return slider_minimum

    @staticmethod
    def getSliderMaximum(name: str) -> float:
        """
        Get maximum value of parameter slider.
        Uses present state of window.

        :param name: name of parameter associated with slider
        """
        slider = ParameterSlider.getInstances(names=name).getSlider()
        slider_max = vars(slider)["Range"][1]
        return slider_max

    @staticmethod
    def getSliderResolution(name: str) -> float:
        """
        Get resolution of parameter slider.

        :param name: name of parameter associated with slider
        """
        slider = ParameterSlider.getInstances(names=name).getSlider()
        slider_resolution = vars(slider)["Resolution"]
        return slider_resolution

    def runWindow(self) -> None:
        """
        Run simulation window.
        This function links each possible event to an action.
        When an event is triggered, its corresponding action is executed.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to run
        """
        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()
        window = window_object.getWindow()

        # noinspection PyTypeChecker
        simulation_tab_object: SimulationTab = window_object.getSimulationTab()
        run_simulation_key = getKeys(simulation_tab_object.getRunButton())

        toolbar_menu_key = getKeys(window_object.getMenu())
        update_plot_key = getKeys(window_object.getUpdatePlotButton())

        while True:
            event, self.values = window.read()
            print(event)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            menu_value = self.getValue(toolbar_menu_key)

            if menu_value is not None:
                if menu_value == "Parameters::Save":
                    self.saveModelParameters()
                elif menu_value == "Functions::Save":
                    self.saveModelFunctions()
                elif menu_value == "Results::Save":
                    self.saveResults()
                elif menu_value == "Time-Evolution Types::Save":
                    self.saveModelTimeEvolutionTypes()
                elif menu_value == "Static::Save Figure":
                    self.saveFigure()
                elif "Save Animated Figure" in menu_value:
                    free_parameter_name = menu_value.split('::')[0]
                    self.saveFigure(free_parameter_name)
            elif event == run_simulation_key:
                self.runSimulations()
            elif fps_pre in event:
                self.updatePlot()
            elif cc_pre in event:
                if ccs_pre in event:
                    axis_name = event.split(' ')[-1].replace("_AXIS", '').replace('-', '').lower()
                    self.updatePlotChoices(axis_name)
                else:
                    window.write_event_value(update_plot_key, None)
            elif event == update_plot_key:
                self.updatePlot()
        window.close()

    @staticmethod
    def updateProgressMeter(title: str, current_value: int, max_value: int) -> sg.OneLineProgressMeter:
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
            max_value=max_value
        )

    def runSimulation(
            self,
            index: Union[tuple, Tuple[int]],
            parameter_values: Dict[str, float] = None,
            variable_names: List[str] = None,
            general_derivative_vector: List[Expr] = None,
            y0: ndarray = None,
            times: ndarray = None,
            model: Model = None
    ) -> None:
        """
        Run simulation for a single set of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        :param index: index for free-parameter values.
            Results are saved at this index.
        :param parameter_values: dictionary of free-parameter values.
            Key is name of each parameter.
            Value is value of corresponding parameter.
            Defaults to empty dictionary.
        :param variable_names: names of temporal variables in model.
            This gives the order for arguments in the lambdified derivative vector.
        :param general_derivative_vector: partially-simplified, symbolic derivative vector.
            Simplified as much as possible, except leave free parameters and variables as symbolic.
        :param y0: initial condition vector for derivative vector
        :param times: vector of time steps to solve ODE at
        :param model: :class:`~Function.Model` to run simulation for
        """
        if any(element is None for element in [variable_names, general_derivative_vector, y0, times]):
            if model is None:
                model = self.getModel()
            if variable_names is None:
                variable_names = model.getVariables(
                    time_evolution_types="Temporal", return_type=str
                )
            if general_derivative_vector is None:
                general_derivative_vector = self.getGeneralDerivativeVector()
            if y0 is None:
                y0 = model.getInitialValues(names=variable_names)
            if times is None:
                times = np.linspace(*self.getInputTimes())
        if parameter_values is None:
            parameter_values = {}

        derivatives = [derivative.subs(parameter_values) for derivative in general_derivative_vector]
        ydot = lambdify((Symbol('t'), tuple(variable_names)), derivatives, modules=["math"])
        solution = solveODE(ydot, y0, times)
        # noinspection PyTypeChecker
        results = formatResultsAsDictionary(*solution, variable_names)
        self.setResults(index=index, results=results)

    def runSimulations(self) -> None:
        """
        Run simulations for all possible combinations of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        """
        self.resetResults()
        free_parameter_names = self.getFreeParameterNames()
        model = self.getModel()
        variable_names = model.getVariables(time_evolution_types="Temporal", return_type=str)
        kwargs = {
            "model": model,
            "variable_names": variable_names,
            "general_derivative_vector": self.getGeneralDerivativeVector(),
            "y0": model.getInitialValues(names=variable_names, return_type=list),
            "times": np.linspace(*self.getInputTimes())
        }

        if (parameter_count := len(free_parameter_names)) == 0:
            self.runSimulation((), **kwargs)
        elif parameter_count >= 1:
            free_parameter_values = {}
            free_parameter_indicies = []
            for parameter_name in free_parameter_names:
                free_parameter_value = self.getFreeParameterValues(parameter_name)
                free_parameter_values[parameter_name] = free_parameter_value
                free_parameter_indicies.append(range(len(free_parameter_value)))

            free_parameter_index_combos = tuple(product(*free_parameter_indicies))
            total_combo_count = len(free_parameter_index_combos)
            for simulation_number, indicies in enumerate(free_parameter_index_combos):
                if not self.updateProgressMeter("Running Simulations", simulation_number, total_combo_count):
                    break
                parameter_simulation_values = {
                    (free_parameter_name := free_parameter_names[free_parameter_index]):
                        free_parameter_values[free_parameter_name][indicies[free_parameter_index]]
                    for free_parameter_index in range(parameter_count)
                }
                self.runSimulation(indicies, parameter_simulation_values, **kwargs)
            self.updateProgressMeter("Running Simulations", total_combo_count, total_combo_count)
        self.updatePlot()

    def getPlotAesthetics(self) -> Dict[str, Optional[Union[str, float]]]:
        """
        Get plot-aesthetics input by user.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve aesthetic inputs from
        """

        def getValues(keys: Union[str, Iterable[str]]) -> Optional[Union[str, float, bool, tuple]]:
            """
            Get value for plot aesthetic from element key.
            
            :param keys: key(s) of element(s) to retrieve value from
            """

            def get(key: str) -> Optional[Union[str, float, bool]]:
                """Base method for :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotAesthetics.getValues`"""
                try:
                    value = self.getValue(key)
                    if isinstance(value, bool):
                        return value
                    elif isinstance(value, str):
                        try:
                            return float(value)
                        except ValueError:
                            return None if len(value) == 0 else value
                    return value
                except KeyError:
                    return None

            return recursiveMethod(
                args=keys,
                base_method=get,
                valid_input_types=str,
                output_type=tuple
            )

        cartesian_names = ('x', 'y', 'z')
        scale_type_dict = {
            "Linear": "linear",
            "Logarithmic": "log"
        }

        # noinspection PyTypeChecker
        axis_tab: AxisTab = AxisTab.getInstances()[0]
        # noinspection PyTypeChecker
        colorbar_tab: ColorbarTab = ColorbarTab.getInstances()[0]

        scale_factor = {}
        autoscale = {}
        lim = {}
        label = {}
        scale_type = {}
        for axis_name in cartesian_names:
            scale_factor[axis_name] = getValues(getKeys(axis_tab.getScaleFactorInputElement(axis_name)))
            label[axis_name] = getValues(getKeys(axis_tab.getTitleInputElement(axis_name)))
            scale_type[axis_name] = scale_type_dict[getValues(getKeys(axis_tab.getScaleTypeInputElement('x')))]

            autoscale[axis_name] = getValues(getKeys(axis_tab.getAutoscaleElement(axis_name)))
            if autoscale[axis_name]:
                lim[axis_name] = (None, None)
            else:
                lim[axis_name] = getValues(getKeys(axis_tab.getLimitInputElements(axis_name)))

        autoscale['c'] = getValues(getKeys(colorbar_tab.getAutoscaleElement()))
        if autoscale['c']:
            lim['c'] = (None, None)
        else:
            lim['c'] = getValues(getKeys(colorbar_tab.getLimitInputElements()))
        scale_factor['c'] = getValues(getKeys(colorbar_tab.getScaleFactorInputElement()))

        aesthetics_kwargs = {
            "scale_factor": scale_factor,

            "clim": lim['c'],
            "autoscalec_on": autoscale['c'],
            "segment_count": int(getValues(getKeys(colorbar_tab.getSegmentCountElement()))),
            "colormap": getValues(getKeys(colorbar_tab.getColormapInputElement())),

            "axes_kwargs": {
                "xlim": lim['x'],
                "xlabel": label['x'],
                "autoscalex_on": autoscale['x'],
                "xscale": scale_type['x'],

                "ylim": lim['y'],
                "ylabel": label['y'],
                "autoscaley_on": autoscale['y'],
                "yscale": scale_type['y'],

                # "zlim": lim['z'],
                # "zlabel": label['z'],
                # "autoscalez_on": autoscale['z'],
                # "zscale": scale_type['z'],

                "title": getValues(getKeys(axis_tab.getTitleInputElement("plot")))
            },

            "colorbar_kwargs": {
                "label": getValues(getKeys(colorbar_tab.getTitleInputElement())),
            }
        }
        return aesthetics_kwargs

    def getFigure(self, data: Dict[str, ndarray] = None, **kwargs) -> Figure:
        """
        Get matplotlib figure for results.
        
        :param data: data to plot.
            Key is name of axis.
            Value is data to plot along axis.
        :param kwargs: additional arguments to pass into :meth:`~SimulationWindow.getFigure`
        :returns: Displayed figure
            if any element in :paramref:`~Layout.SimulationWindow.SimulationWindowRunner.getFigure.data` is not ndarray.
            Figure with given data plotted if both are ndarray.
        """
        if data is None or any(not isinstance(x, ndarray) for x in data.values()):
            figure_canvas = self.getFigureCanvas()
            figure_canvas_attributes = vars(figure_canvas)
            figure = figure_canvas_attributes["figure"]
            return figure
        else:
            aesthetics = self.getPlotAesthetics()
            return getFigure(data, **kwargs, **aesthetics)

    def getFigureCanvas(self) -> FigureCanvasTkAgg:
        """
        Get figure-canvas in present state.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve figure-canvas from
        :returns: Figure-canvas object if figure has been drawn on canvas previously. None otherwise.
        """
        return self.figure_canvas

    def updateFigureCanvas(self, figure: Figure) -> Optional[FigureCanvasTkAgg]:
        """
        Update figure-canvas aggregate in simulation window.
        This plots the most up-to-date data and aesthetics on the plot.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to update figure-canvas in
        :param figure: new figure (containing most up-to-date info) to plot
        :returns: New figure-canvas stored in window runner.
        """
        figure_canvas = self.getFigureCanvas()
        if isinstance(figure_canvas, FigureCanvasTkAgg):
            clearFigure(figure_canvas)

        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()
        canvas = window_object.getCanvas()

        # noinspection PyTypeChecker
        self.figure_canvas = drawFigure(canvas.TKCanvas, figure)
        window_object.getWindow().Refresh()
        return self.figure_canvas

    def getCondensorKwargs(self, condensor_specie: str) -> Dict[str, Union[str, int]]:
        """
        Get condensor kwargs indicated how to calculate condensed variable.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve arguments from
        :param condensor_specie: specie of condensor to retrieve arguments for
        """
        if condensor_specie == "Frequency":
            condensor_kwargs = {
                "calculation_method": self.getFrequencyMethod(),
                "condensing_method": self.getFrequencyCondensor()
            }
        elif condensor_specie == "Mean":
            condensor_kwargs = {
                "order": self.getMeanOrder()
            }
        elif condensor_specie == "Standard Deviation":
            condensor_kwargs = {}
        else:
            raise TypeError(f"invalid value for condensor specie ({condensor_specie:s})")

        return condensor_kwargs

    def updatePlot(
            self,
            index: Union[tuple, Tuple[int]] = None,
            plot_quantities: Dict[str, Tuple[str, str]] = None,
            transform_name: str = None,
            **figure_kwargs
    ) -> Optional[Figure]:
        """
        Update window-embedded plot.
        Do nothing if simulation has never been run.
        
        :param index: index of parameter value for free parameter(s)
        :param plot_quantities: dictionary storing info for plot quantities.
            Key is name of axis to plot quantity on.
            Value is tuple of quantity info.
            First element gives quantity name.
            Second element gives quantity type.
            Third element gives condensor name.
        :param transform_name: type of transform to perform on quantities before plotting
        :param figure_kwargs: additional arguments to pass into :meth:`~SimulationWindow.SimulationWindow.getFigure`
        :returns: New matplotlib Figure displayed on canvas. None if figure has not been displayed yet.
        """
        if index is None:
            index = self.getClosestSliderIndex()
        if plot_quantities is None:
            plot_quantities = {}
        plot_quantities_keys = plot_quantities.keys()

        for axis_name in self.getAxisNames():
            if axis_name not in plot_quantities_keys:
                quantity = self.getAxisQuantity(axis_name)
                if quantity[0] is None:
                    return None
                else:
                    plot_quantities[axis_name] = quantity

        if transform_name is None:
            # noinspection PyTypeChecker
            window_object: SimulationWindow = self.getWindowObject()
            # noinspection PyTypeChecker
            plotting_tab: PlottingTab = window_object.getPlottingTab()

            combobox_key = getKeys(plotting_tab.getTransformElement())
            transform_name = self.getValue(combobox_key)
            if transform_name is None:
                return None
            self.getValue(combobox_key)

        results_object = self.getResultsObject()
        timelike_species = self.getLikeSpecies("timelike")
        parameterlike_species = self.getLikeSpecies("parameterlike")
        results_kwargs = {
            "index": index,
            "transform_name": transform_name
        }

        axis2name = {
            axis: plot_quantities[axis][0] 
            for axis in plot_quantities_keys
        }

        axis2specie = {
            axis: plot_quantities[axis][1] 
            for axis in plot_quantities_keys
        }
        axis2condensor_name = {
            axis: plot_quantities[axis][2] 
            for axis in plot_quantities_keys
        }
        is_timelike = {
            axis: (
                axis2specie[axis] in timelike_species and 
                axis2condensor_name[axis] == "None"
            )
            for axis in plot_quantities_keys
        }
        is_condensed = {
            axis: (
                axis2specie[axis] in timelike_species and 
                axis2condensor_name[axis] != "None"
            )
            for axis in plot_quantities_keys
        }
        is_parameterlike = {
            axis: axis2specie[axis] in parameterlike_species
            for axis in plot_quantities_keys
        }
        is_nonelike = {
            axis: (
                not is_timelike[axis] and 
                not is_condensed[axis] and 
                not is_parameterlike[axis]
            )
            for axis in plot_quantities_keys
        }

        timelike_count = sum(is_timelike.values())
        parameterlike_count = sum(is_parameterlike.values())
        condensed_count = sum(is_condensed.values())

        results = {}
        try:
            if parameterlike_count == 0:
                getResultsOverTime = partial(results_object.getResultsOverTime, **results_kwargs)

                for axis_name, quantity_name in axis2name.items():
                    if is_timelike[axis_name]:
                        results[axis_name] = getResultsOverTime(quantity_names=quantity_name)
                    elif is_condensed[axis_name]:
                        condensor_name = axis2condensor_name[axis_name]
                        results[axis_name] = getResultsOverTime(
                            quantity_names=quantity_name,
                            condensor_name=condensor_name,
                            **self.getCondensorKwargs(condensor_name)
                        )
            else:
                parameter_names = tuple(
                    [
                        axis2name[axis]
                        for axis in plot_quantities_keys
                        if is_parameterlike[axis]
                    ]
                )

                getResultsOverTimePerParameter = partial(
                    results_object.getResultsOverTimePerParameter,
                    parameter_names=parameter_names,
                    **results_kwargs
                )

                for axis_name, quantity_name in axis2name.items():
                    if is_timelike[axis_name]:
                        parameter_results, quantity_results = getResultsOverTimePerParameter(
                            quantity_names=quantity_name
                        )
                        results[axis_name] = quantity_results[0]
                    elif is_parameterlike[axis_name]:
                        results[axis_name] = results_object.getFreeParameterValues(names=quantity_name)
                    elif is_condensed[axis_name]:
                        condensor_name = axis2condensor_name[axis_name]
                        parameter_results, quantity_results = getResultsOverTimePerParameter(
                            quantity_names=quantity_name,
                            condensor_name=condensor_name,
                            **self.getCondensorKwargs(condensor_name)
                        )
                        results[axis_name] = quantity_results[0]

        except (UnboundLocalError, KeyError, IndexError, AttributeError, ValueError):
            print("data:", traceback.print_exc())

        if timelike_count == 1:
            plot_type = ''
        elif condensed_count >= 1 and parameterlike_count == 0:
            plot_type = ''
        elif timelike_count >= 1 and condensed_count >= 1:
            plot_type = ''
        elif is_nonelike['x'] or is_nonelike['y']:
            plot_type = ''
        else:
            if is_parameterlike['c']:
                plot_type = f"nc"
            elif is_condensed['c']:
                plot_type = 'c'
            elif is_timelike['c']:
                plot_type = 't'
            elif is_nonelike['c']:
                plot_type = ''
            else:
                plot_type = ''

            for axis_name in ['x', 'y', 'z']:
                if is_parameterlike[axis_name]:
                    plot_type += f"n{axis_name:s}"
                elif is_condensed[axis_name] or is_timelike[axis_name]:
                    plot_type += axis_name
                elif is_nonelike[axis_name]:
                    pass

        try:
            print(plot_type, {key: value.shape for key, value in results.items()})
            if plot_type != '':
                figure = self.getFigure(results, plot_type=plot_type, **figure_kwargs)
            self.updateFigureCanvas(figure)
            return figure
        except UnboundLocalError:
            print("todo plots:", plot_quantities, traceback.print_exc())
        except (KeyError, AttributeError):
            print("figure:", traceback.print_exc())

    def updatePlotChoices(self, names: Union[str, List[str]] = None) -> None:
        """
        Update plot choices for desired axis(es).
        This allows user to select new set of quantities to plot.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to change plot choices in
        :param names: name(s) of axis(es) to update choices for.
            Updates all axes by default.
        """

        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()
        # noinspection PyTypeChecker
        plotting_tab: PlottingTab = window_object.getPlottingTab()

        def update(name: str) -> None:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindowRunner.updatePlotChoices`"""
            specie = self.getValue(getKeys(plotting_tab.getAxisQuantitySpeciesElement(name)))
            choices = self.getPlotChoices(species=specie) if specie != "None" else ['']
            quantity_combobox = plotting_tab.getAxisQuantityElement(name)

            quantity_combobox_key = getKeys(quantity_combobox)
            quantity_combobox = self.getElements(quantity_combobox_key)

            try:
                condensor_combobox = plotting_tab.getAxisCondensorElement(name)
                is_over_time_quantity_specie = specie in ["Variable", "Function"]
                condensor_combobox.update(visible=is_over_time_quantity_specie)
            except KeyError:
                pass

            if choices != vars(quantity_combobox)["Values"]:
                kwargs = {
                    "values": choices,
                    "disabled": specie == "None",
                    "size": vars(quantity_combobox)["Size"]
                }

                old_chosen_quantity = self.getValue(quantity_combobox_key)
                change_quantity = old_chosen_quantity not in choices
                if change_quantity:
                    new_chosen_quantity = choices[0]
                    kwargs["value"] = new_chosen_quantity
                    quantity_combobox.update(**kwargs)
                    window_object.getWindow().write_event_value(quantity_combobox_key, new_chosen_quantity)
                else:
                    quantity_combobox.update(**kwargs)

        return recursiveMethod(
            args=names,
            base_method=update,
            valid_input_types=str,
            default_args=self.getAxisNames()
        )

    def saveResults(self) -> None:
        """
        Save :class:`~Results.Results` stored by simulation into file.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve results from
        """
        file_types = (("Compressed File", "*.zip"), ("ALL files", "*.*"),)

        filename = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Results",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filename, str):
            self.getResultsObject().saveToFile(filename)

    def saveModelParameters(self) -> None:
        """
        Save parameter values from model into file.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        file_types = (
            *config_file_types,
            ("ALL Files", "*.*"),
        )

        filename = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Parameters",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filename, str):
            self.getModel().saveParametersToFile(filename)

    def saveModelFunctions(self) -> None:
        """
        Save functions from model into file.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        file_types = (
            *config_file_types,
            ("LaTeX", "*.tex"),
            ("Portable Document Format", "*.pdf"),
            ("ALL Files", "*.*"),
        )
        
        filename = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Functions",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filename, str):
            model = self.getModel()
            model.saveFunctionsToFile(filename)

    def saveModelTimeEvolutionTypes(self) -> None:
        """
        Save time-evolution types from model into file.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        file_types = (
            *config_file_types,
            ("ALL Files", "*.*"),
        )

        filename = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Time-Evolution Types",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filename, str):
            self.getModel().saveTimeEvolutionTypesToFile(filename)

    def saveFigure(self, name: str = None) -> None:
        """
        Save displayed figure as image file if name is None.
        Save animation of figures while sweeping over a free parameter if name is str.
        
        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve figure from
        :param name: name of parameter to loop over for GIF
        """
        if name is None:
            figure = self.getFigure()
            file_types = [
                (name, [f"*.{extension:s}" for extension in extensions])
                for name, extensions in figure.canvas.get_supported_filetypes_grouped().items()
            ]
            # noinspection PyTypeChecker
            file_types.append(("ALL Files", "*.*"))

            filepath = sg.PopupGetFile(
                message="Enter Filename",
                title="Save Figure",
                save_as=True,
                file_types=tuple(file_types)
            )

            if isinstance(filepath, str):
                try:
                    figure.savefig(filepath)
                except OSError:
                    sg.PopupError(f"cannot save figure as {filepath:s}")
        elif isinstance(name, str):
            file_types = [("Compressed File", "*.zip"), ("ALL Files", "*.*")]

            zip_filepath = sg.PopupGetFile(
                message="Enter Filename",
                title="Save Figure",
                save_as=True,
                file_types=tuple(file_types)
            )

            if isinstance(zip_filepath, str):
                save_directory = Path(zip_filepath).parent
                gif_filepath = zip_filepath.replace(".zip", ".gif")
                yaml_filepath = join(save_directory, "values.yml")

                parameter_index = self.getFreeParameterIndex(name)
                default_index = list(self.getClosestSliderIndex())
                parameter_values = self.getFreeParameterValues(name)

                inset_parameters = {
                    name: {
                        "range": (parameter_values.min(), parameter_values.max())
                    }
                }
                image_count = len(parameter_values)

                with ZipFile(zip_filepath, 'w') as zipfile:
                    with imageio.get_writer(gif_filepath, mode='I') as writer:
                        for i in range(image_count):
                            if not self.updateProgressMeter("Saving Animation", i, image_count):
                                break

                            data_index = default_index
                            data_index[parameter_index] = i
                            parameter_value = parameter_values[i]
                            inset_parameters[name]["value"] = parameter_value
                            figure = self.updatePlot(index=tuple(data_index), inset_parameters=inset_parameters)

                            png_filepath = join(save_directory, f"{name:s}_{i:d}.png")
                            figure.savefig(png_filepath)
                            writer.append_data(imageio.imread(png_filepath))
                            zipfile.write(png_filepath, basename(png_filepath))
                            os.remove(png_filepath)

                    zipfile.write(gif_filepath, basename(gif_filepath))
                    os.remove(gif_filepath)

                    yaml_parameter_values = list(map(float, parameter_values))
                    yaml_parameter_indicies = list(range(len(parameter_values)))
                    parameter_values_dict = dict(zip(yaml_parameter_indicies, yaml_parameter_values))
                    saveConfig(parameter_values_dict, yaml_filepath)

                    zipfile.write(yaml_filepath, basename(yaml_filepath))
                    os.remove(yaml_filepath)

                    self.updateProgressMeter("Saving Animation", image_count, image_count)
            else:
                sg.PopupError("invalid filepath")
