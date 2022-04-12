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
from typing import Dict, Iterable, List, Optional, Tuple, Union
# noinspection PyPep8Naming
from zipfile import ZipFile

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from CustomFigure import getFigure
from Function import Model
from macros import StoredObject, formatValue, getIndicies, getTexImage, recursiveMethod, removeAtIndicies, unique
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import ndarray
from pint import Quantity, Unit
from Results import Results
from Simulation import formatResultsAsDictionary, solveODE
from sympy import Expr, Symbol
from sympy.core import function
from sympy.utilities.lambdify import lambdify
from YML import config_file_types, getDimensions, getStates, loadConfig, saveConfig

from Layout.Layout import Element, Frame, Layout, RadioGroup, Row, Tab, TabbedWindow, TabGroup, WindowRunner, getKeys, storeElement

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


def clearFigure(figure_canvas: FigureCanvasTkAgg) -> None:
    """
    Clear figure on figure-canvas aggregate.

    :param figure_canvas: figure-canvas aggregate to clear canvas on
    """
    if isinstance(figure_canvas, FigureCanvasTkAgg):
        figure_canvas.get_tk_widget().forget()
    plt.close("all")


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

        return sg.Submit(
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


class ColorbarAestheticsTab(Tab, StoredObject):
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
        Constructor for :class:`~Layout.SimulationWindow.ColorbarAestheticsTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the purpose of each input column.

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve row for
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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve elements for
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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
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
        Get element that allows user to choose colormap.

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
        """
        cmaps = plt.colormaps()
        default_cmap = "viridis"

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

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
        """
        values = [f"{int(hundred):d}00" for hundred in np.linspace(1, 9, 9)]

        return sg.Spin(
            values=values,
            initial_value=values[0],
            tooltip="Choose segment count for colorbar segments.",
            size=self.getDimensions(name="colorbar_segment_count_spin"),
            key="-COLORBAR SEGMENT COUNT-"
        )

    @storeElement
    def getLocationElement(self) -> sg.InputCombo:
        """
        Get element that allows user to choose colorbar location.

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve element from
        """
        locations = ("left", "right", "top", "bottom")

        return sg.InputCombo(
            values=locations,
            default_value=locations[0],
            tooltip="Choose colormap for colorbar.",
            size=self.getDimensions(name="colorbar_location_combobox"),
            key="-COLORBAR LOCATION-"
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
            self.getSegmentCountElement(),
            self.getLocationElement()
        ]

        layout = Layout(rows=header_rows)
        layout.addRows(Row(window=self.getWindowObject(), elements=row_elements))
        return layout.getLayout()


class AxisAestheticsTab(Tab, StoredObject):
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
        Constructor for :class:`~Layout.SimulationWindow.AxisAestheticsTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve row for
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

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve elements for
        :param name: name of axis
        """
        lower_limit = sg.InputText(
            default_text='',
            tooltip=f"Enter lower limit for {name:s}-axis",
            size=self.getDimensions(name="axis_lower_limit_input_field"),
            key=f"-LOWER LIMIT {name:s}_AXIS-"
        )
        upper_limit = sg.InputText(
            default_text='',
            tooltip=f"Enter upper limit for {name:s}-axis",
            size=self.getDimensions(name="axis_upper_limit_input_field"),
            key=f"-UPPER LIMIT {name:s}_AXIS-"
        )

        return lower_limit, upper_limit

    @storeElement
    def getTitleInputElement(self, name: str) -> sg.InputText:
        """
        Get element that allows user to input axis labels.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve element from
        :param name: name of axis
        """
        return sg.InputText(
            default_text='',
            tooltip=f"Enter label for {name:s}-axis",
            size=self.getDimensions(name="axis_row_title_input_field"),
            key=f"-TITLE {name:s}_AXIS-"
        )

    @storeElement
    def getAutoscaleElement(self, name: str) -> sg.Checkbox:
        """
        Get element that allows user to determine whether axis is autoscaled.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve element from
        :param name: name of axis
        """
        return sg.Checkbox(
            text='',
            tooltip=f"Choose boolean for {name:s}-axis."
            f"When set True, {name:s}-axis will be autoscaled and limit inputs will be ignored."
            f"When set False, limits inputs will be used if available.",
            default=True,
            size=self.getDimensions(name="autoscale_toggle_checkbox"),
            key=f"-AUTOSCALE {name:s}_AXIS-"
        )

    @storeElement
    def getScaleFactorInputElement(self, name: str) -> sg.Spin:
        """
        Get element that allows user to input scale factor.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve element from
        :param name: name of axis
        """
        values = [f"1e{int(exponent):d}" for exponent in np.linspace(-24, 24, 49)]

        return sg.Spin(
            values=values,
            initial_value="1e0",
            tooltip=f"Choose scale factor for {name:s}-axis. Data is divided by this factor.",
            size=self.getDimensions(name="scale_factor_spin"),
            key=f"-SCALE FACTOR {name:s}_AXIS-"
        )

    @storeElement
    def getScaleTypeInputElement(self, name: str) -> sg.InputCombo:
        """
        Get element that allows user to choose axis scale.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve element from
        :param name: name of axis
        """
        return sg.InputCombo(
            values=["Linear", "Logarithmic"],
            default_value="Linear",
            tooltip=f"Choose scale type for {name:s}-axis",
            size=self.getDimensions(name="scale_type_combobox"),
            key=f"-SCALE TYPE {name:s}_AXIS-"
        )

    @storeElement
    def getAxisLabelElement(self, name: str) -> sg.Text:
        """
        Get label to indicate which axis the row affects.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve element from
        :param name: name of axis
        """
        return sg.Text(
            text=name,
            size=self.getDimensions(name="axis_row_label")
        )

    def getInputRow(self, name: str, is_cartesian: bool = False) -> Row:
        """
        Get row that allows user input for a single axis.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve row for
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

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve layout for
        """
        header_rows = self.getHeaderRows()
        axis_input_rows = [
            self.getInputRow("plot", is_cartesian=False),
            self.getInputRow('x', is_cartesian=True),
            self.getInputRow('y', is_cartesian=True),
            self.getInputRow('z', is_cartesian=True),
            self.getInputRow('X', is_cartesian=False),
            self.getInputRow('Y', is_cartesian=False)
        ]

        layout = Layout(rows=header_rows)
        layout.addRows(axis_input_rows)
        return layout.getLayout()


class AestheticsTabGroup(TabGroup):
    """
    This class contains
        #. :class:`~Layout.SimulationWindow.AxisAestheticsTab`
        #. :class:`~Layout.SimulationWindow.ColorbarAestheticsTab`
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AestheticTabGroup`.

        :param name: name of tabgroup
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tabgroup is stored in.
        """
        tabs = [
            AxisAestheticsTab("Axis", window),
            ColorbarAestheticsTab("Colorbar", window)
        ]
        super().__init__(tabs, name=name)


class AxisQuantityElement:
    def __init__(
        self,
        quantity_count_per_axis: int = 2,
        transform_names: List[str] = None,
        coordinate_names: List[str] = None,
        functional_names: List[str] = None,
        complex_names: List[str] = None
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisQuantityElement`.

        :param quantity_count_per_axis: number of possible quantities to select per axis
        :param transform_names: collection of names for math transforms to perform on results.
            Defaults to ["None"].
        :param coordinate_names: collection of names for coordinate transforms to perform on results.
            Defaults to ["None"].
        :param functional_names: collection of names for functionals to perform on results.
            Defaults to ["None"].
        :param complex_names: collection of names for complex-reduction methods to perform on results.
            Defaults to ["None"].
        """
        self.quantity_count_per_axis = quantity_count_per_axis

        self.transform_names = ["None"] if transform_names is None else transform_names
        self.coordinate_names = ["None"] if coordinate_names is None else coordinate_names
        self.functional_names = ["None"] if functional_names is None else functional_names
        self.complex_names = ["None"] if complex_names is None else complex_names

    def getQuantityCountPerAxis(self) -> int:
        """
        Get number of possible quantities to select per axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityElement` to retrieve count from
        """
        return self.quantity_count_per_axis

    def getTransformNames(self) -> List[str]:
        """
        Get names of transforms.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityElement` to retrieve names from
        """
        return self.transform_names

    def getCoordinateNames(self) -> List[str]:
        """
        Get names of coordinate transforms.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityElement` to retrieve names from
        """
        return self.coordinate_names

    def getFunctionalNames(self) -> List[str]:
        """
        Get names of functionals.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityElement` to retrieve names from
        """
        return self.functional_names

    def getComplexNames(self) -> List[str]:
        """
        Get names of complex-reduction methods.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityElement` to retrieve names from
        """
        return self.complex_names


class AxisQuantityTabGroup(TabGroup, AxisQuantityElement):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisQuantityTabGroup`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in
        :param kwargs: additional arguments to input into :class:`~Layout.SimulationWindow.AxisQuantityElement`
        """
        AxisQuantityElement.__init__(self, **kwargs)

        quantity_count_per_axis = self.getQuantityCountPerAxis()
        transform_names = self.getTransformNames()
        coordinate_names = self.getCoordinateNames()
        functional_names = self.getFunctionalNames()

        tabs = [
            StandardTab(
                "Standard",
                window,
                quantity_count_per_axis=quantity_count_per_axis,
                **kwargs
            ),
            NonStandardTab(
                "Other",
                window,
                quantity_count_per_axis=quantity_count_per_axis,
                **kwargs
            )
        ]
        TabGroup.__init__(self, tabs, name=name)


class AxisQuantityTab(Tab, AxisQuantityElement):
    """
    This class contains the layout for the plotting tab in the simulation window.
        #. Header row to identify functions for input
        #. Axis name label for each axis to identify which axis input affects
        #. Combobox to input quantity species for each axis
        #. Combobox to input quantity for each axis
        #. Combobox to input transform type for each axis
    """

    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisQuantityTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in
        :param kwargs: additional arguments to input into :class:`~Layout.SimulationWindow.AxisQuantityElement`
        """

        Tab.__init__(self, name, window)
        AxisQuantityElement.__init__(self, **kwargs)


class StandardTab(AxisQuantityTab):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.StandardTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in
        :param kwargs: additional arguments to pass into :class:`~Layout.SimulationWindow.AxisQuantityTab`
        """
        super().__init__(
            name,
            window,
            **kwargs
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for standard-axis plotting tab.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityTab` to retrieve layout from
        """
        quantity_count_per_axis = self.getQuantityCountPerAxis()
        coordinate_names = self.getCoordinateNames()
        transform_names = self.getTransformNames()
        functional_names = self.getFunctionalNames()
        complex_names = self.getComplexNames()
        window_object = self.getWindowObject()

        axis_frames = []
        for axis_name in ['x', 'y', 'z', 'c']:
            include_none = False if axis_name in ('x', 'y') else True

            axis_frame = AxisQuantityFrame(
                axis_name,
                window_object,
                quantity_count_per_axis=quantity_count_per_axis,
                include_none=include_none,
                include_continuous=True,
                transform_names=transform_names,
                coordinate_names=coordinate_names,
                functional_names=functional_names,
                complex_names=complex_names
            )
            axis_frames.append(axis_frame.getFrame())

        layout = [[axis_frame] for axis_frame in axis_frames]
        return layout


class NonStandardTab(AxisQuantityTab):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        **kwargs
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.StandardTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in
        :param kwargs: additional arguments to pass into :class:`~Layout.SimulationWindow.AxisQuantityTab`
        """
        super().__init__(
            name,
            window,
            **kwargs
        )

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for standard-axis plotting tab.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityTab` to retrieve layout from
        """
        quantity_count_per_axis = self.getQuantityCountPerAxis()
        coordinate_names = self.getCoordinateNames()
        transform_names = self.getTransformNames()
        functional_names = self.getFunctionalNames()
        complex_names = self.getComplexNames()
        window_object = self.getWindowObject()

        axis_frames = []
        for axis_name in ['X', 'Y', 'C']:
            include_continuous = True if axis_name == 'C' else False

            axis_frame = AxisQuantityFrame(
                axis_name,
                window_object,
                quantity_count_per_axis=quantity_count_per_axis,
                include_none=True,
                include_continuous=include_continuous,
                transform_names=transform_names,
                coordinate_names=coordinate_names,
                functional_names=functional_names,
                complex_names=complex_names
            )
            axis_frames.append(axis_frame.getFrame())

        layout = [[axis_frame] for axis_frame in axis_frames]
        return layout


class AxisQuantityFrame(Frame, AxisQuantityElement, StoredObject):
    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        include_none: bool = True,
        include_continuous: bool = True,
        include_discrete: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisQuantityFrame`.

        :param name: name of axis
        :param include_none: set True to allow user to choose "None" for quantity.
            Set False otherwise.
        :param include_continuous: set True to allow user to choose continuous-like quantities.
            Set False otherwise.
        :param include_discrete: set True to allow user to choose discrete-like quantities.
            Set False otherwise.
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that frame is stored in
        :param kwargs:   
        """
        Frame.__init__(self, name, window)
        StoredObject.__init__(self, name)
        AxisQuantityElement.__init__(self, **kwargs)

        self.include_none = include_none
        self.include_continuous = include_continuous
        self.include_discrete = include_discrete

    def includeNone(self) -> bool:
        """
        Get whether to include "None" choice for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve boolean from
        """
        return self.include_none

    def includeContinuous(self) -> bool:
        """
        Get whether to include continuous (timelike) choice(s) for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve boolean from
        """
        return self.include_continuous

    def includeDiscrete(self) -> bool:
        """
        Get whether to include dicrete (parameterlike) choice(s) for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve boolean from
        """
        return self.include_discrete

    @storeElement
    def getAxisTransformElement(self) -> sg.InputCombo:
        """
        Get element to take user input for tranform.
        This allows user to choose which transform to perform on plot quantities.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        """
        name = self.getName()
        transform_names = self.getTransformNames()

        return sg.InputCombo(
            values=transform_names,
            default_value=transform_names[0],
            tooltip=f"Choose transform to perform on {name:s}axis of plot",
            enable_events=True,
            size=self.getDimensions(name="transform_type_combobox"),
            key=f"{cc_pre:s} TRANSFORM {name:s}_AXIS"
        )

    @storeElement
    def getAxisFunctionalElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis functional.
        This allows user to choose which type of functional to calculate for a plot quantity (e.g. frequency, amplitude).

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        """
        name = self.getName()
        functional_names = self.getFunctionalNames()
        default_value = functional_names[0]

        return sg.InputCombo(
            values=functional_names,
            default_value=default_value,
            tooltip=f"Choose functional to perform on {name:s}-axis of plot",
            enable_events=True,
            size=self.getDimensions(name="axis_functional_combobox"),
            key=f"-{cc_pre:s} CONDENSOR {name:s}_AXIS-"
        )

    @storeElement
    def getAxisNormalizeElement(self) -> sg.Checkbox:
        """
        Get element to take user input for whether to normalize axis data.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        """
        name = self.getName()
        return sg.Checkbox(
            text='',
            default=False,
            key=f"-NORMALIZE IS ACTIVE {name:s}_AXIS-"
        )

    @storeElement
    def getAxisComplexGroup(self) -> ComplexRadioGroup:
        """
        Get element to take user input for complex-reduction method (e.g. "Magnitude", "Phase").

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        """
        axis_name = self.getName()
        window_obj = self.getWindowObject()
        complex_names = self.getComplexNames()

        complex_radio_group = ComplexRadioGroup(
            axis_name=axis_name,
            window=window_obj,
            complex_names=complex_names
        )
        return complex_radio_group

    @storeElement
    def getAxisCoordinateGroup(self) -> CoordinateRadioGroup:
        """
        Get element to take user input for coordinate system (e.g. "Cartesian").

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        """
        axis_name = self.getName()
        window_obj = self.getWindowObject()
        coordinate_names = self.getCoordinateNames()

        coordinate_radio_group = CoordinateRadioGroup(
            axis_name=axis_name,
            window=window_obj,
            coordinate_names=coordinate_names
        )
        return coordinate_radio_group

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get frame that allows user input for a single axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve frame from
        """
        name = self.getName()
        quantity_count_per_axis = self.getQuantityCountPerAxis()
        include_none = self.includeNone()
        include_continuous = self.includeContinuous()
        include_discrete = self.includeDiscrete()

        rows = []

        coordinate_group = self.getAxisCoordinateGroup()
        coordinate_row_elements = [
            sg.Text("Envelope:"),
            *coordinate_group.getRadios()
        ]
        rows.append(Row(elements=coordinate_row_elements))

        transform_row_elements = [
            sg.Text("Transform:"),
            self.getAxisTransformElement()
        ]
        functional_row_elements = [
            sg.Text("Functional:"),
            self.getAxisFunctionalElement()
        ]
        normalize_row_elements = [
            sg.Text("Normalize:"),
            self.getAxisNormalizeElement()
        ]
        row_elements = [
            *transform_row_elements,
            *functional_row_elements,
            *normalize_row_elements
        ]
        rows.append(Row(elements=row_elements))

        complex_group = self.getAxisComplexGroup()
        complex_row_elements = [
            sg.Text("Complex:"),
            *complex_group.getRadios()
        ]
        rows.append(Row(elements=complex_row_elements))

        for index in range(quantity_count_per_axis):
            include_none_per_quantity = index != 0 or include_none
            axis_row = AxisQuantityRow(
                name,
                window=self.getWindowObject(),
                index=index,
                include_none=include_none_per_quantity,
                include_continuous=include_continuous,
                include_discrete=include_discrete
            )
            rows.append(axis_row)

        layout = Layout(rows).getLayout()
        return layout


class CoordinateRadioGroup(RadioGroup, StoredObject):
    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        coordinate_names: List[str] = None
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.CoordinateRadioGroup`.

        :param axis_name: name of axis associated with group
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that group is stored in
        :param coordinate_names: collection of names for coordinates (e.g. "Cartesian", "Angle", "Modulus").
            Defaults to ["Cartesian", "Angle", "Modulus"].
        """
        group_id = f"-COORDINATE {axis_name:s}_AXIS-"

        RadioGroup.__init__(
            self,
            radios=[],
            name=axis_name,
            group_id=group_id,
            window=window
        )
        StoredObject.__init__(self, axis_name)

        if coordinate_names is None:
            self.coordinate_names = ["Cartesian"]
        else:
            self.coordinate_names = coordinate_names

        for coordinate_name in coordinate_names:
            radio = self.getCoordinateRadio(coordinate_name)
            self.addElements(radio)

    @storeElement
    def getCoordinateRadio(
        self,
        coordinate_name: str,
        default_name: str = "Cartesian"
    ) -> sg.Radio:
        """
        Get element to take user input for coordinate system (individual).

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        :param coordinate_name: name coordinate type
        :param default_name: name of default radio to be set true
        """
        axis_name = self.getName()
        is_default = coordinate_name == default_name
        radio_group_id = self.getGroupId()
        radio_key = radio_group_id + f"{coordinate_name.upper():s}-"

        radio = sg.Radio(
            text=coordinate_name,
            tooltip=f"Choose coordinate transformation to perform on {axis_name:s}-axis of plot",
            group_id=radio_group_id,
            default=is_default,
            key=radio_key
        )
        return radio


class ComplexRadioGroup(RadioGroup, StoredObject):
    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        complex_names: List[str] = None
    ):
        """
        Constructor for :class:`~Layout.SimulationWindow.ComplexRadioGroup`.

        :param axis_name: name of axis associated with group
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that group is stored in
        :param complex_names: collection of names for complexe-reduction methods (e.g. "Magnitude", "Phase")
        """
        group_id = f"-COMPLEX {axis_name:s}_AXIS-"

        RadioGroup.__init__(
            self,
            radios=[],
            name=axis_name,
            group_id=group_id,
            window=window
        )
        StoredObject.__init__(self, axis_name)

        if complex_names is None:
            self.complex_names = ["Magnitude"]
        else:
            self.complex_names = complex_names

        for complex_name in complex_names:
            radio = self.getComplexRadio(complex_name)
            self.addElements(radio)

    @storeElement
    def getComplexRadio(
        self,
        complex_name: str,
        default_name: str = "Magnitude"
    ) -> sg.Radio:
        """
        Get element to take user input for complex system (individual).

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityFrame` to retrieve element from
        :param complex_name: name of complex-reduction method
        :param default_name: name of default radio to be set true
        """
        axis_name = self.getName()
        is_default = complex_name == default_name
        radio_group_id = self.getGroupId()
        radio_key = radio_group_id + f"{complex_name.upper():s}-"

        radio = sg.Radio(
            text=complex_name,
            tooltip=f"Choose complex-reduction method to perform on {axis_name:s}-axis of plot",
            group_id=radio_group_id,
            default=is_default,
            key=radio_key
        )
        return radio


class AxisQuantityRow(Row, StoredObject):
    """
    :ivar getPlotChoices: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotChoices`
    """

    def __init__(
        self,
        axis_name: str,
        window: SimulationWindow,
        index: int = 0,
        include_none: bool = None,
        include_continuous: bool = True,
        include_discrete: bool = True
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisQuantityRow`.

        :param axis_name: name of axis
        :param index: quantity index per axis
        :param include_none: set True to allow user to choose "None" for quantity.
            Set False otherwise.
            Must be True if index != 0.
            Defaults to index != 0.
        :param include_continuous: set True to allow user to choose continuous-like quantities.
            Set False otherwise.
        :param include_discrete: set True to allow user to choose discrete-like quantities.
            Set False otherwise.
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that row is stored in
        """
        name = axis_name + str(index)
        Row.__init__(self, name, window=window)
        StoredObject.__init__(self, name)

        self.getPlotChoices = window.getPlotChoices

        self.axis_name = axis_name
        self.index = index

        if index != 0:
            assert include_none
        if include_none is None:
            self.include_none = index != 0
        else:
            self.include_none = index != 0 or include_none

        self.include_continuous = include_continuous
        self.include_discrete = include_discrete

        elements = [
            sg.Text("   Species:"),
            self.getAxisQuantitySpeciesElement(),
            sg.Text("Quantity:"),
            self.getAxisQuantityElement()
        ]
        self.addElements(elements)

    def getAxisName(self) -> str:
        """
        Get name of axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve name from
        """
        return self.axis_name

    def getIndex(self) -> int:
        """
        Get index of row within axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve index from
        """
        return self.index

    def includeNone(self) -> bool:
        """
        Get whether to include "None" choice for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve boolean from
        """
        return self.include_none

    def includeContinuous(self) -> bool:
        """
        Get whether to include continuous (timelike) choice(s) for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve boolean from
        """
        return self.include_continuous

    def includeDiscrete(self) -> bool:
        """
        Get whether to include dicrete (parameterlike) choice(s) for quantity.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve boolean from
        """
        return self.include_discrete

    @storeElement
    def getAxisQuantityElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity.
        This allows user to choose which quantity to plot on the axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve element from
        """
        name = self.getName()

        include_none = self.includeNone()
        if include_none:
            values = ['']
            disabled = True
        else:
            values = self.getPlotChoices(species="Variable")
            disabled = False

        sg_kwargs = {
            "tooltip": f"Choose quantity to display on {name:s}-axis of plot",
            "enable_events": True,
            "size": self.getDimensions(name="axis_quantity_combobox"),
            "key": f"-{cc_pre:s} QUANTITY {name:s}_AXIS-"
        }

        sg_kwargs["values"] = values
        sg_kwargs["default_value"] = sg_kwargs["values"][0]
        sg_kwargs["disabled"] = disabled

        elem = sg.InputCombo(**sg_kwargs)
        return elem

    @storeElement
    def getAxisQuantitySpeciesElement(self) -> sg.InputCombo:
        """
        Get element to take user input for an axis quantity type.
        This allows user to choose which type of quantity to plot on the axis.

        :param self: :class:`~Layout.SimulationWindow.AxisQuantityRow` to retrieve element from
        """
        name = self.getName()
        include_none = self.includeNone()
        include_continuous = self.includeContinuous()
        include_discrete = self.includeDiscrete()

        axis_quantity_species = []
        if include_none:
            axis_quantity_species.append("None")
        if include_continuous:
            axis_quantity_species.extend(["Variable", "Function"])
        if include_discrete:
            if len(self.getPlotChoices(species="Parameter")) >= 1:
                axis_quantity_species.append("Parameter")

        return sg.InputCombo(
            values=axis_quantity_species,
            default_value=axis_quantity_species[0],
            tooltip=f"Choose quantity species to display on {name:s}-axis of plot",
            enable_events=True,
            size=self.getDimensions(name="axis_quantity_species_combobox"),
            key=f"-{ccs_pre:s} {name:s}_AXIS-"
        )


class FilterTab(Tab, StoredObject):
    """
    This class contains the layout for the filter tab in the simulation window.
        #. One or more :class:`~Layout.SimulationWindow.FilterRow`, each comprising one node of AND-gate filter

    :ivar row_count: number of rows to include in tab
    """

    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        row_count: int = 1,
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.FilterTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

        self.row_count = row_count

    def getRowCount(self) -> int:
        """
        Get number of rows to include in tab.

        :param self: :class:`~Layout.SimulationWindow.FilterTab` to retrieve count from
        """
        return self.row_count

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for filter tab.

        :param self: :class:`~Layout.SimulationWindow.FilterTab` to retrieve layout from
        """
        window = self.getWindowObject()

        row_count = self.getRowCount()
        row_ids = range(row_count)

        rows = []
        for row_id in row_ids:
            new_filter_row = FilterRow(row_id, window=window)
            rows.append(new_filter_row)

        layout = Layout(rows=rows)
        return layout.getLayout()


class FilterRow(Row, StoredObject):
    """
    This class contains the layout for the filter row in the filter tab.
        #. Combobox to select left side of inequality (variable name)
        #. Combobox to select type of inequality (e.g. '>', '<=')
        #. Input field to type into  right side of inequality (float)
        #. Checkbox to set whether filter is active or not.

    :ivar row_id: number to indicate id of row within :class:`~Layout.SimulationWindow.FilterTab`
    """

    valid_inequality_types = ('>', '<', '==', '>=', '<=', '!=')

    def __init__(self, row_id: int, window: SimulationWindow) -> None:
        row_name = f"-FILTER ROW {row_id:d}"
        Row.__init__(self, row_name, window=window)
        StoredObject.__init__(self, row_name)

        self.row_id = row_id

        elements = [
            self.getLeftVariableElement(),
            self.getInequalityTypeElement(),
            self.getRightFloatElement(),
            self.getIsActiveElement()
        ]

        self.addElements(elements)

    def getRowId(self) -> int:
        """
        Get ID associated with row within tab.

        :param self: :class:`~Layout.SimulationWindow.FilterRow` to retrieve ID from
        """
        return self.row_id

    @storeElement
    def getLeftVariableElement(self) -> sg.Combo:
        """
        Get element that allows user input for variable in inequality.

        :param self: :class:`~Layout.SimulationWindow.FilterRow` to retrieve element from
        """
        row_id = self.getRowId()

        window_obj: SimulationWindow = self.getWindowObject()
        variable_names = window_obj.getPlotChoices(species="Variable")
        function_names = window_obj.getPlotChoices(species="Function")

        inequality_choices = variable_names + function_names

        return sg.Combo(
            values=inequality_choices,
            default_value=inequality_choices[0],
            key=f"-FILTER LEFT CHOICE {row_id:d}-"
        )

    @storeElement
    def getInequalityTypeElement(self) -> sg.Combo:
        """
        Get element that allows user input for inequality type.

        :param self: :class:`~Layout.SimulationWindow.FilterRow` to retrieve element from
        """
        row_id = self.getRowId()
        valid_inequality_types = FilterRow.valid_inequality_types

        return sg.Combo(
            values=valid_inequality_types,
            default_value=valid_inequality_types[0],
            key=f"FILTER TYPE {row_id:d}"
        )

    @storeElement
    def getRightFloatElement(self) -> sg.Input:
        """
        Get element that allows user input for inequality type.

        :param self: :class:`~Layout.SimulationWindow.FilterRow` to retrieve element from
        """
        row_id = self.getRowId()

        return sg.Input(
            default_text='0',
            key=f"-FILTER RIGHT CHOICE {row_id:d}-"
        )

    @storeElement
    def getIsActiveElement(self) -> sg.Checkbox:
        """
        Get element that allows user to set whether or not filter dictating by row is active.

        :param self: :class:`~Layout.SimulationWindow.FilterTab` to retrieve element from
        """
        row_id = self.getRowId()

        return sg.Checkbox(
            text="Active?",
            default=False,
            key=f"-FILTER IS ACTIVE{row_id:d}-"
        )


class AnalysisTab(Tab, StoredObject):
    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AnalysisTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab group is stored in.
        """
        Tab.__init__(self, name, window)
        StoredObject.__init__(self, name)

    @storeElement
    def getFrequencyMethodElement(self) -> sg.InputCombo:
        """
        Get element to take user input for calculation method.
        This allows user to choose which method to calculate frequency with.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve element from
        """
        frequency_method = [
            "Separation of Maxima",
            "Separation of Minima",
            "Separation of Extrema"
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
    def getFrequencyFunctionalElement(self) -> sg.InputCombo:
        """
        Get element to take user input for functional.
        This allows user to choose which functional to reduce frequency with.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve element from
        :param functional_name: name of functional to retrieve element for.
            Must be "frequency".
        """
        axis_functional = ["Average", "Maximum", "Minimum", "Initial", "Final"]

        return sg.InputCombo(
            values=axis_functional,
            default_value=axis_functional[0],
            tooltip=f"Choose method to condense frequency",
            enable_events=True,
            size=self.getDimensions(name=f"frequency_functional_combobox"),
            key=f"ANALYSIS CONDENSOR FREQUENCY"
        )

    def getFrequencyRow(self) -> Row:
        """
        Get row to take user input for frequency functional.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve row from
        """
        frequency_text = sg.Text(
            text="Frequency -",
            justification="left"
        )
        method_text = sg.Text(
            text="Method:",
            justification="right"
        )
        functional_text = sg.Text(
            text="Functional:",
            justification="right"
        )

        method_element = self.getFrequencyMethodElement()
        functional_element = self.getFrequencyFunctionalElement()

        elements = [
            frequency_text,
            method_text,
            method_element,
            functional_text,
            functional_element
        ]
        row = Row(window=self.getWindowObject(), elements=elements)
        return row

    @storeElement
    def getMeanOrderElement(self) -> sg.Spin:
        """
        Get element to take user input for order.
        This allows user to choose order of Holder mean.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve element from
        """
        values = ['-inf', '-1', '0', '1', '2', 'inf']

        return sg.Spin(
            values=values,
            initial_value='1',
            size=self.getDimensions(name="mean_order_spin"),
            key=f"-ANALYSIS ORDER MEAN-"
        )

    def getMeanRow(self) -> Row:
        """
        Get row to take user input for Holder mean functional.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve row from
        """
        mean_text = sg.Text(
            text="Holder Mean -",
            justification="left"
        )
        order_text = sg.Text(
            text="Order:",
            justification="right"
        )

        order_element = self.getMeanOrderElement()

        elements = [
            mean_text,
            order_text,
            order_element
        ]
        row = Row(window=self.getWindowObject(), elements=elements)
        return row

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for analysis tab.

        :param self: :class:`~Layout.SimulationWindow.AnalysisTab` to retrieve layout from
        """
        layout = Layout()
        layout.addRows(self.getFrequencyRow())
        layout.addRows(self.getMeanRow())

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
        plot_choices: Dict[str, List[str]],
        include_simulation_tab: bool = True,
        transform_config_filepath: str = "transforms/transforms.json",
        coordinate_config_filepath: str = "transforms/coordinates.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json"
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
        :param transform_config_filepath: filepath for file containing info about transforms
        :param coordinate_config_filepath: filepath for file containing info about coordinate transformations
        :param functional_config_filepath: filepath for file containing info about functionals
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
            "colorbar_location_combobox": getDimensions(
                ["simulation_window", "aesthetics_tab", "colorbar_tab", "location_combobox"]
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
            "axis_header_row_functional_name": getDimensions(
                ["simulation_window", "plotting_tab", "header_row", "functional_name"]
            ),
            "axis_quantity_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_quantity_combobox"]
            ),
            "axis_quantity_species_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_quantity_species_combobox"]
            ),
            "axis_functional_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "axis_row", "axis_functional_combobox"]
            ),
            "transform_type_combobox": getDimensions(
                ["simulation_window", "plotting_tab", "transform_type_combobox"]
            ),
            "frequency_header_row_method_name": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "header_row", "method_name"]
            ),
            "frequency_header_row_functional_name": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "header_row", "functional_name"]
            ),
            "frequency_method_combobox": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "method_combobox"]
            ),
            "frequency_functional_combobox": getDimensions(
                ["simulation_window", "analysis_tab", "frequency_tab", "functional_combobox"]
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
        simulation_tab = SimulationTab(simulation_tab_name, self)
        self.getSimulationTab = partial(self.getTabs, names=simulation_tab_name)

        plotting_tabgroup_name = "Plotting"
        transform_config = loadConfig(transform_config_filepath)
        transform_names = list(transform_config.keys())
        coordinate_config = loadConfig(coordinate_config_filepath)
        coordinate_names = list(coordinate_config.keys())
        functional_config = loadConfig(functional_config_filepath)
        functional_names = list(functional_config.keys())
        complex_config = loadConfig(complex_config_filepath)
        complex_names = list(complex_config.keys())

        plotting_tabgroup = AxisQuantityTabGroup(
            plotting_tabgroup_name,
            self,
            transform_names=transform_names,
            coordinate_names=coordinate_names,
            functional_names=functional_names,
            complex_names=complex_names
        )
        self.getQuantityCountPerAxis = plotting_tabgroup.getQuantityCountPerAxis
        self.getTransformNames = plotting_tabgroup.getTransformNames

        if include_simulation_tab:
            self.addTabs(simulation_tab)

        self.addTabs(AestheticsTabGroup("Aesthetics", self).getAsTab())
        self.addTabs(plotting_tabgroup.getAsTab())
        self.addTabs(AnalysisTab("Analysis", self))
        self.addTabs(FilterTab("Filter", self, row_count=4))

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
                        "Variables::Save"
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

        layout = prefix_layout.getLayout() + [[sg.Column(left_layout.getLayout()), sg.Column(right_layout.getLayout())]]
        return layout


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
    :ivar include_simulation_tab: whether or not to include tab to run simulation.
        Set True to include tab. Set False to exclude tab. Defaults to True.
    :ivar getPlotChoices: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotChoices`
    :ivar getFreeParameterNames: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getFreeParameterNames`
    :ivar resetResults: pointer to :meth:`~Results.Results.resetResults`
    :ivar setResults: pointer to :meth:`~Results.Results.setResults`
    :ivar getQuantityCountPerAxis: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getQuantityCountPerAxis`
    """

    def __init__(
        self,
        name: str,
        model: Model = None,
        results: Results = None,
        include_simulation_tab: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param name: title of window
        :param model: :class:`~Function.Model` to simulate.
            Only called and required
            if :paramref:`~Layout.SimulationWindow.SimulationWindowRunner.__init__.results` is None.
        :param results: initial results to start simulation with
        :param **kwargs: additional arguments to pass into :class:`~Layout.SimulationWindow.SimulationWindow`
        """
        window_object = SimulationWindow(
            name,
            self,
            include_simulation_tab=include_simulation_tab,
            **kwargs
        )
        super().__init__(window_object)
        window_object.getWindow()

        self.getPlotChoices = window_object.getPlotChoices
        self.getFreeParameterNames = window_object.getFreeParameterNames
        self.getQuantityCountPerAxis = window_object.getQuantityCountPerAxis

        self.include_simulation_tab = include_simulation_tab
        self.axis_names = {
            "grid": ('X', 'Y'),
            "color": ('c'),
            "cartesian": ('x', 'y', 'z')
        }
        self.timelike_species = ["Variable", "Function"]
        self.parameterlike_species = ["Parameter"]
        self.values = None
        self.figure_canvas = None
        self.general_derivative_vector = None
        self.plot_quantities = PlotQuantities(self)

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
        self.setResults = results_object.storeResult
        self.saveResult = results_object.saveResult
        self.saveResults = results_object.saveResults
        self.setResultsFolderpath = results_object.setFolderpath

    def includeSimulationTab(self) -> bool:
        """
        Return whether or not simulation tab is present in window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve tab presence from
        :returns: True if tab is present.
            False if tab is not present.
        """
        return self.include_simulation_tab

    def getPlotQuantities(self) -> PlotQuantities:
        """
        Get object to determine plot quantities in window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retreive object from
        """
        return self.plot_quantities

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

    def getAxisNames(self, filter: str = None) -> List[str]:
        """
        Get names for axes.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve names from
        :param filter: only include axis types that satisfy this filter.
            Must be "cartesian", "grid", or "color".
        :returns: List of axis names (e.g. ['x', 'y', 'c'])
        """

        if filter is None:
            all_names = (
                axis_name
                for axis_names in self.axis_names.values()
                for axis_name in axis_names
            )
            return all_names
        elif isinstance(filter, str):
            filter = filter.lower()
            return self.axis_names[filter]
        else:
            raise TypeError("filter must be str")

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

    def getFrequencyMethod(self) -> str:
        """
        Get selected method to calculate frequency.
        Uses present state of window runner.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve method from
        """
        # noinspection PyTypeChecker
        analysis_tab: AnalysisTab = AnalysisTab.getInstances()[0]
        method_key = getKeys(analysis_tab.getFrequencyMethodElement())
        method = self.getValue(method_key)
        return method

    def getFrequencyFunctional(self) -> str:
        """
        Get selected functional to calculate frequency.
        Uses present state of window runner.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve functional from
        """
        # noinspection PyTypeChecker
        analysis_tab: AnalysisTab = AnalysisTab.getInstances()[0]
        functional_key = getKeys(analysis_tab.getFrequencyFunctionalElement())
        functional = self.getValue(functional_key)
        return functional

    def getMeanOrder(self) -> float:
        """
        Get selected order for Holder mean.
        Uses present state of window runner.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve order from
        """
        # noinspection PyTypeChecker
        analysis_tab: AnalysisTab = AnalysisTab.getInstances()[0]
        order_key = getKeys(analysis_tab.getMeanOrderElement())
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

        is_simulation = self.includeSimulationTab()
        if is_simulation:
            # noinspection PyTypeChecker
            simulation_tab_object: SimulationTab = window_object.getSimulationTab()
            run_simulation_key = getKeys(simulation_tab_object.getRunButton())

        toolbar_menu_key = getKeys(window_object.getMenu())
        update_plot_key = getKeys(window_object.getUpdatePlotButton())

        while True:
            event, self.values = window.read()
            print('event:', event)

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
                elif menu_value == "Variables::Save":
                    self.saveModelVariables()
                elif menu_value == "Static::Save Figure":
                    self.saveFigure()
                elif "Save Animated Figure" in menu_value:
                    free_parameter_name = menu_value.split('::')[0]
                    self.saveFigure(free_parameter_name)
            elif fps_pre in event:
                self.updatePlot()
            elif cc_pre in event:
                if ccs_pre in event:
                    axis_name = event.split(' ')[-1].replace("_AXIS", '').replace('-', '')[0]
                    self.updatePlotChoices(axis_name)
                else:
                    window.write_event_value(update_plot_key, None)
            elif event == update_plot_key:
                self.updatePlot()
            elif is_simulation and event == run_simulation_key:
                self.runSimulations()
        window.close()

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

    def runSimulation(
        self,
        index: Union[tuple, Tuple[int]],
        parameter_values: Dict[str, float],
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
        ydot = lambdify(
            (Symbol('t'), tuple(variable_names)),
            derivatives,
            modules=["math"]
        )
        try:
            solution = solveODE(ydot, y0, times)
            # noinspection PyTypeChecker
            results = formatResultsAsDictionary(*solution, variable_names)
            for variable_name, result in results.items():
                self.saveResult(
                    index=index,
                    name=variable_name,
                    result=result
                )
        except OverflowError:
            print('overflow:', parameter_values)

    def runSimulations(self) -> None:
        """
        Run simulations for all possible combinations of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        """
        self.resetResults()
        free_parameter_names = self.getFreeParameterNames()
        model = self.getModel()
        variable_names = model.getVariables(
            time_evolution_types="Temporal",
            return_type=str
        )
        times = np.linspace(*self.getInputTimes())
        y0 = model.getInitialValues(
            names=variable_names,
            return_type=list
        )
        general_derivative_vector = self.getGeneralDerivativeVector()

        runSimulation = partial(
            self.runSimulation,
            model=model,
            variable_names=variable_names,
            general_derivative_vector=general_derivative_vector,
            y0=y0,
            times=times
        )

        save_folderpath = sg.PopupGetFolder(
            message="Enter Folder to Save Into",
            title="Run Simulation"
        )

        self.setResultsFolderpath(save_folderpath)
        self.saveResults()

        if (parameter_count := len(free_parameter_names)) == 0:
            runSimulation((), {})
        elif parameter_count >= 1:
            free_parameter_values = {}
            free_parameter_indicies = []
            for parameter_name in free_parameter_names:
                free_parameter_value = self.getFreeParameterValues(parameter_name)
                free_parameter_values[parameter_name] = free_parameter_value
                free_parameter_indicies.append(range(len(free_parameter_value)))

            free_parameter_index_combos = tuple(product(*free_parameter_indicies))
            total_combo_count = len(free_parameter_index_combos)
            updateProgressMeter = partial(
                self.updateProgressMeter,
                title="Running Simulation",
                max_value=total_combo_count
            )

            for simulation_number, indicies in enumerate(free_parameter_index_combos):
                if not updateProgressMeter(simulation_number):
                    break

                parameter_simulation_values = {}
                for free_parameter_index in range(parameter_count):
                    free_parameter_name = free_parameter_names[free_parameter_index]
                    free_parameter_value = free_parameter_values[free_parameter_name][indicies[free_parameter_index]]
                    parameter_simulation_values[free_parameter_name] = free_parameter_value

                runSimulation(
                    indicies,
                    parameter_simulation_values
                )
            updateProgressMeter(total_combo_count)

        self.updatePlot()

    def getPlotAesthetics(self) -> Dict[str, Optional[Union[str, float]]]:
        """
        Get plot-aesthetics input by user.

        :param self: :class:`~Layout.SimulationWindow.SimulationTab` to retrieve aesthetic inputs from
        """

        def getValues(elements: Union[str, Iterable[str]]) -> Optional[Union[str, float, bool, tuple]]:
            """
            Get value for plot aesthetic from element key.

            :param elements: :class:`~PySimpleGUI.Element`(s) to retrieve value(s) from
            """

            def get(element: str) -> Optional[Union[str, float, bool]]:
                """Base method for :meth:`~Layout.SimulationWindow.SimulationWindow.getPlotAesthetics.getValues`"""
                key = getKeys(element)
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
                args=elements,
                base_method=get,
                valid_input_types=sg.Element,
                output_type=tuple
            )

        axis_names = self.getAxisNames()
        grid_axis_names = self.getAxisNames("grid")
        cartesian_axis_names = self.getAxisNames("cartesian")
        colorbar_axis_name = self.getAxisNames("color")[0]

        scale_type_dict = {
            "Linear": "linear",
            "Logarithmic": "log"
        }

        axis_aesthetics_tab: AxisAestheticsTab = AxisAestheticsTab.getInstances()[0]
        colorbar_aesthetics_tab: ColorbarAestheticsTab = ColorbarAestheticsTab.getInstances()[0]

        scale_type = {}
        normalize = {}
        scale_factor = {}
        label = {}
        autoscale_on = {}
        lim = {}

        for axis_name in axis_names:
            axis_quantity_frame: AxisQuantityFrame = AxisQuantityFrame.getInstances(names=axis_name)

            #axis_quantity_row: AxisQuantityRow = AxisQuantityRow.getInstances(names=axis_name)
            normalize_element = axis_quantity_frame.getAxisNormalizeElement()
            normalize[axis_name] = getValues(normalize_element)

        for cartesian_axis_name in cartesian_axis_names:
            scale_factor_element = axis_aesthetics_tab.getScaleFactorInputElement(cartesian_axis_name)
            scale_factor[cartesian_axis_name] = getValues(scale_factor_element)
            label_element = axis_aesthetics_tab.getTitleInputElement(cartesian_axis_name)
            label[cartesian_axis_name] = getValues(label_element)
            scale_type_element = axis_aesthetics_tab.getScaleTypeInputElement(cartesian_axis_name)
            scale_type[cartesian_axis_name] = scale_type_dict[getValues(scale_type_element)]

            autoscale_element = axis_aesthetics_tab.getAutoscaleElement(cartesian_axis_name)
            autoscale = getValues(autoscale_element)
            autoscale_on[cartesian_axis_name] = autoscale
            if autoscale:
                lim[cartesian_axis_name] = (None, None)
            else:
                lim_elements = axis_aesthetics_tab.getLimitInputElements(cartesian_axis_name)
                lim[cartesian_axis_name] = getValues(lim_elements)

        for grid_axis_name in grid_axis_names:
            label_element = axis_aesthetics_tab.getTitleInputElement(axis_name)
            label[grid_axis_name] = getValues(label_element)

        scale_factor_element = colorbar_aesthetics_tab.getScaleFactorInputElement()
        scale_factor[colorbar_axis_name] = getValues(scale_factor_element)
        label_element = colorbar_aesthetics_tab.getTitleInputElement()
        label[colorbar_axis_name] = getValues(label_element)

        autoscale_element = colorbar_aesthetics_tab.getAutoscaleElement()
        autoscale = getValues(autoscale_element)
        autoscale_on[colorbar_axis_name] = autoscale
        if autoscale:
            lim[colorbar_axis_name] = (None, None)
        else:
            lim_elements = colorbar_aesthetics_tab.getLimitInputElements()
            lim[colorbar_axis_name] = getValues(lim_elements)

        aesthetics_kwargs = {
            "scale_factor": scale_factor,
            "normalize": normalize,

            "segment_count": int(getValues(colorbar_aesthetics_tab.getSegmentCountElement())),
            "colormap": getValues(colorbar_aesthetics_tab.getColormapInputElement()),

            "axes_kwargs": {
                "xlim": lim['x'],
                "xlabel": label['x'],
                "autoscalex_on": autoscale_on['x'],
                "xscale": scale_type['x'],

                "ylim": lim['y'],
                "ylabel": label['y'],
                "autoscaley_on": autoscale_on['y'],
                "yscale": scale_type['y'],

                "zlim": lim['z'],
                "zlabel": label['z'],
                "autoscalez_on": autoscale_on['z'],
                "zscale": scale_type['z'],

                "clim": lim['c'],
                "clabel": label['c'],
                "autoscalec_on": autoscale_on['c'],
                "cloc": getValues(colorbar_aesthetics_tab.getLocationElement()),

                "Xlabel": label['X'],
                "Ylabel": label['Y'],
                "title": getValues(axis_aesthetics_tab.getTitleInputElement("plot"))
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

    def getFunctionalKwargs(self, functional_specie: str) -> Dict[str, Union[str, int]]:
        """
        Get functional kwargs indicated how to calculate functional variable.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve arguments from
        :param functional_specie: specie of functional to retrieve arguments for
        """
        if functional_specie == "Frequency":
            functional_kwargs = {
                "calculation_method": self.getFrequencyMethod(),
                "condensing_method": self.getFrequencyFunctional()
            }
        elif functional_specie == "Holder Mean":
            functional_kwargs = {
                "order": self.getMeanOrder()
            }
        elif functional_specie == "Standard Deviation":
            functional_kwargs = {}
        elif functional_specie in ["None", "Arithmetic Mean", "Geometric Mean", "Root-Mean-Square"]:
            functional_kwargs = {}
        else:
            sg.PopupError(f"invalid value for functional specie ({functional_specie:s})")

        return functional_kwargs

    def getInequalityFilters(self) -> List[Tuple[str, str, float]]:
        """
        Get filters to apply to results.
        Uses present state of window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve filters from
        """
        inequality_filters = []
        filter_rows: List[FilterRow] = FilterRow.getInstances()

        for filter_row in filter_rows:
            is_active_element = filter_row.getIsActiveElement()
            is_active = self.getValue(getKeys(is_active_element))

            if is_active:
                left_element = filter_row.getLeftVariableElement()
                left_value = self.getValue(getKeys(left_element))

                inequality_element = filter_row.getInequalityTypeElement()
                inequality_type = self.getValue(getKeys(inequality_element))

                right_element = filter_row.getRightFloatElement()
                right_value = self.getValue(getKeys(right_element))

                new_inequality_filter = (left_value, inequality_type, right_value)
                inequality_filters.append(new_inequality_filter)

        return inequality_filters

    def updatePlot(
        self,
        index: Union[tuple, Tuple[int]] = None,
        plot_quantities: PlotQuantities = None,
        transform_name: str = None,
        **figure_kwargs
    ) -> Optional[Figure]:
        """
        Update window-embedded plot.
        Do nothing if simulation has never been run.

        :param index: index of parameter value for free parameter(s)
        :param plot_quantities: object storing info for plot quantities in window.
            See :meth:`~Layout.SimulationWindow.SimulationWindow.PlotQuantities`.
        :param figure_kwargs: additional arguments to pass into :meth:`~SimulationWindow.SimulationWindow.getFigure`
        :returns: New matplotlib Figure displayed on canvas. None if figure has not been displayed yet.
        """
        if index is None:
            index = self.getClosestSliderIndex()
        if plot_quantities is None:
            plot_quantities = self.getPlotQuantities()

        axis_names = self.getAxisNames()
        cartesian_axis_names = self.getAxisNames("cartesian")
        valid_axis_names = plot_quantities.getValidAxisNames()

        results_object = self.getResultsObject()

        axis2names = {}
        axis2species = {}
        axis2coordinate = {}
        axis2transform = {}
        axis2functional = {}
        axis2complex = {}
        is_transformed = {}
        is_timelike = {}
        is_functional = {}
        is_parameterlike = {}
        is_nonelike = {}
        parameter_names = []

        for axis_name in axis_names:
            is_functional[axis_name] = False
            is_timelike[axis_name] = False
            is_parameterlike[axis_name] = False
            is_nonelike[axis_name] = axis_name not in valid_axis_names

        for valid_axis_name in valid_axis_names:
            quantity_names = plot_quantities.getAxisQuantityNames(
                valid_axis_name,
                include_none=False
            )
            specie_names = plot_quantities.getAxisQuantitySpecies(
                valid_axis_name,
                include_none=False
            )
            coordinate_name = plot_quantities.getAxisCoordinateName(valid_axis_name)
            functional_name = plot_quantities.getAxisFunctionalName(valid_axis_name)
            transform_name = plot_quantities.getAxisTransformName(valid_axis_name)
            axis_is_transformed = transform_name != "None"
            complex_name = plot_quantities.getAxisComplexName(valid_axis_name)

            axis2names[valid_axis_name] = quantity_names
            axis2species[valid_axis_name] = specie_names
            axis2coordinate[valid_axis_name] = coordinate_name
            axis2functional[valid_axis_name] = functional_name
            axis2transform[valid_axis_name] = transform_name
            is_transformed[valid_axis_name] = axis_is_transformed
            axis2complex[valid_axis_name] = complex_name

            exists_timelike_subaxis = plot_quantities.existsLikeSpecies(
                specie_names,
                "timelike"
            )
            exists_parameterlike_subaxis = plot_quantities.existsLikeSpecies(
                specie_names,
                "parameterlike"
            )

            if exists_timelike_subaxis and exists_parameterlike_subaxis:
                is_nonelike[valid_axis_name] = True
            elif functional_name != "None":
                is_functional[valid_axis_name] = True
            else:
                if exists_timelike_subaxis and exists_parameterlike_subaxis:
                    is_nonelike[valid_axis_name] = True
                elif exists_timelike_subaxis:
                    is_timelike[valid_axis_name] = True
                elif exists_parameterlike_subaxis:
                    parameterlike_count_subaxis = plot_quantities.getLikeCount(
                        specie_names,
                        "parameterlike"
                    )
                    if parameterlike_count_subaxis == 1:
                        is_parameterlike[valid_axis_name] = True
                        parameter_names.append(quantity_names[0])
                    elif parameterlike_count_subaxis >= 2:
                        is_nonelike[valid_axis_name] = True
                else:
                    is_nonelike[valid_axis_name] = True
        
        timelike_count = sum(is_timelike.values())
        parameterlike_count = sum(is_parameterlike.values())
        functional_count = sum(is_functional.values())

        exists_parameterlike = parameterlike_count >= 1
        multiple_parameterlikes = parameterlike_count >= 2
        exists_cartesian_parameterlike = any([
            is_parameterlike[cartesian_axis_name]
            for cartesian_axis_name in cartesian_axis_names
        ])
        exists_functional = functional_count >= 1
        multiple_functionals = functional_count >= 2
        exists_timelike = timelike_count >= 1
        multiple_timelikes = timelike_count >= 2

        inequality_filters = self.getInequalityFilters()
        print('filters:', inequality_filters)

        results = {}
        try:
            if not exists_parameterlike:
                getResultsOverTime = partial(
                    results_object.getResultsOverTime,
                    inequality_filters=inequality_filters,
                    index=index
                )

                for valid_axis_name, quantity_names in axis2names.items():
                    coordinate_name = axis2coordinate[valid_axis_name]
                    transform_name = axis2transform[valid_axis_name]
                    axis_is_transformed = is_transformed[valid_axis_name]
                    complex_name = axis2complex[valid_axis_name]

                    if not axis_is_transformed or len(quantity_names) == 1:
                        quantity_names = quantity_names[0]

                    if is_timelike[valid_axis_name]:
                        results[valid_axis_name] = getResultsOverTime(
                            quantity_names=quantity_names,
                            coordinate_name=coordinate_name,
                            transform_name=transform_name,
                            complex_name=complex_name
                        )
                    elif is_functional[valid_axis_name]:
                        functional_name = axis2functional[valid_axis_name]
                        functional_kwargs = self.getFunctionalKwargs(functional_name)
                        results[valid_axis_name] = getResultsOverTime(
                            quantity_names=quantity_names,
                            coordinate_name=coordinate_name,
                            transform_name=transform_name,
                            functional_name=functional_name,
                            functional_kwargs=functional_kwargs,
                            complex_name=complex_name
                        )
            else:
                getResultsOverTimePerParameter = partial(
                    results_object.getResultsOverTimePerParameter,
                    parameter_names=parameter_names,
                    inequality_filters=inequality_filters,
                    index=index
                )

                for valid_axis_name, quantity_names in axis2names.items():
                    coordinate_name = axis2coordinate[valid_axis_name]
                    transform_name = axis2transform[valid_axis_name]
                    axis_is_transformed = is_transformed[valid_axis_name]
                    complex_name = axis2complex[valid_axis_name]

                    if is_functional[valid_axis_name]:
                        functional_name = axis2functional[valid_axis_name]
                        functional_kwargs = self.getFunctionalKwargs(functional_name)
                        parameter_results, quantity_results = getResultsOverTimePerParameter(
                            quantity_names=quantity_names,
                            coordinate_name=coordinate_name,
                            transform_name=transform_name,
                            functional_name=functional_name,
                            functional_kwargs=functional_kwargs,
                            complex_name=complex_name
                        )
                        results[valid_axis_name] = quantity_results[0]
                    else:
                        if not axis_is_transformed or len(quantity_names) == 1:
                            quantity_names = quantity_names[0]

                        if is_timelike[valid_axis_name]:
                            parameter_results, quantity_results = getResultsOverTimePerParameter(
                                quantity_names=quantity_names,
                                coordinate_name=coordinate_name,
                                transform_name=transform_name,
                                complex_name=complex_name
                            )
                            results[valid_axis_name] = quantity_results[0]
                        elif is_parameterlike[valid_axis_name]:
                            results[valid_axis_name] = results_object.getFreeParameterValues(names=quantity_names)

        except (UnboundLocalError, KeyError, IndexError, AttributeError, ValueError, AssertionError):
            print('data:', traceback.print_exc())
        except TypeError:
            print("calculation cancelled")

        if is_nonelike['x'] or is_nonelike['y']:
            plot_type = ''
        elif exists_timelike and not multiple_timelikes:
            plot_type = ''
        elif exists_timelike and exists_functional:
            plot_type = ''
        elif exists_functional and not exists_parameterlike:
            plot_type = ''
        else:
            if is_parameterlike['c']:
                if exists_timelike:
                    plot_type = 'nc'
                elif exists_functional:
                    if multiple_parameterlikes:
                        plot_type = 'nc'
                    elif exists_parameterlike:
                        plot_type = 'nt'
            elif is_functional['c']:
                assert exists_cartesian_parameterlike
                if multiple_functionals:
                    if multiple_parameterlikes:
                        plot_type = 't'
                    else:
                        plot_type = 'nt'
                elif exists_functional:
                    plot_type = 'c'
            elif is_timelike['c']:
                plot_type = 'nt'
            elif is_nonelike['c']:
                plot_type = ''

            for cartesian_axis_name in cartesian_axis_names:
                if is_parameterlike[cartesian_axis_name]:
                    if exists_functional:
                        if multiple_parameterlikes:
                            plot_type += f"n{cartesian_axis_name:s}"
                        elif exists_parameterlike:
                            plot_type += cartesian_axis_name
                    elif exists_timelike:
                        plot_type += f"n{cartesian_axis_name:s}"
                elif is_functional[cartesian_axis_name]:
                    if is_parameterlike['c']:
                        plot_type += cartesian_axis_name
                    elif multiple_parameterlikes:
                        plot_type += cartesian_axis_name
                    else:
                        assert exists_parameterlike
                        if multiple_functionals:
                            plot_type += cartesian_axis_name
                        else:
                            plot_type += f"n{cartesian_axis_name:s}"
                elif is_timelike[cartesian_axis_name]:
                    plot_type += cartesian_axis_name
                elif is_nonelike[cartesian_axis_name]:
                    pass

            if is_parameterlike['X'] or is_parameterlike['Y']:
                plot_type += '_'
                for valid_axis_name in ['X', 'Y']:
                    if is_parameterlike[valid_axis_name]:
                        plot_type += valid_axis_name
                    elif is_nonelike[valid_axis_name]:
                        pass

        plot_quantities.reset()

        try:
            print(plot_type, {key: value.shape for key, value in results.items()})
            if plot_type != '':
                figure = self.getFigure(results, plot_type=plot_type, **figure_kwargs)
                self.updateFigureCanvas(figure)
            return figure
        except UnboundLocalError:
            print('todo plots:', plot_quantities, traceback.print_exc())
        except (KeyError, AttributeError, AssertionError):
            print('figure:', traceback.print_exc())

    def updatePlotChoices(self, names: Union[str, List[str]] = None) -> None:
        """
        Update plot choices for desired axis(es).
        This allows user to select new set of quantities to plot.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to change plot choices in
        :param names: name(s) of axis(es) to update choices for.
            Updates all axes by default.
        """
        quantity_count_per_axis = self.getQuantityCountPerAxis()

        # noinspection PyTypeChecker
        window_object: SimulationWindow = self.getWindowObject()

        def update(name: str) -> None:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindowRunner.updatePlotChoices`"""
            axis_quantity_frame: AxisQuantityFrame = AxisQuantityFrame.getInstances(names=name)
            plot_quantities = self.getPlotQuantities()

            species = []
            for index in range(quantity_count_per_axis):
                row_name = name + str(index)
                axis_quantity_row: AxisQuantityRow = AxisQuantityRow.getInstances(names=row_name)
                quantity_names_combobox = axis_quantity_row.getAxisQuantityElement()

                specie = plot_quantities.getAxisQuantitySpecie(name, index)
                species.append(specie)

                choices = self.getPlotChoices(species=specie) if specie != "None" else ['']

                if choices != vars(quantity_names_combobox)["Values"]:
                    quantity_names_combobox_key = getKeys(quantity_names_combobox)

                    kwargs = {
                        "values": choices,
                        "disabled": specie == "None",
                        "size": vars(quantity_names_combobox)["Size"]
                    }

                    old_chosen_quantity = self.getValue(quantity_names_combobox_key)
                    change_quantity = old_chosen_quantity not in choices
                    if change_quantity:
                        new_chosen_quantity = choices[0]
                        kwargs["value"] = new_chosen_quantity
                        quantity_names_combobox.update(**kwargs)
                        window_object.getWindow().write_event_value(quantity_names_combobox_key, new_chosen_quantity)
                    else:
                        quantity_names_combobox.update(**kwargs)

            timelike_species = plot_quantities.getSpecies("timelike") + plot_quantities.getSpecies("nonelike")
            all_species = plot_quantities.getSpecies()
            nontimelike_species = [
                specie
                for specie in all_species
                if specie not in timelike_species
            ]

            exists_nontimelike = False
            for nontimelike_specie in nontimelike_species:
                if nontimelike_specie in species:
                    exists_nontimelike = True
                    break
            coordinate_disabled = transform_disabled = functional_disabled = exists_nontimelike

            functional_combobox = axis_quantity_frame.getAxisFunctionalElement()
            functional_combobox.update(disabled=functional_disabled)

            coordinate_radio_group = axis_quantity_frame.getAxisCoordinateGroup()
            coordinate_radios = coordinate_radio_group.getRadios()
            for coordinate_radio in coordinate_radios:
                coordinate_radio.update(disabled=coordinate_disabled)

            transform_combobox = axis_quantity_frame.getAxisTransformElement()
            transform_combobox.update(disabled=transform_disabled)

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
        folderpath = sg.PopupGetFolder(
            message="Enter Folderpath",
            title="Save Results"
        )

        if isinstance(folderpath, str):
            results_obj = self.getResultsObject()
            results_obj.saveResults(folderpath)

    def saveModelParameters(self) -> None:
        """
        Save parameter values from model into file.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        file_types = (
            *config_file_types,
            ("LaTeX", "*.tex"),
            ("Portable Document Format", "*.pdf"),
            ("ALL Files", "*.*"),
        )

        filepath = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Parameters",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filepath, str):
            model = self.getModel()
            model.saveParametersToFile(filepath)

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

        filepath = sg.PopupGetFile(
            message="Enter Filename",
            title="Save Functions",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filepath, str):
            model = self.getModel()
            model.saveFunctionsToFile(filepath)

    def saveModelVariables(self) -> None:
        """
        Save time-evolution types from model into file.

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
            title="Save Time-Evolution Types",
            save_as=True,
            file_types=file_types
        )

        if isinstance(filename, str):
            model = self.getModel()
            model.saveVariablesToFile(filename)

    def saveFigure(
        self,
        parameter_name: str = None,
        dpi: int = 300
    ) -> None:
        """
        Save displayed figure as image file if name is None.
        Save animation of figures while sweeping over a free parameter if name is str.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve figure from
        :param name: name of parameter to loop over for GIF
        :param dpi: DPI to save figure as
        """
        if parameter_name is None:
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
                    figure.savefig(filepath, dpi=dpi)
                except OSError:
                    sg.PopupError(f"cannot save figure as {filepath:s}")
            elif filepath is None:
                pass
            else:
                sg.PopupError("invalid filepath")
        elif isinstance(parameter_name, str):
            file_types = [
                ("Compressed File", "*.zip"),
                ("ALL Files", "*.*"),
            ]

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

                parameter_index = self.getFreeParameterIndex(parameter_name)
                default_index = list(self.getClosestSliderIndex())
                parameter_values = self.getFreeParameterValues(parameter_name)

                inset_parameters = {
                    parameter_name: {
                        "range": (parameter_values.min(), parameter_values.max())
                    }
                }
                image_count = len(parameter_values)

                updateProgressMeter = partial(
                    self.updateProgressMeter,
                    title="Saving Animation",
                    max_value=image_count
                )
                with ZipFile(zip_filepath, 'w') as zipfile:
                    with imageio.get_writer(gif_filepath, mode='I') as writer:
                        for frame_index in range(image_count):
                            if not updateProgressMeter(frame_index):
                                break

                            data_index = default_index
                            data_index[parameter_index] = frame_index
                            parameter_value = parameter_values[frame_index]
                            inset_parameters[parameter_name]["value"] = parameter_value
                            figure = self.updatePlot(
                                index=tuple(data_index),
                                inset_parameters=inset_parameters
                            )

                            png_filepath = join(save_directory, f"{parameter_name:s}_{frame_index:d}.png")
                            try:
                                figure.savefig(png_filepath, dpi=dpi)
                                writer.append_data(imageio.imread(png_filepath))
                                zipfile.write(png_filepath, basename(png_filepath))
                                os.remove(png_filepath)
                            except AttributeError:
                                pass

                    zipfile.write(gif_filepath, basename(gif_filepath))
                    os.remove(gif_filepath)

                    yaml_parameter_values = list(map(float, parameter_values))
                    yaml_parameter_indicies = list(range(len(parameter_values)))
                    parameter_values_dict = dict(zip(yaml_parameter_indicies, yaml_parameter_values))
                    saveConfig(parameter_values_dict, yaml_filepath)

                    zipfile.write(yaml_filepath, basename(yaml_filepath))
                    os.remove(yaml_filepath)

                    updateProgressMeter(image_count)
            elif zip_filepath is None:
                pass
            else:
                sg.PopupError("invalid filepath")


class PlotQuantities:
    def __init__(self, window_runner: SimulationWindowRunner) -> None:
        self.window_runner = window_runner
        self.getAxisNames = window_runner.getAxisNames
        self.getQuantityCountPerAxis = window_runner.getQuantityCountPerAxis

        self.axis_names = []
        self.axis2names = {}
        self.axis2species = {}
        self.axis2coordinate = {}
        self.axis2transform = {}
        self.axis2functional = {}
        self.axis2complex = {}

        self.specie_types = ["timelike", "parameterlike", "nonelike"]
        self.timelike_species = ["Variable", "Function"]
        self.parameterlike_species = ["Parameter"]
        self.nonelike_species = ["None"]

    def reset(self) -> None:
        """
        Reset attribute in object.

        :param self: :class:`~Layout.SimulationWindowRunner.PlotQuantities` to reset attributes for
        """
        self.axis_names = []
        self.axis2names = {}
        self.axis2species = {}
        self.axis2coordinate = {}
        self.axis2transform = {}
        self.axis2functional = {}
        self.axis2complex = {}

    def getWindowRunner(self) -> SimulationWindowRunner:
        """
        Get window runner associated with self.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve window runner from
        """
        return self.window_runner

    def getSpecies(self, like: str = None) -> List[str]:
        """
        Get collection of quantity species that may be treated as over-time.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve species from
        :param like: species type to retrieve collection of species of.
            Can be "timelike", "parameterlike", "nonelike".
            Defaults to all species.
        """
        if like == "timelike":
            return self.timelike_species
        elif like == "parameterlike":
            return self.parameterlike_species
        elif like == "nonelike":
            return self.nonelike_species
        elif like == None:
            all_species = []
            for specie_type in self.specie_types:
                species_of_type = self.getSpecies(specie_type)
                all_species.extend(species_of_type)
            return all_species
        else:
            raise ValueError("like must be 'timelike', 'parameterlike', or 'nonelike'")

    def getLikeCount(
        self,
        species: List[str],
        like: str
    ):
        """
        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve like-species from
        :param species: names of species to count like-type in
        :param like: type of species to check for in collection of species.
            See :meth:`~Layout.SimulationWindow.PlotQuantities.getSpecies`.
        """
        like_species = self.getSpecies(like)
        like_count = 0
        for specie_name in species:
            if specie_name in like_species:
                like_count += 1

        return like_count

    def existsLikeSpecies(
        self,
        species: List[str],
        like: str
    ):
        """
        Get whether at least one specie is of like-type.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve like-species from
        :param species: names of species to check for like-type in
        :param like: type of species to check for in collection of species.
            See :meth:`~Layout.SimulationWindow.PlotQuantities.getSpecies`.
        """
        like_species = self.getSpecies(like)
        exists_like = False
        for specie_name in species:
            if specie_name in like_species:
                exists_like = True

        return exists_like

    def getValidAxisNames(self) -> List[str]:
        """
        Get names of axes that require plotting a valid quantity.
        Ignore axes that are to be left empty.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve names from
        """
        axis_names = self.getAxisNames()

        valid_axis_names = []
        for axis_name in axis_names:
            specie_names = self.getAxisQuantitySpecies(axis_name)

            specie_is_none = [
                specie_name == "None"
                for specie_name in specie_names
            ]
            none_count = sum(specie_is_none)
            specie_count = len(specie_names)
            nonnone_count = specie_count - none_count

            if nonnone_count >= 1:
                transform_name = self.getAxisTransformName(axis_name)

                if transform_name == "None":
                    if nonnone_count == 1:
                        valid_axis_names.append(axis_name)
                elif transform_name != "None":
                    valid_axis_names.append(axis_name)

        return valid_axis_names

    def getAxisCoordinateName(self, axis_name: str) -> str:
        """
        Get name of coordinate type (e.g. "Cartesian") for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        """
        axis2coordinate = self.axis2coordinate
        if axis_name in axis2coordinate.keys():
            coordinate_name = axis2coordinate[axis_name]
        else:
            exists_nontimelike = self.existsNontimelikeSpecies(axis_name)

            if exists_nontimelike:
                coordinate_name = "Cartesian"
            else:
                coordinate_radio_group: CoordinateRadioGroup = CoordinateRadioGroup.getInstances(names=axis_name)

                chosen_radio = coordinate_radio_group.getChosenRadio()
                chosen_radio_text = vars(chosen_radio)["Text"]
                coordinate_name = chosen_radio_text

            self.axis2coordinate[axis_name] = coordinate_name

        return coordinate_name

    def getAxisFunctionalName(self, axis_name: str) -> str:
        """
        Get name of functional for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        """
        axis2functional = self.axis2functional
        if axis_name in axis2functional.keys():
            functional_name = axis2functional[axis_name]
        else:
            exists_nontimelike = self.existsNontimelikeSpecies(axis_name)

            if exists_nontimelike:
                functional_name = "None"
            else:
                window_runner = self.getWindowRunner()
                axis_quantity_frame: AxisQuantityFrame = AxisQuantityFrame.getInstances(names=axis_name)

                functional_element = axis_quantity_frame.getAxisFunctionalElement()
                functional_key = getKeys(functional_element)
                functional_name = window_runner.getValue(functional_key)

            self.axis2functional[axis_name] = functional_name

        return functional_name

    def getAxisTransformName(self, axis_name: str) -> str:
        """
        Get name of transform for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        """
        axis2transform = self.axis2transform
        if axis_name in axis2transform.keys():
            transform_name = axis2transform[axis_name]
        else:
            exists_nontimelike = self.existsNontimelikeSpecies(axis_name)

            if exists_nontimelike:
                transform_name = "None"
            else:
                window_runner = self.getWindowRunner()
                axis_quantity_frame: AxisQuantityFrame = AxisQuantityFrame.getInstances(names=axis_name)

                transform_element = axis_quantity_frame.getAxisTransformElement()
                transform_key = getKeys(transform_element)
                transform_name = window_runner.getValue(transform_key)

            self.axis2transform[axis_name] = transform_name

        return transform_name

    def getAxisComplexName(self, axis_name: str) -> str:
        """
        Get name of complex-reduction method for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        """
        axis2complex = self.axis2complex
        if axis_name in axis2complex.keys():
            complex_name = axis2complex[axis_name]
        else:
            exists_nontimelike = self.existsNontimelikeSpecies(axis_name)

            if exists_nontimelike:
                complex_name = "Magnitude"
            else:
                complex_radio_group: ComplexRadioGroup = ComplexRadioGroup.getInstances(names=axis_name)

                chosen_radio = complex_radio_group.getChosenRadio()
                chosen_radio_text = vars(chosen_radio)["Text"]
                complex_name = chosen_radio_text

            self.axis2complex[axis_name] = complex_name
        
        return complex_name

    def getAxisQuantityNames(
        self,
        axis_name: str,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected quantity names for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retreive names from
        :param axis_name: name of axis to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        axis2names = self.axis2names
        if axis_name in axis2names.keys():
            quantity_names = axis2names[axis_name]
        else:
            window_runner = self.getWindowRunner()
            quantity_count_per_axis = self.getQuantityCountPerAxis()

            quantity_names = []
            for index in range(quantity_count_per_axis):
                row_name = axis_name + str(index)
                axis_quantity_row: AxisQuantityRow = AxisQuantityRow.getInstances(row_name)

                quantity_element = axis_quantity_row.getAxisQuantityElement()
                quantity_name_key = getKeys(quantity_element)
                quantity_name = window_runner.getValue(quantity_name_key)

                quantity_names.append(quantity_name)

            self.axis2names[axis_name] = quantity_names
            self.getAxisQuantitySpecies(axis_name)

        if not include_none:
            specie_names = self.getAxisQuantitySpecies(axis_name)
            none_indicies = getIndicies("None", specie_names)
            quantity_names = removeAtIndicies(quantity_names, none_indicies)

        return quantity_names

    def getAxisQuantityName(
        self,
        axis_name: str,
        index: int
    ) -> str:
        """
        Get selected quantity name for subaxis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        :param index: index of axis quantity per axis name
        """
        axis2names = self.axis2names
        if axis_name in axis2names.keys():
            quantity_name = axis2names[axis_name][index]
        else:
            quantity_names = self.getAxisQuantityNames(axis_name)
            quantity_name = quantity_names[index]

        return quantity_name

    def existsNontimelikeSpecies(self, axis_name: str) -> bool:
        """
        Get whether axis has some nontimelike species.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve name from
        :param axis_name: name of axis to retrieve name from
        :returns: True if axis contains at least one nontimelike species.
            False if all species are timelike.
        """
        specie_names = self.getAxisQuantitySpecies(axis_name)
        timelike_species = self.getSpecies("timelike") + self.getSpecies("nonelike")

        exists_nontimelike_specie = False
        for specie_name in specie_names:
            if specie_name not in timelike_species:
                exists_nontimelike_specie = True
                break

        return exists_nontimelike_specie

    def getAxisQuantitySpecies(
        self,
        axis_name: str,
        include_none: bool = True
    ) -> List[str]:
        """
        Get selected specie names for axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retreive names from
        :param axis_name: name of axis to retrieve names from
        :param include_none: set True to include "None" species. Set False otherwise.
        """
        axis2species = self.axis2species
        if axis_name in axis2species.keys():
            specie_names = axis2species[axis_name]
        else:
            window_runner = self.getWindowRunner()
            quantity_count_per_axis = self.getQuantityCountPerAxis()

            specie_names = []
            for index in range(quantity_count_per_axis):
                row_name = axis_name + str(index)
                axis_quantity_row: AxisQuantityRow = AxisQuantityRow.getInstances(row_name)

                specie_element = axis_quantity_row.getAxisQuantitySpeciesElement()
                specie_name_key = getKeys(specie_element)
                specie_name = window_runner.getValue(specie_name_key)

                specie_names.append(specie_name)

            self.axis2species[axis_name] = specie_names
            self.getAxisQuantityNames(axis_name)

        if not include_none:
            none_indicies = getIndicies("None", specie_names)
            specie_names = removeAtIndicies(specie_names, none_indicies)

        return specie_names

    def getAxisQuantitySpecie(
        self,
        axis_name: str,
        index: int
    ) -> str:
        """
        Get selected quantity specie for desired axis.

        :param self: :class:`~Layout.SimulationWindow.PlotQuantities` to retrieve quantity name from
        :param axis_name: name of axis to retrieve quantity specie from
        :param index: index of axis quantity per axis name
        """
        axis2specie = self.axis2species
        if axis_name in axis2specie.keys():
            specie_name = axis2specie[axis_name][index]
        else:
            specie_names = self.getAxisQuantitySpecies(axis_name)
            specie_name = specie_names[index]

        return specie_name
