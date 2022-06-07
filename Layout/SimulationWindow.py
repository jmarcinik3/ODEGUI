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
from zipfile import ZipFile

import dill
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from CustomFigure import getFigure
from Function import Model
from macros import (StoredObject, formatValue, getTexImage, recursiveMethod,
                    unique)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import ndarray
from pint import Quantity, Unit
from Results import GridResults
from Simulation import RunGridSimulation
from sympy.core import function
from YML import (config_file_types, getDimensions, getStates, loadConfig,
                 saveConfig)

from Layout.AxisQuantity import (AxisQuantity, AxisQuantityTabGroup,
                                 ComplexRadioGroup, EnvelopeRadioGroup,
                                 PlotQuantities, cc_pre, ccs_pre, fps_pre)
from Layout.Layout import (Element, Layout, Row, Tab, TabbedWindow, TabGroup,
                           WindowRunner, getKeys, storeElement)


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
    distinct_values = step_count - 1
    if distinct_values == 0:
        return 0
    elif distinct_values >= 1:
        range_values = maximum - minimum
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


class ParameterSlider(Element):
    """
    Slider to choose value for free parameter.
    This contains
        # . Four labels. One for parameter name. One for minimum parameter value. One for maximum parameter value.
        One for number of distinct parameter values
        # . Slider. This allows the user to choose which parameter value to plot a simulation for.

    :ivar name: name of parameter
    :ivar minimum: minimum value of parameter
    :ivar maximum: maximum value of parameter
    :ivar stepcount: number of disticnt parameter values
    :ivar units: units of parameter
    """

    def __init__(
        self,
        name: str,
        window: SimulationWindow,
        values: Tuple[float, float, int, Unit]
    ) -> None:
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
        Element.__init__(
            self,
            window,
            name=name
        )

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

        :param self: :class:`~Layout.SimulationWindow.ParameterSlider` to retrieve resolution from
        """
        minimum = self.getMinimum()
        maximum = self.getMaximum()
        stepcount = self.getStepCount()
        resolution = calculateResolution(minimum, maximum, stepcount)
        return resolution

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
        resolution = self.getResolution()
        name = self.getName()

        return sg.Slider(
            range=(minimum, maximum),
            default_value=minimum,
            resolution=resolution,
            orientation="horizontal",
            enable_events=True,
            size=self.getDimensions(name="parameter_slider_slider"),
            border_width=0,
            pad=(0, 0),
            key=f"-{fps_pre:s} {name:s}-"
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
        # . Input fields to set time steps.
        This allows the user to set the minimum, maximum, and number of steps for time in the simulation.
        # . Run button. This allows the user to run the simulation.
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


class ColorbarAestheticsTab(Tab):
    """
    This class contains the layout for the aesthetics tab in the simulation window.
        # . Header row to identify purpose for each column
        # . Input fields to set lower and upper limit for colorbar
        # . Checkbox to choose whether colorbar is autoscaled or manually scaled
        # . Combobox to choose colobar scale type (e.g. linear, logarithmic)
        # . Spin to set scale factor for colorbar
        # . Combobox to choose colorbar colormap
        # . Spin to set segment count for colormap
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.ColorbarAestheticsTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(
            self,
            name=name,
            window=window
        )

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the purpose of each input column.

        :param self: :class:`~Layout.SimulationWindow.ColorbarAestheticsTab` to retrieve row for
        """
        window_obj = self.getWindowObject()

        top_row = Row(window=window_obj)
        texts = ["Limits"]
        dimension_keys = [f"axis_header_row_{string:s}" for string in ["limits"]]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="center"
            )
            top_row.addElements(text_element)

        bottom_row = Row(window=window_obj)
        texts = ["Title", "Lower", "Upper", "Auto"]
        dimension_keys = [
            f"axis_header_row_{string:s}"
            for string in
            ["element_name", "element_title", "lower_limit", "upper_limit", "autoscale"]
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
            self.getColormapInputElement(),
            self.getSegmentCountElement(),
            self.getLocationElement()
        ]

        layout = Layout(rows=header_rows)
        layout.addRows(Row(window=self.getWindowObject(), elements=row_elements))
        return layout.getLayout()


class AxisAestheticsTab(Tab):
    """
    This class contains the layout for the aesthetics tab in the simulation window.
        # . Header row to identify functions for input
        # . Axis name label for each axis to identify which axis input affects
        # . Input fields to set lower and upper limit for each axis. Two fields per axis
        # . Checkbox to choose whether each axis is autoscaled or manually determined
        # . Combobox to choose each axis scale type (e.g. linear, logarithmic)
        # . Spin to set scale factor for each axis
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AxisAestheticsTab`.

        :param name: name of tab
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tab is stored in.
        """
        Tab.__init__(
            self,
            name=name,
            window=window
        )

    def getHeaderRows(self) -> List[Row]:
        """
        Get row that labels the function of each input column.

        :param self: :class:`~Layout.SimulationWindow.AxisAestheticsTab` to retrieve row for
        """
        window_obj = self.getWindowObject()

        top_row = Row(window=window_obj)
        texts = ["", "Limits"]
        dimension_keys = [f"axis_header_row_{string:s}" for string in ["element", "limits"]]
        for index in range(len(texts)):
            text_element = sg.Text(
                text=texts[index],
                size=self.getDimensions(name=dimension_keys[index]),
                justification="center"
            )
            top_row.addElements(text_element)

        bottom_row = Row(window=window_obj)
        texts = ["Element", "Title", "Lower", "Upper", "Auto", "Type"]
        dimension_keys = [
            f"axis_header_row_{string:s}"
            for string in
            ["element_name", "element_title", "lower_limit", "upper_limit", "autoscale", "scale_type"]
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
            scale_type_input = self.getScaleTypeInputElement(name)
            row.addElements([lowerlimit_input, upperlimit_input, autoscale_input, scale_type_input])
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
        # . :class:`~Layout.SimulationWindow.AxisAestheticsTab`
        # . :class:`~Layout.SimulationWindow.ColorbarAestheticsTab`
    """

    def __init__(self, name: str, window: SimulationWindow) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.AestheticTabGroup`.

        :param name: name of tabgroup
        :param window: :class:`~Layout.SimulationWindow.SimulationWindow` that tabgroup is stored in.
        """
        axis_aesthetics_tab_name = "Axis"
        axis_aesthetics_tab = AxisAestheticsTab("Axis", window)
        self.getAxisAestheticsTab = partial(self.getTabs, names=axis_aesthetics_tab_name)

        colorbar_aesthetics_tab_name = "Colorbar"
        colorbar_aesthetics_tab = ColorbarAestheticsTab("Colorbar", window)
        self.getColorbarAestheticsTab = partial(self.getTabs, names=colorbar_aesthetics_tab_name)

        tabs = [
            axis_aesthetics_tab,
            colorbar_aesthetics_tab
        ]
        super().__init__(tabs, name=name)


class FilterTab(Tab):
    """
    This class contains the layout for the filter tab in the simulation window.
        # . One or more :class:`~Layout.SimulationWindow.FilterRow`, each comprising one node of AND-gate filter

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
        # . Combobox to select left side of inequality (variable name)
        # . Combobox to select type of inequality (e.g. '>', '<=')
        # . Input field to type into  right side of inequality (float)
        # . Checkbox to set whether filter is active or not.

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
            enable_events=False,
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
            enable_events=False,
            size=self.getDimensions(name=f"frequency_functional_combobox"),
            key=f"ANALYSIS FUNCTIONAL FREQUENCY"
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
        # . Simulation tab. This tab allows the user to run the simulation and display desired results.
        # . Aesthetics tab. This tab allows the user to set plot aesthetic for the displayed figure.
        # . Analysis tabgroup. This tabgroup allows the user to determine how to execute analysis for plot.

    :ivar plot_choices: name(s) of variable(s) and/or function(s) that the user may choose to plot
    :ivar free_parameters: name(s) of parameter(s) that the user may choose multiple values for in model
    """

    def __init__(
        self,
        name: str,
        runner: SimulationWindowRunner,
        free_parameter_values: Dict[str, Tuple[float, float, int, Quantity]],
        plot_choices: Dict[str, List[str]],
        transform_config_filepath: str = "transforms/transforms.json",
        envelope_config_filepath: str = "transforms/envelopes.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json",
        include_simulation_tab: bool = True
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
        :param envelope_config_filepath: filepath for file containing info about envelope transformations
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
        TabbedWindow.__init__(
            self,
            name,
            runner,
            dimensions=dimensions
        )

        assert isinstance(plot_choices, dict)
        for specie, names in plot_choices.items():
            assert isinstance(specie, str)
            assert isinstance(names, Iterable)
            for name in names:
                assert isinstance(name, str)
        self.plot_choices = plot_choices

        assert isinstance(transform_config_filepath, str)
        transform_config = loadConfig(transform_config_filepath)
        transform_names = list(transform_config.keys())

        assert isinstance(envelope_config_filepath, str)
        envelope_config = loadConfig(envelope_config_filepath)
        envelope_names = list(envelope_config.keys())

        assert isinstance(functional_config_filepath, str)
        functional_config = loadConfig(functional_config_filepath)
        functional_names = list(functional_config.keys())

        assert isinstance(complex_config_filepath, str)
        complex_config = loadConfig(complex_config_filepath)
        complex_names = list(complex_config.keys())

        self.free_parameter_values = free_parameter_values

        plotting_tabgroup_name = "Plotting"
        plotting_tabgroup = AxisQuantityTabGroup(
            plotting_tabgroup_name,
            self,
            transform_names=transform_names,
            envelope_names=envelope_names,
            functional_names=functional_names,
            complex_names=complex_names
        )

        self.getAxisQuantityFrames = plotting_tabgroup.getAxisQuantityFrames

        assert isinstance(include_simulation_tab, bool)
        self.include_simulation_tab = include_simulation_tab
        if include_simulation_tab:
            simulation_tab_name = "Simulation"
            simulation_tab = SimulationTab(simulation_tab_name, self)
            self.getSimulationTab = partial(self.getTabs, names=simulation_tab_name)
            self.addTabs(simulation_tab)

        aesthetics_tabgroup = AestheticsTabGroup("Aesthetics", self)
        aesthetics_tab = aesthetics_tabgroup.getAsTab()
        self.getAxisAestheticsTab = aesthetics_tabgroup.getAxisAestheticsTab
        self.getColorbarAestheticsTab = aesthetics_tabgroup.getColorbarAestheticsTab
        self.addTabs(aesthetics_tab)

        plotting_tab = plotting_tabgroup.getAsTab()
        self.addTabs(plotting_tab)

        analysis_tab = AnalysisTab("Analysis", self)
        self.addTabs(analysis_tab)

        filter_tab = FilterTab("Filter", self, row_count=4)
        self.addTabs(filter_tab)

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

    def getFreeParameterName2Value(
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
            free_parameter_values = self.getFreeParameterName2Value()
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

    def getParameterSliders(
        self,
        names: Union[str, List[str]] = None
    ) -> Union[ParameterSlider, List[ParameterSlider]]:
        """
        Get all parameter slider objects for window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve sliders from
        :param names: name(s) of free parameter(s) associated with parameter slider(s).
            Defaults to names of all free parameters.
        """
        parameter_name2value = self.getFreeParameterName2Value()
        free_parameter_names = list(parameter_name2value.keys())

        try:
            def get(name: str):
                return self.parameter_name2slider[name]

            return recursiveMethod(
                args=names,
                base_method=get,
                valid_input_types=str,
                output_type=list,
                default_args=free_parameter_names
            )
        except AttributeError:
            self.parameter_name2slider = {}
            for free_parameter_name in free_parameter_names:
                free_parameter_value = parameter_name2value[free_parameter_name]

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

                self.parameter_name2slider[free_parameter_name] = parameter_slider

            return self.getParameterSliders(names=names)

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

    def includeSimulationTab(self) -> bool:
        """
        Return whether or not simulation tab is present in window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve tab presence from
        :returns: True if tab is present.
            False if tab is not present.
        """
        return self.include_simulation_tab

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for simulation window.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindow` to retrieve layout from
        """
        menu = self.getMenu()
        canvas = self.getCanvas()
        exit_button = sg.Exit()
        update_plot_button = self.getUpdatePlotButton()

        tabs = self.getTabs()
        tabgroup_obj = TabGroup(tabs)
        tabgroup = tabgroup_obj.getTabGroup()

        parameter_sliders = self.getParameterSliders()
        parameter_slider_rows = [
            Row(window=self, elements=parameter_slider.getElement())
            for parameter_slider in parameter_sliders
        ]

        prefix_layout = Layout(rows=Row(window=self, elements=menu))

        left_layout = Layout()
        left_layout.addRows(Row(window=self, elements=tabgroup))
        left_layout.addRows(Row(window=self, elements=[update_plot_button, exit_button]))

        right_layout = Layout()
        right_layout.addRows(Row(window=self, elements=canvas))
        right_layout.addRows(parameter_slider_rows)

        layout = prefix_layout.getLayout() + [[sg.Column(left_layout.getLayout()), sg.Column(right_layout.getLayout())]]
        return layout


class SimulationWindowRunner(WindowRunner, SimulationWindow):
    """
    This class runs the simulation and displays results.
    This window allows the user to...
        # . Choose which variable/function to plot on each x-axis and y-axis
        # . Choose which value of free parameters to assume for present plot

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
    :ivar getQuantityCountPerAxis: pointer to :meth:`~Layout.SimulationWindow.SimulationWindow.getQuantityCountPerAxis`
    """

    def __init__(
        self,
        name: str,
        model: Model = None,
        results_obj: GridResults = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param name: title of window
        :param model: :class:`~Function.Model` to simulate.
            Only called and required
            if :paramref:`~Layout.SimulationWindow.SimulationWindowRunner.__init__.results` is None.
        :param results_obj: preloaded results to start simulation with
        :param **kwargs: additional arguments to pass into :class:`~Layout.SimulationWindow.SimulationWindow`
        """
        SimulationWindow.__init__(
            self,
            name,
            self,
            **kwargs
        )
        WindowRunner.__init__(self)

        self.axis_names = {
            "contour": ('C', ),
            "grid": ('X', 'Y'),
            "color": ('c'),
            "cartesian": ('x', 'y', 'z')
        }
        self.timelike_species = ["Variable", "Function"]
        self.parameterlike_species = ["Parameter"]
        self.values = None
        self.figure_canvas = None
        self.general_derivative_vector = None

        axis_names = self.getAxisNames()
        axis_quantity_name2frame = {
            axis_name: self.getAxisQuantityFrames(names=axis_name)
            for axis_name in axis_names
        }
        self.plot_quantities = PlotQuantities(axis_quantity_name2frame)

        if results_obj is not None:
            self.model = results_obj.getModel()
            results_obj = results_obj
            self.results_obj = results_obj
        else:
            self.model = model
            self.results_obj = None

    def runWindow(self) -> None:
        """
        Run simulation window.
        This function links each possible event to an action.
        When an event is triggered, its corresponding action is executed.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to run
        """
        window = self.getWindow()

        is_simulation = self.includeSimulationTab()
        if is_simulation:
            simulation_tab_obj: SimulationTab = self.getSimulationTab()
            run_simulation_key = getKeys(simulation_tab_obj.getRunButton())

        toolbar_menu_key = getKeys(self.getMenu())
        update_plot_key = getKeys(self.getUpdatePlotButton())

        window.bind("<F5>", update_plot_key)
        window.bind("<Escape>", "Exit")

        while True:
            event, self.values = window.read()
            print('event:', event)

            if event == sg.WIN_CLOSED or event == "Exit":
                break
            menu_value = self.getValue(toolbar_menu_key)

            if menu_value is not None:
                if menu_value == "Parameters::Save":
                    self.saveModelParameters()
                elif menu_value == "Functions::Save":
                    self.saveModelFunctions()
                elif menu_value == "Results::Save":
                    results_obj = self.getResultsObject()
                    if isinstance(results_obj, GridResults):
                        results_obj.saveResultsMetadata()
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
                    pass  # window.write_event_value(update_plot_key, None)
            elif event == update_plot_key:
                self.updatePlot()
            elif is_simulation and event == run_simulation_key:
                self.runSimulations()
        window.close()

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
            results_obj = self.getResultsObject()
            results_obj.setEquilibriumExpressions(equilibrium_expressions=equilibrium_substitutions)

        return self.general_derivative_vector

    def getStepCount(self) -> int:
        """
        Get number of steps per result.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve stepcount from
        """
        input_times = self.getInputTimes()
        times_stepcount = input_times[2]
        return times_stepcount

    def clearResultsObject(self) -> int:
        """
        Change results object attribute to None.
        Useful to run a new simulation.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to clear from
        """
        self.results_obj = None

    def getResultsObject(self) -> GridResults:
        """
        Get stored :class:`~Results.Results`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve object from
        """
        results_obj = self.results_obj

        if not isinstance(results_obj, GridResults):
            save_folderpath = sg.PopupGetFolder(
                message="Enter Folder to Save Into",
                title="Run Simulation"
            )
            if save_folderpath is not None:
                stepcount = self.getStepCount()
                free_parameter_names = self.getFreeParameterNames()
                free_parameter_values = {
                    free_parameter_name: self.getFreeParameterValues(free_parameter_name)
                    for free_parameter_name in free_parameter_names
                }

                model = self.getModel()

                results_obj = GridResults(
                    model,
                    free_parameter_values,
                    folderpath=save_folderpath,
                    stepcount=stepcount
                )

            if isinstance(results_obj, GridResults):
                self.results_obj = results_obj

        return results_obj

    def getAxisNames(self, filter: str = None) -> List[str]:
        """
        Get names for axes.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve names from
        :param filter: only include axis types that satisfy this filter.
            Must be "cartesian", "grid", or "color".
        :returns: List of axis names (e.g. ['x', 'y', 'c'])
        """

        if filter is None:
            all_names = [
                axis_name
                for axis_names in self.axis_names.values()
                for axis_name in axis_names
            ]
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
        simulation_tab_obj: SimulationTab = self.getSimulationTab()

        initial_time_key = getKeys(simulation_tab_obj.getInitialTimeInputElement())
        final_time_key = getKeys(simulation_tab_obj.getFinalTimeInputElement())
        timestep_count_key = getKeys(simulation_tab_obj.getTimeStepCountInputElement())

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
        parameter_slider_obj = self.getParameterSliders(names=name)
        parameter_slider = parameter_slider_obj.getSlider()
        slider_key = getKeys(parameter_slider)
        slider_value = self.getValue(slider_key)
        return slider_value

    def getSliderMinimum(self, name: str) -> float:
        """
        Get minimum value of parameter slider.
        Uses present state of window.

        :param name: name of parameter associated with slider
        """
        parameter_slider_obj = self.getParameterSliders(names=name)
        parameter_slider = parameter_slider_obj.getSlider()
        slider_minimum = vars(parameter_slider)["Range"][0]
        return slider_minimum

    def getSliderMaximum(self, name: str) -> float:
        """
        Get maximum value of parameter slider.
        Uses present state of window.

        :param name: name of parameter associated with slider
        """
        parameter_slider_obj = self.getParameterSliders(names=name)
        parameter_slider = parameter_slider_obj.getSlider()
        slider_max = vars(parameter_slider)["Range"][1]
        return slider_max

    def getSliderResolution(self, name: str) -> float:
        """
        Get resolution of parameter slider.

        :param name: name of parameter associated with slider
        """
        parameter_slider_obj = self.getParameterSliders(names=name)
        parameter_slider = parameter_slider_obj.getSlider()
        slider_resolution = vars(parameter_slider)["Resolution"]
        return slider_resolution

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

    def runSimulations(
        self,
        show_progress: bool = True
    ) -> None:
        """
        Run simulations for all possible combinations of free-parameter values.
        Save results in :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from and save results in
        :param show_progress: set True to show simulation progress visually;
            Set False to not show progress.
        """
        self.clearResultsObject()
        results_obj = self.getResultsObject()
        if not isinstance(results_obj, GridResults):
            return None

        results_file_handler = results_obj.getResultsFileHandler()
        results_file_handler.deleteResultsFiles()
        results_obj.saveResultsMetadata()

        free_parameter_names = self.getFreeParameterNames()
        model = self.getModel()
        variable_names = model.getVariables(
            time_evolution_types="Temporal",
            return_type=str
        )
        times = np.linspace(*self.getInputTimes())
        initial_value_vector = model.getInitialValues(
            names=variable_names,
            return_type=ndarray
        )
        general_derivative_vector = self.getGeneralDerivativeVector()

        parameter_count = len(free_parameter_names)
        if parameter_count == 0:
            run_simulation = RunGridSimulation(
                index=(),
                parameter_name2value={},
                variable_names=variable_names,
                general_derivative_vector=general_derivative_vector,
                initial_value_vector=initial_value_vector,
                times=times,
                results_file_handler=results_file_handler
            )
            run_simulation.runSimulation()
        elif parameter_count >= 1:
            parameter_name2values = {
                free_parameter_name: self.getFreeParameterValues(free_parameter_name)
                for free_parameter_name in free_parameter_names
            }
            parameter_indicies = [
                range(len(parameter_values))
                for parameter_values in parameter_name2values.values()
            ]
            free_parameter_index_combos = tuple(product(*parameter_indicies))

            if show_progress:
                total_combo_count = len(free_parameter_index_combos)

                updateProgressMeter = partial(
                    self.updateProgressMeter,
                    title="Running Simulation",
                    max_value=total_combo_count
                )

            def getParameterValues(parameter_index):
                parameter_name2value = {}
                for free_parameter_index in range(parameter_count):
                    free_parameter_name = free_parameter_names[free_parameter_index]
                    free_parameter_value = parameter_name2values[free_parameter_name][parameter_index[free_parameter_index]]
                    parameter_name2value[free_parameter_name] = free_parameter_value

                return parameter_name2value

            for simulation_number, parameter_index in enumerate(free_parameter_index_combos):
                if show_progress and not updateProgressMeter(simulation_number):
                    break

                parameter_name2value = getParameterValues(parameter_index)

                run_simulation = RunGridSimulation(
                    index=parameter_index,
                    variable_names=variable_names,
                    general_derivative_vector=general_derivative_vector,
                    parameter_name2value=parameter_name2value,
                    initial_value_vector=initial_value_vector,
                    times=times,
                    results_file_handler=results_file_handler
                )
                run_simulation.runSimulation()

            if show_progress:
                updateProgressMeter(total_combo_count)

        results_file_handler.closeResultsFiles()
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

        grid_axis_names = self.getAxisNames("grid")
        cartesian_axis_names = self.getAxisNames("cartesian")
        colorbar_axis_name = self.getAxisNames("color")[0]

        scale_type_dict = {
            "Linear": "linear",
            "Logarithmic": "log"
        }

        axis_aesthetics_tab = self.getAxisAestheticsTab()
        colorbar_aesthetics_tab = self.getColorbarAestheticsTab()

        scale_type = {}
        scale_factor = {}
        label = {}
        autoscale_on = {}
        lim = {}

        for cartesian_axis_name in cartesian_axis_names:
            axis_quantity_frame = self.getAxisQuantityFrames(names=cartesian_axis_name)
            scale_factor_element = axis_quantity_frame.getScaleFactorElement()
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
            label_element = axis_aesthetics_tab.getTitleInputElement(grid_axis_name)
            label[grid_axis_name] = getValues(label_element)

        axis_quantity_frame = self.getAxisQuantityFrames(names=colorbar_axis_name)
        scale_factor_element = axis_quantity_frame.getScaleFactorElement()
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

        canvas = self.getCanvas()
        self.figure_canvas = drawFigure(canvas.TKCanvas, figure)
        self.getWindow().Refresh()

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
        else:
            functional_kwargs = {}

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
        grid_axis_names = self.getAxisNames("grid")
        valid_axis_names = plot_quantities.getValidAxisNames()

        results_obj = self.getResultsObject()
        if not isinstance(results_obj, GridResults):
            return None
        results_file_handler = results_obj.getResultsFileHandler()

        parameter_names = plot_quantities.getParameterNames()
        parameterlike_count = len(parameter_names)
        exists_parameterlike = parameterlike_count >= 1

        inequality_filters = self.getInequalityFilters()
        print('filters:', inequality_filters)

        results = {}
        try:
            if exists_parameterlike:
                getResults = partial(
                    results_obj.getResultsOverTimePerParameter,
                    show_progress=True,
                    parameter_names=parameter_names
                )
            else:
                getResults = results_obj.getResultsOverTime

            getResults = partial(
                getResults,
                inequality_filters=inequality_filters,
                index=index
            )

            for valid_axis_name in valid_axis_names:
                axis_quantity = plot_quantities.getAxisQuantities(names=valid_axis_name)
                quantity_names = axis_quantity.getQuantityNames(include_none=False)

                axis_is_parameterlike = axis_quantity.isParameterLike()
                axis_is_functionallike = axis_quantity.isFunctionalLike()
                axis_is_timelike = axis_quantity.isTimeLike()

                if axis_is_parameterlike:
                    assert len(quantity_names) == 1
                    quantity_name = quantity_names[0]
                    quantity_results = results_file_handler.getFreeParameterValues(names=quantity_name)
                elif axis_is_timelike or axis_is_functionallike:
                    envelope_name = axis_quantity.getEnvelopeName()
                    transform_name = axis_quantity.getTransformName()
                    is_transformed = axis_quantity.isTransformed()
                    complex_name = axis_quantity.getComplexName()

                    if not is_transformed or len(quantity_names) == 1:
                        quantity_names = quantity_names[0]

                    getAxisResults = partial(
                        getResults,
                        quantity_names=quantity_names,
                        envelope_name=envelope_name,
                        transform_name=transform_name,
                        complex_name=complex_name
                    )

                    parameter_functional_names = axis_quantity.getFunctionalFunctionalNames(include_none=False)
                    if len(parameter_functional_names) >= 1:
                        functional_parameter_namess = axis_quantity.getFunctionalParameterNamess(include_none=False)
                        getAxisResults = partial(
                            getAxisResults,
                            parameter_functional_names=parameter_functional_names,
                            functional_parameter_namess=functional_parameter_namess
                        )

                    if axis_is_functionallike:
                        functional_name = axis_quantity.getFunctionalName()
                        functional_kwargs = self.getFunctionalKwargs(functional_name)
                        getAxisResults = partial(
                            getAxisResults,
                            functional_name=functional_name,
                            functional_kwargs=functional_kwargs
                        )

                    if exists_parameterlike:
                        normalize_parameter_names = axis_quantity.getNormalizeNames()
                        parameter_results, quantity_results = getAxisResults(
                            normalize_names=normalize_parameter_names
                        )
                        quantity_results = quantity_results[0]
                    else:
                        quantity_results = getAxisResults()

                results[valid_axis_name] = quantity_results
        except (UnboundLocalError, KeyError, IndexError, AttributeError, ValueError, AssertionError):
            print('data:', traceback.print_exc())
        except TypeError:
            print('calculation cancelled', traceback.print_exc())

        is_timelike = {}
        is_functionallike = {}
        is_parameterlike = {}
        is_nonelike = {}
        for axis_name in axis_names:
            axis_quantity = plot_quantities.getAxisQuantities(names=axis_name)
            is_parameterlike[axis_name] = axis_quantity.isParameterLike()
            is_functionallike[axis_name] = axis_quantity.isFunctionalLike()
            is_timelike[axis_name] = axis_quantity.isTimeLike()
            is_nonelike[axis_name] = axis_quantity.isNoneLike()

        timelike_count = sum(is_timelike.values())
        exists_timelike = timelike_count >= 1
        multiple_timelikes = timelike_count >= 2
        exists_exactly_one_timelike = exists_timelike and not multiple_timelikes

        functionallike_count = sum(is_functionallike.values())
        exists_functionallike = functionallike_count >= 1

        timelike_and_functionallike = exists_timelike and exists_functionallike
        timelike_nor_functionallike = not exists_timelike and not exists_functionallike
        functionallike_and_not_parameterlike = exists_functionallike and not exists_parameterlike

        x_is_nonelike = is_nonelike['x']
        y_is_nonelike = is_nonelike['y']
        x_or_y_is_nonelike = x_is_nonelike or y_is_nonelike

        if x_or_y_is_nonelike:
            plot_type = ''
        elif exists_exactly_one_timelike:
            plot_type = ''
        elif timelike_and_functionallike:
            plot_type = ''
        elif timelike_nor_functionallike:
            plot_type = ''
        elif functionallike_and_not_parameterlike:
            plot_type = ''
        else:
            multiple_functionals = functionallike_count >= 2

            exists_cartesian_parameterlike = False
            cartesian_axis_quantities: List[AxisQuantity] = plot_quantities.getAxisQuantities(names=cartesian_axis_names)
            for cartesian_axis_quantity in cartesian_axis_quantities:
                cartesian_axis_is_parameterlike = cartesian_axis_quantity.isParameterLike()
                if cartesian_axis_is_parameterlike:
                    exists_cartesian_parameterlike = True
                    break

            grid_parameterlike_count = 0
            grid_axis_quantities: List[AxisQuantity] = plot_quantities.getAxisQuantities(names=grid_axis_names)
            for grid_axis_quantity in grid_axis_quantities:
                grid_axis_is_parameterlike = grid_axis_quantity.isParameterLike()
                if grid_axis_is_parameterlike:
                    grid_parameterlike_count += 1

            exists_grid_parameterlike = grid_parameterlike_count >= 1

            nongrid_parameterlike_count = parameterlike_count - grid_parameterlike_count
            multiple_nongrid_parameterlikes = nongrid_parameterlike_count >= 2

            if is_parameterlike['c']:
                if exists_timelike:
                    plot_type = 'nc'
                elif exists_functionallike:
                    if multiple_nongrid_parameterlikes:
                        plot_type = 'nc'
                    elif exists_parameterlike:
                        plot_type = 'nt'
            elif is_functionallike['c']:
                assert exists_cartesian_parameterlike
                if multiple_functionals:
                    if multiple_nongrid_parameterlikes:
                        plot_type = 't'
                    else:
                        plot_type = 'nt'
                elif exists_functionallike:
                    plot_type = 'c'
            elif is_timelike['c']:
                plot_type = 'nt'
            elif is_nonelike['c']:
                plot_type = ''

            for cartesian_axis_name in cartesian_axis_names:
                if is_parameterlike[cartesian_axis_name]:
                    if exists_functionallike:
                        if multiple_nongrid_parameterlikes:
                            plot_type += f"n{cartesian_axis_name:s}"
                        elif exists_parameterlike:
                            plot_type += cartesian_axis_name
                    elif exists_timelike:
                        plot_type += f"n{cartesian_axis_name:s}"
                elif is_functionallike[cartesian_axis_name]:
                    if is_parameterlike['c']:
                        plot_type += cartesian_axis_name
                    elif multiple_nongrid_parameterlikes:
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

            if is_functionallike['C']:
                plot_type += 'C'

            if exists_grid_parameterlike:
                plot_type += '_'
                for grid_axis_name in grid_axis_names:
                    if is_parameterlike[grid_axis_name]:
                        plot_type += grid_axis_name
                    elif is_nonelike[grid_axis_name]:
                        pass

        with open("results_temp.pkl", 'wb') as file:
            dill.dump(results, file)
        with open("plot_type_temp.pkl", 'wb') as file:
            dill.dump(plot_type, file)

        plot_quantities.reset()
        results_file_handler.closeResultsFiles()

        try:
            print(plot_type, {key: value.shape for key, value in results.items()})
            if plot_type != '':
                figure = self.getFigure(results, plot_type=plot_type, **figure_kwargs)
                self.updateFigureCanvas(figure)
            return figure
        except UnboundLocalError:
            print('todo plots:', vars(plot_quantities), traceback.print_exc())
        except RuntimeError:
            clearFigure(self.getFigureCanvas())
            print('LaTeX failed', traceback.print_exc())
        except (KeyError, AttributeError, AssertionError):
            print('figure:', traceback.print_exc())

    def updatePlotChoices(
        self,
        names: Union[str, List[str]] = None,
    ) -> None:
        """
        Update plot choices for desired axis(es).
        This allows user to select new set of quantities to plot.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to change plot choices in
        :param names: name(s) of axis(es) to update choices for.
            Updates all axes by default.
        """
        getSpecies = PlotQuantities.getSpecies
        timelike_species = getSpecies("timelike") + getSpecies("nonelike")
        all_species = getSpecies()
        nontimelike_species = [
            specie
            for specie in all_species
            if specie not in timelike_species
        ]

        def update(name: str) -> None:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindowRunner.updatePlotChoices`"""
            plot_quantities = self.getPlotQuantities()
            axis_quantity = plot_quantities.getAxisQuantities(names=name)
            axis_quantity_frame = axis_quantity.getAxisQuantityFrame()

            species = axis_quantity.getSpecieNames()
            axis_quantity_rows = axis_quantity_frame.getAxisQuantityRows()
            for index, axis_quantity_row in enumerate(axis_quantity_rows):
                specie = species[index]
                quantity_names_combobox = axis_quantity_row.getAxisQuantityElement()

                new_choices = self.getPlotChoices(species=specie) if specie != "None" else ['']
                old_choices = vars(quantity_names_combobox)["Values"]

                if new_choices != old_choices:
                    quantity_names_combobox_key = getKeys(quantity_names_combobox)

                    kwargs = {
                        "values": new_choices,
                        "disabled": specie == "None",
                        "size": vars(quantity_names_combobox)["Size"]
                    }

                    old_chosen_quantity = self.getValue(quantity_names_combobox_key)
                    change_quantity = old_chosen_quantity not in new_choices
                    if change_quantity:
                        new_chosen_quantity = new_choices[0]
                        kwargs["value"] = new_chosen_quantity
                        quantity_names_combobox.update(**kwargs)
                        self.getWindow().write_event_value(quantity_names_combobox_key, new_chosen_quantity)
                    else:
                        quantity_names_combobox.update(**kwargs)

            exists_nontimelike = False
            for nontimelike_specie in nontimelike_species:
                if nontimelike_specie in species:
                    exists_nontimelike = True
                    break
            envelope_disabled = transform_disabled = functional_disabled = complex_disabled = exists_nontimelike

            envelope_radio_group = axis_quantity_frame.getAxisEnvelopeGroup()
            if isinstance(envelope_radio_group, EnvelopeRadioGroup):
                envelope_radios = envelope_radio_group.getRadios()
                for envelope_radio in envelope_radios:
                    envelope_radio.update(disabled=envelope_disabled)

            transform_combobox = axis_quantity_frame.getAxisTransformElement()
            if isinstance(transform_combobox, sg.Combo):
                transform_combobox.update(disabled=transform_disabled)

            functional_combobox = axis_quantity_frame.getAxisFunctionalElement()
            if isinstance(functional_combobox, sg.Combo):
                functional_combobox.update(disabled=functional_disabled)

            complex_radio_group = axis_quantity_frame.getAxisComplexGroup()
            if isinstance(complex_radio_group, ComplexRadioGroup):
                complex_radios = complex_radio_group.getRadios()
                for complex_radio in complex_radios:
                    complex_radio.update(disabled=complex_disabled)

        return recursiveMethod(
            args=names,
            base_method=update,
            valid_input_types=str,
            default_args=self.getAxisNames()
        )

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
            file_types.append(("Python Pickle", "*.pkl"))
            file_types.append(("ALL Files", "*.*"))

            filepath = sg.PopupGetFile(
                message="Enter Filename",
                title="Save Figure",
                save_as=True,
                file_types=tuple(file_types)
            )

            if isinstance(filepath, str):
                if filepath.endswith(".pkl"):
                    dill.dump(figure, open(filepath, 'wb'))
                else:
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
