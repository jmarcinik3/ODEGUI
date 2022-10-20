from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import PySimpleGUI as sg
from Function import Model
from macros import recursiveMethod
from numpy import ndarray
from pint import Quantity
from Results import GridResults

from Layout.Layout import Layout, Row
from Layout.SimulationWindow import (ParameterSlider, SimulationWindow, SimulationWindowRunner,
                                     getUnitConversionFactor)


class GridSimulationWindow(SimulationWindow):
    def __init__(
        self,
        free_parameter_name2metadata: Dict[str, Tuple[float, float, int, Quantity]],
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.GridSimulationWindow.GridSimulationWindow`

        :param free_parameter_name2metadata: dictionary of free-parameter values.
            Key is name of free parameter.
            Value is tuple of (minimum, maximum, stepcount, Quantity) for free parameter.
            Leave as empty dictionary if there exist zero free parameters.
        :param kwargs: additional arguments to pass into :class:`~Layout.GridSimulationWindow.GridSimulationWindow`
        """
        assert isinstance(free_parameter_name2metadata, dict)
        for free_parameter_name, free_parameter_metadatas in free_parameter_name2metadata.items():
            assert isinstance(free_parameter_name, str)
            assert isinstance(free_parameter_metadatas, tuple)
            lower_bound, upper_bound, free_parameter_count, quantity = free_parameter_metadatas
            assert isinstance(lower_bound, float)
            assert isinstance(upper_bound, float)
            assert isinstance(free_parameter_count, int)
            assert isinstance(quantity, Quantity)
        self.free_parameter_name2metadata = free_parameter_name2metadata
        free_parameter_names = list(free_parameter_name2metadata.keys())

        SimulationWindow.__init__(
            self,
            free_parameter_names=free_parameter_names,
            **kwargs
        )

    def getFreeParameterName2Metadata(
        self,
        names: str = None
    ) -> Union[Dict[str, Tuple[float, float, int, Quantity]], Tuple[float, float, int, Quantity]]:
        """
        Get stored values for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindow` to retrieve names from
        :param names: name(s) of free parameter(s) to retrieve metadata(s) of
        :returns: Tuple of (minimum, maximum, stepcount, quantity) if name is str.
            Dict of equivalent tuples if name is None; all parameter tuples are returned.
        """
        free_parameter_name2metadata = self.free_parameter_name2metadata

        def get(name: str) -> Tuple[float, float, int, Quantity]:
            free_parameter_metadata = free_parameter_name2metadata[name]
            return free_parameter_metadata

        free_parameter_names = self.getFreeParameterNames()
        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=dict,
            default_args=free_parameter_names
        )

    def getFreeParameterValues(self, name: str) -> ndarray:
        """
        Get possible values for free-parameter slider.
        This corresponds to the values the parameter is simulated at.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindow` to retrieve values from
        :param name: name of parameter to retrieve values
        """
        parameter_slider_obj = self.getParameterSliders(names=name)
        slider_min = parameter_slider_obj.getMinimum()
        slider_max = parameter_slider_obj.getMaximum()
        slider_resolution = parameter_slider_obj.getResolution()
        step_count = round((slider_max - slider_min) / slider_resolution + 1)
        return np.linspace(slider_min, slider_max, step_count)

    def getParameterSliders(
        self,
        names: Union[str, List[str]] = None
    ) -> Union[ParameterSlider, List[ParameterSlider]]:
        """
        Get all parameter slider objects for window.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindow` to retrieve sliders from
        :param names: name(s) of free parameter(s) associated with parameter slider(s).
            Defaults to names of all free parameters.
        """
        try:
            parameter_name2slider = self.parameter_name2slider

            def get(name: str):
                """Base method for :meth:`~Layout.GridSimulationWindow.GridSimulationWindow.getParameterSliders`"""
                return parameter_name2slider[name]

            free_parameter_names = self.getFreeParameterNames()
            return recursiveMethod(
                args=names,
                base_method=get,
                valid_input_types=str,
                output_type=list,
                default_args=free_parameter_names
            )
        except AttributeError:
            parameter_name2value = self.getFreeParameterName2Metadata()
            parameter_name2slider = {}

            for free_parameter_name, free_parameter_value in parameter_name2value.items():
                default_unit = free_parameter_value[3].units
                unit_conversion_factor = getUnitConversionFactor(default_unit)
                minimum_value = float(free_parameter_value[0]) * unit_conversion_factor
                maximum_value = float(free_parameter_value[1]) * unit_conversion_factor
                value_stepcount = int(free_parameter_value[2])
                
                parameter_slider = ParameterSlider(
                    name=free_parameter_name,
                    window=self,
                    minimum=minimum_value,
                    maximum=maximum_value,
                    stepcount=value_stepcount,
                    unit=default_unit
                )
                parameter_name2slider[free_parameter_name] = parameter_slider

            self.parameter_name2slider = parameter_name2slider

            parameter_sliders = self.getParameterSliders(names=names)
            return parameter_sliders

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for grid-simulation window.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindow` to retrieve layout from
        """
        parameter_selection_layout = Layout()
        parameter_slider_objs: List[ParameterSlider] = self.getParameterSliders()
        for parameter_slider_obj in parameter_slider_objs:
            parameter_slider = parameter_slider_obj.getAsColumn()
            parameter_slider_row = Row(
                window=self,
                elements=parameter_slider
            )
            parameter_selection_layout.addRows(parameter_slider_row)

        layout = self.getBaseLayout(parameter_selection_layout)
        return layout


class GridSimulationWindowRunner(
    GridSimulationWindow,
    SimulationWindowRunner
):
    def __init__(
        self,
        name: str,
        free_parameter_name2metadata: Dict[str, Tuple[float, float, int, Quantity]],
        model: Model = None,
        results_obj: GridResults = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.GridSimulationWindow.GridSimulationWindowRunner`.

        :param name: title of window
        :param model: :paramref:`~Layout.GridSimulationWindow.GridSimulationWindowRunner.model`
        :param results_obj: :paramref:`~Layout.GridSimulationWindow.GridSimulationWindowRunner.results_obj`
        :param free_parameter_name2metadata: see :paramref:`~Layout.GridSimulationWindow.GridSimulationWindow.free_parameter_name2metadata`
        :param kwargs: additional arguments to pass into :class:`~Layout.GridSimulationWindow.GridSimulationWindow`
        """
        if results_obj is not None:
            assert isinstance(results_obj, GridResults)

        GridSimulationWindow.__init__(
            self,
            name=name,
            runner=self,
            free_parameter_name2metadata=free_parameter_name2metadata,
            **kwargs
        )
        
        axis_quantity_frames = self.getAxisQuantityFrames()
        SimulationWindowRunner.__init__(
            self,
            model=model,
            results_obj=results_obj,
            axis_quantity_frames=axis_quantity_frames
        )

    def runWindow(self) -> None:
        window = self.getWindow()
        while True:
            event, self.values = window.read()
            print('event:', event)

            if event == sg.WIN_CLOSED or event == "Exit":
                break

            self.runSimulationWindow(event)

        window.close()

    def getParameterIndex(self) -> Union[int, Tuple[int, ...]]:
        """
        Get index for set of parameters (e.g. to retrieve data for).

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindowRunner` to retrieve index from
        """
        index = self.getChosenSliderIndicies()
        return index

    def getChosenSliderIndicies(
        self,
        names: Union[str, List[str]] = None
    ) -> Union[int, Tuple[int, ...]]:
        """
        Get location/index of slider closest to value of free parameter.
        Location is discretized from zero to the number of free-parameter values.
        Uses present state of :class:`~Layout.GridSimulationWindow.GridSimulationWindowRunner`.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindowRunner` to retrieve slider from
        :param names: name(s) of parameter(s) associated with slider.
            Defaults to names of all free parameters.
        :returns: Slider index for free parameter if names is str.
            Tuple of slider indicies for all given free parameters if names is list or tuple.
        """

        def get(name: str) -> int:
            """Base method for :meth:`~Layout.GridSimulationWindow.GridSimulationWindowRunner.getClosestSliderIndex`"""
            parameter_slider_obj = self.getParameterSliders(names=name)
            slider_value = parameter_slider_obj.getSliderValue()
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

    def getResultsObject(self) -> GridResults:
        """
        Get stored :class:`~Results.GridResults`.

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
                model = self.getModel()
                free_parameter_names = self.getFreeParameterNames()
                free_parameter_name2values = {
                    free_parameter_name: self.getFreeParameterValues(free_parameter_name)
                    for free_parameter_name in free_parameter_names
                }

                results_obj = GridResults(
                    model=model,
                    free_parameter_name2values=free_parameter_name2values,
                    folderpath=save_folderpath,
                    stepcount=stepcount
                )

            if isinstance(results_obj, GridResults):
                self.results_obj = results_obj

        return results_obj
