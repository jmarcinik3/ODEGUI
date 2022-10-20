from __future__ import annotations

import traceback
from os.path import isfile
from typing import Dict, List, Optional, Tuple, Union

import PySimpleGUI as sg
from Config import getDimensions
from matplotlib.figure import Figure
from pint import Quantity
from Function import Model
from Layout.CanvasWindow import CanvasWindow
from Results import OptimizationResults, Results

from Layout.AxisQuantity import AxisQuantity, AxisQuantityMetadata
from Layout.Layout import (Layout, Row, Window, WindowRunner, getKeys,
                           storeElement)
from Layout.SimulationWindowElement import ParameterSelectionSection
from macros import recursiveMethod


class FitAnalysisWindow(CanvasWindow, Window):
    def __init__(
        self,
        name: str,
        runner: FitAnalysisWindowRunner,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        fit_axis_quantity: AxisQuantity,
        fitdata_filepath: str,
        sample_count_per_sample_sizes: Dict[int, int]
    ) -> None:
        """
        Constructor for :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow`.

        :param name: name of window
        :param runner: window runner associated with window
        :param free_parameter_name2quantity: dictionary of free-parameter quantity objects.
            Key is name of parameter.
            Value is quantity object associated with parameter.
        :param fit_parameter_names: names of fit parameters
        :param fit_parameter_metadata: object indicating how to generate output vector from ODE
        :param fit_data_filepath: filepath for file containing experimental data vectors,
            with parameters ordered same as names in list of fit-parameter names.
        """
        CanvasWindow.__init__(self)

        dimensions = {
            "window": (None, None),  # dim
            "parameter_slider_slider": getDimensions(
                ["simulation_window", "parameter_slider", "slider"]
            )
        }
        Window.__init__(
            self,
            name,
            runner,
            dimensions=dimensions
        )

        assert isinstance(free_parameter_name2quantity, dict)
        for free_parameter_name, free_parameter_quantity in free_parameter_name2quantity.items():
            assert isinstance(free_parameter_name, str)
            assert isinstance(free_parameter_quantity, Quantity)
        self.free_parameter_name2quantity = free_parameter_name2quantity
        free_parameter_names = list(free_parameter_name2quantity.keys())
        self.free_parameter_names = free_parameter_names

        assert isinstance(fit_parameter_names, list)
        assert len(fit_parameter_names) >= 1
        for fit_parameter_name in fit_parameter_names:
            assert isinstance(fit_parameter_name, str)
        self.fit_parameter_names = fit_parameter_names

        assert isinstance(fit_axis_quantity, AxisQuantityMetadata)
        self.fit_axis_quantity = fit_axis_quantity

        assert isinstance(fitdata_filepath, str)
        assert isfile(fitdata_filepath)
        self.fitdata_filepath = fitdata_filepath

        parameter_selection_section = ParameterSelectionSection(
            window=self,
            sample_count_per_sample_sizes=sample_count_per_sample_sizes
        )
        self.getSampleSizes = parameter_selection_section.getSampleSizes
        self.getSimulationCountPerSampleSize = parameter_selection_section.getSampleCountPerSampleSize
        self.parameter_selection_section = parameter_selection_section

    @storeElement
    def getUpdatePlotButton(self) -> sg.Button:
        """
        Get button that allows user to update the plot with new aesthetics.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve element from
        """
        button_text = "Update Plot"

        return sg.Button(
            button_text=button_text,
            tooltip="Click to update plot with new axis settings.",
            key="-UPDATE PLOT-"
        )

    def getFitParameterNames(self) -> List[str]:
        """
        Get names of parameters corresponding to fit-data.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve names from
        """
        return self.fit_parameter_names

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
        free_parameter_names = self.free_parameter_names

        def get(index: int) -> str:
            """Base method for :meth:`~Layout.SimulationWindow.SimulationWindow.getFreeParameterNames`"""
            return free_parameter_names[index]

        free_parameter_count = len(free_parameter_names)
        return recursiveMethod(
            args=indicies,
            base_method=get,
            valid_input_types=int,
            output_type=list,
            default_args=range(free_parameter_count)
        )

    def getFitAxisQuantity(self) -> AxisQuantity:
        """
        Get object indicating how to generate experimental data from ODE simulation.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve axis-quantity from
        """
        return self.fit_axis_quantity

    def getFitdataFilepath(self) -> str:
        """
        Get filepath of file with experimental data to fit ODE.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve filepath from
        """
        return self.fitdata_filepath

    def getFreeParameterName2Quantity(
        self,
        names: str = None
    ) -> Union[Dict[str, Quantity], Quantity]:
        """
        Get stored values for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.GridSimulationWindow.GridSimulationWindow` to retrieve names from
        :param names: name(s) of free parameter(s) to retrieve quantity(s) of
        """
        free_parameter_name2quantity = self.free_parameter_name2quantity

        def get(name: str) -> Quantity:
            """Base method for :meth:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow.getFreeParameterName2Quantity`"""
            free_parameter_quantity = free_parameter_name2quantity[name]
            return free_parameter_quantity

        free_parameter_names = self.getFreeParameterNames()
        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=dict,
            default_args=free_parameter_names
        )

    def getParameterSelectionSection(self) -> ParameterSelectionSection:
        """
        Get selection allowing user to choose index for set of parameters.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve section from
        """
        return self.parameter_selection_section

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for fit-analysis window.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindow` to retrieve layout from
        """
        layout_obj = Layout()

        canvas = self.getCanvas()
        canvas_row = Row(elements=canvas)
        layout_obj.addRows(canvas_row)

        exit_button = sg.Exit()
        update_plot_button = self.getUpdatePlotButton()
        button_row = Row(elements=[update_plot_button, exit_button])
        layout_obj.addRows(button_row)

        parameter_selection_section = self.getParameterSelectionSection()
        parameter_selection_column = parameter_selection_section.getElement()
        parameter_selection_row = Row(elements=parameter_selection_column)
        layout_obj.addRows(parameter_selection_row)

        return layout_obj.getLayout()


class FitAnalysisWindowRunner(WindowRunner, FitAnalysisWindow):
    def __init__(
        self,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        fit_axis_quantity: AxisQuantity,
        fitdata_filepath: str,
        sample_count_per_sample_sizes: Dict[int, int],
        results_obj: OptimizationResults,
        name: str = "Fit Analysis"
    ) -> None:
        WindowRunner.__init__(self)
        FitAnalysisWindow.__init__(
            self,
            name=name,
            runner=self,
            free_parameter_name2quantity=free_parameter_name2quantity,
            fit_parameter_names=fit_parameter_names,
            fit_axis_quantity=fit_axis_quantity,
            fitdata_filepath=fitdata_filepath,
            sample_count_per_sample_sizes=sample_count_per_sample_sizes
        )

        assert isinstance(results_obj, OptimizationResults)
        self.model = results_obj.getModel()
        self.results_obj = results_obj

    def getModel(self) -> Model:
        """
        Get model to simulate.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve model from
        """
        return self.model

    def getResultsObject(self) -> OptimizationResults:
        """
        Get stored :class:`~Results.OptimizationResults`.

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindowRunner` to retrieve object from
        """
        return self.results_obj

    def runWindow(self) -> None:
        window = self.getWindow()
        update_plot_button = self.getUpdatePlotButton()
        update_plot_key = getKeys(update_plot_button)

        window.bind("<F5>", update_plot_key)
        window.bind("<Escape>", "Exit")

        parameter_selection_section = self.getParameterSelectionSection()

        simulation_index_slider_obj = parameter_selection_section.getGroupIndexSlider()
        simulation_index_slider_key = getKeys(simulation_index_slider_obj.getSlider())

        sample_index_slider_obj = parameter_selection_section.getSampleIndexSlider()
        sample_index_slider_key = getKeys(sample_index_slider_obj.getSlider())

        sample_size_radio_group = parameter_selection_section.getSampleSizeGroup()
        sample_size_radio_group_id = sample_size_radio_group.getGroupId()

        while True:
            event, self.values = window.read()
            print('event:', event)

            if event in (sg.WIN_CLOSED, "Exit"):
                break

            if event in (sample_index_slider_key, simulation_index_slider_key):
                self.updatePlot()
            elif sample_size_radio_group_id in event:
                parameter_selection_section.changedSampleSizeGroup()
                self.updatePlot()

        window.close()

    def getParameterIndex(self) -> Tuple[int, int]:
        """
        Get index for set of parameters (e.g. to retrieve data for).

        :param self: :class:`~Layout.FitAnalysisWindow.FitAnalysisWindowRunner` to retrieve index from
        """
        parameter_selection_section = self.getParameterSelectionSection()
        index = parameter_selection_section.getParameterIndex()
        return index

    def updatePlot(
        self,
        index: Union[tuple, Tuple[int]] = None,
        event: str = None,
        **figure_kwargs
    ) -> Optional[Figure]:
        """
        Update window-embedded plot.
        Do nothing if simulation has never been run.

        :param index: index of parameter value for free parameter(s)
        :param figure_kwargs: additional arguments to pass into :meth:`~FitAnalysisWindow.FitAnalysisWindow.getFigure`
        :returns: New matplotlib Figure displayed on canvas. None if figure has not been displayed yet.
        """
        if index is None:
            index = self.getParameterIndex()

        results_obj = self.getResultsObject()
        if not isinstance(results_obj, Results):
            return None
        results_file_handler = results_obj.getResultsFileHandler()


        fit_parameter_names = self.getFitParameterNames()
        fitdata_filepath = self.getFitdataFilepath()


        results_file_handler.closeResultsFiles()

        try:
            fit_axis_quantity = self.getFitAxisQuantity()
            
            figure = None
            self.updateFigureCanvas(figure)
            return figure
        except (KeyError, AttributeError, AssertionError):
            print('figure:', traceback.print_exc())
