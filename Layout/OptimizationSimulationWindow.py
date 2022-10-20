from __future__ import annotations

from os.path import isfile
from typing import Dict, List, Tuple, Union

import numpy as np
import PySimpleGUI as sg
from Function import Model
from macros import recursiveMethod
from numpy import ndarray
from pint import Quantity
from Results import OptimizationResults, OptimizationResultsFileHandler

from Layout.AxisQuantity import AxisQuantity, AxisQuantityMetadata
from Layout.FitAnalysisWindow import FitAnalysisWindowRunner
from Layout.Layout import Layout, Row, getKeys
from Layout.SimulationWindow import SimulationWindow, SimulationWindowRunner
from Layout.SimulationWindowElement import ParameterSelectionSection


class OptimizationSimulationWindow(SimulationWindow):
    def __init__(
        self,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        fit_axis_quantity: AxisQuantity,
        fitdata_filepath: str,
        sample_count_per_sample_sizes: Dict[int, int],
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow`

        :param free_parameter_name2quantity: dictionary of free-parameter quantity objects.
            Key is name of parameter.
            Value is quantity object associated with parameter.
        :param fit_parameter_names: names of fit parameters
        :param fit_parameter_metadata: object indicating how to generate output vector from ODE
        :param fitdata_filepath: filepath for file containing experimental data vectors,
            with parameters ordered same as names in list of fit-parameter names.
        :param kwargs: additional arguments to pass into :class:`~Layout.OptimizationSimulationWindow.SimulationWindow`
        """
        assert isinstance(free_parameter_name2quantity, dict)
        for free_parameter_name, free_parameter_quantity in free_parameter_name2quantity.items():
            assert isinstance(free_parameter_name, str)
            assert isinstance(free_parameter_quantity, Quantity)
        self.free_parameter_name2quantity = free_parameter_name2quantity
        free_parameter_names = list(free_parameter_name2quantity.keys())

        assert isinstance(fit_parameter_names, list)
        assert len(fit_parameter_names) >= 1
        for fit_parameter_name in fit_parameter_names:
            assert isinstance(fit_parameter_name, str)

        SimulationWindow.__init__(
            self,
            free_parameter_names=free_parameter_names,
            fit_parameter_names=fit_parameter_names,
            **kwargs
        )

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

    def getFitAxisQuantity(self) -> AxisQuantity:
        """
        Get object indicating how to generate experimental data from ODE simulation.

        :param self: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` to retrieve axis-quantity from
        """
        return self.fit_axis_quantity

    def getFitdataFilepath(self) -> str:
        """
        Get filepath of file with experimental data to fit ODE.

        :param self: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` to retrieve filepath from
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
        Get layout for optimization-simulation window.

        :param self: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow` to retrieve layout from
        """
        layout_obj = Layout()

        parameter_selection_section = self.getParameterSelectionSection()
        parameter_selection_column = parameter_selection_section.getElement()
        parameter_selection_row = Row(elements=parameter_selection_column)
        layout_obj.addRows(parameter_selection_row)

        fit_analysis_button = sg.Button("Fit Analysis")
        fit_analysis_row = Row(elements=fit_analysis_button)
        layout_obj.addRows(fit_analysis_row)

        layout = self.getBaseLayout(parameter_selection_layout_obj=layout_obj)
        return layout


class OptimizationSimulationWindowRunner(
    OptimizationSimulationWindow,
    SimulationWindowRunner
):
    def __init__(
        self,
        name: str,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_names: List[str],
        fit_axis_quantity: AxisQuantity,
        fitdata_filepath: str,
        sample_sizes: Tuple[int, ...] = None,
        model: Model = None,
        results_obj: OptimizationResults = None,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindowRunner`.

        :param name: title of window
        :param free_parameter_name2quantity: see :paramref:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow.free_parameter_name2quantity`
        :param fit_parameter_names: see :paramref:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow.fit_parameter_names`
        :param fit_axis_quantity: see :paramref:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow.fit_axis_quantity`
        :param fit_data_filepath: see :paramref:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow.fit_data_filepath`
        :param model: :paramref:`~Layout.OptimizationSimulationWindow.SimulationWindowRunner.model`
        :param results_obj: :paramref:`~Layout.OptimizationSimulationWindow.SimulationWindowRunner.results_obj`
        :param kwargs: additional arguments to pass into :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindow`
        """
        if results_obj is not None:
            assert isinstance(results_obj, OptimizationResults)

        if sample_sizes is None:
            fit_data: ndarray = np.load(fitdata_filepath)
            fit_data_count = fit_data.shape[-1]
            sample_sizes = list(np.logspace(
                0,
                np.log10(fit_data_count),
                3,
                endpoint=True,
                dtype=np.int32
            ))
            sample_sizes = tuple(list(map(int, sample_sizes)))
        else:
            assert isinstance(sample_sizes, tuple)

        sample_count_per_sample_sizes = {}
        for sample_size in sample_sizes:
            index = (sample_size, 0, 0)

            try:
                results_file_handler: OptimizationResultsFileHandler = results_obj.getResultsFileHandler()
                time_results_file = results_file_handler.getResultsFile(
                    quantity_name='t',
                    index=index
                )
                time_results = time_results_file['t']
                time_results_shape = time_results.shape
                sample_count_per_sample_size = time_results_shape[0]
                results_file_handler.closeResultsFiles()
            except:
                sample_count_per_sample_size = 1

            sample_count_per_sample_sizes[sample_size] = sample_count_per_sample_size

        OptimizationSimulationWindow.__init__(
            self,
            name=name,
            runner=self,
            free_parameter_name2quantity=free_parameter_name2quantity,
            fit_parameter_names=fit_parameter_names,
            fit_axis_quantity=fit_axis_quantity,
            fitdata_filepath=fitdata_filepath,
            sample_count_per_sample_sizes=sample_count_per_sample_sizes,
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
            self.runSimulationWindow(event)

            if event in (sample_index_slider_key, simulation_index_slider_key):
                self.updatePlot()
            elif sample_size_radio_group_id in event:
                parameter_selection_section.changedSampleSizeGroup()
                self.updatePlot()
            elif event == "Fit Analysis":
                free_parameter_name2quantity = self.getFreeParameterName2Quantity()
                fit_parameter_names = self.getFitParameterNames()
                fit_axis_quantity = self.getFitAxisQuantity()
                fitdata_filepath = self.getFitdataFilepath()
                sample_count_per_sample_sizes = self.getSimulationCountPerSampleSize()
                results_obj = self.getResultsObject()

                fit_analysis_window_runner = FitAnalysisWindowRunner(
                    free_parameter_name2quantity=free_parameter_name2quantity,
                    fit_parameter_names=fit_parameter_names,
                    fit_axis_quantity=fit_axis_quantity,
                    fitdata_filepath=fitdata_filepath,
                    sample_count_per_sample_sizes=sample_count_per_sample_sizes,
                    results_obj=results_obj
                )
                fit_analysis_window_runner.runWindow()

        window.close()

    def getResultsObject(self) -> OptimizationResults:
        """
        Get stored :class:`~Results.OptimizationResults`.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to retrieve object from
        """
        results_obj = self.results_obj

        if not isinstance(results_obj, OptimizationResults):
            save_folderpath = sg.PopupGetFolder(
                message="Enter Folder to Save Into",
                title="Run Simulation"
            )
            if save_folderpath is not None:
                stepcount = self.getStepCount()
                model = self.getModel()
                fit_axis_quantity = self.getFitAxisQuantity()
                fitdata_filepath = self.getFitdataFilepath()
                fit_parameter_names = self.getFitParameterNames()
                free_parameter_name2quantity = self.getFreeParameterName2Quantity()

                results_obj = OptimizationResults(
                    model=model,
                    folderpath=save_folderpath,
                    stepcount=stepcount,
                    free_parameter_name2quantity=free_parameter_name2quantity,
                    fit_parameter_names=fit_parameter_names,
                    fitdata_filepath=fitdata_filepath,
                    fit_axis_quantity_metadata=fit_axis_quantity
                )

                results_obj.saveResultsMetadata()
                results_obj.saveFitData()

            if isinstance(results_obj, OptimizationResults):
                self.results_obj = results_obj

        return results_obj

    def getParameterIndex(self) -> Tuple[int, int]:
        """
        Get index for set of parameters (e.g. to retrieve data for).

        :param self: :class:`~Layout.OptimizationSimulationWindow.OptimizationSimulationWindowRunner` to retrieve index from
        """
        parameter_selection_section = self.getParameterSelectionSection()
        index = parameter_selection_section.getParameterIndex()
        return index
