from __future__ import annotations

from os.path import isfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import PySimpleGUI as sg
from CustomErrors import RecursiveTypeError
from macros import formatQuantity, getTexImage, recursiveMethod
from pint import Quantity
from Config import loadConfig

from Layout.AxisQuantity import (AxisQuantity, AxisQuantityFrame,
                                 AxisQuantityWindowRunner, PlotQuantities, ccs_pre)
from Layout.Layout import (ChooseFileRow, Layout, Row, Window, WindowRunner,
                           getKeys, storeElement)


class SetParameterValuesRow(Row):
    def __init__(
        self,
        name: str,
        window: SetFreeParametersWindow,
        quantity: Quantity
    ) -> None:
        Row.__init__(
            self,
            name=name,
            window=window,
        )

        self.quantity = quantity

        elements = [
            self.getNameLabel(),
            self.getQuantityLabel(),
            self.getMinimumInputElement(),
            self.getMaximumInputElement(),
            self.getStepCountInputElement()
        ]
        self.addElements(elements)

    def getQuantity(self) -> Quantity:
        """
        Get quantity storing value and unit for parameter.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve quantity from
        """
        return self.quantity

    def getNameLabel(self) -> Union[sg.Image, sg.Text]:
        """
        Get tex image of parameter to label name for parameter.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve image from
        """
        parameter_name = self.getName()
        tex_image = getTexImage(parameter_name)
        return tex_image

    def getQuantityLabel(self) -> sg.Text:
        """
        Get text label to show default value and unit for parameter.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve label from
        """
        quantity = self.getQuantity()
        formatted_value = formatQuantity(quantity)
        parameter_name = self.getName()

        return sg.Text(
            text=formatted_value,
            tooltip="Present value of parameter",
            size=(10, 1),  # dim
            key=self.getKey("quantity_label", parameter_name)
        )

    def getMinimumInputElement(self) -> sg.InputText:
        """
        Get element to allow user to set minimum value for free-parameter values.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve element from
        """
        parameter_name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter minimum value for parameter",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_minimum", parameter_name)
        )

    def getMaximumInputElement(self) -> sg.InputText:
        """
        Get element to allow user to set maximum value for free-parameter values.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve element from
        """
        parameter_name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter maximum value for parameter",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_maximum", parameter_name)
        )

    def getStepCountInputElement(self) -> sg.InputText:
        """
        Get element to allow user to set stepcount for free-parameter value.

        :param self: :class:`~Layout.SetFreeParameterWindow.SetParameterValuesRow` to retrieve element from
        """
        parameter_name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter number of parameter values",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_stepcount", parameter_name)
        )


class SetFreeParametersWindow(Window):
    def __init__(
        self,
        name: str,
        runner: SetFreeParametersWindowRunner,
        free_parameter_name2quantity: Dict[str, Quantity],
        fit_parameter_name2quantity: Dict[str, Quantity],
        transform_config_filepath: str = "transforms/transforms.json",
        envelope_config_filepath: str = "transforms/envelopes.json",
        functional_config_filepath: str = "transforms/functionals.json",
        complex_config_filepath: str = "transforms/complexes.json",
        get_plot_choices: Callable[[Optional[str]], List[str]] = None
    ) -> None:
        dimensions = {
            "window": (600, 550),
            "axis_quantity_species_combobox": (10, 10),
            "axis_quantity_combobox": (10, 20),
            "axis_functional_combobox": (24, 10),
            "transform_type_combobox": (24, 10),
            "scale_factor_spin": (6, 1)
        }
        Window.__init__(
            self,
            name,
            runner,
            dimensions=dimensions
        )

        if get_plot_choices is not None:
            assert callable(get_plot_choices)

        parameter_name2quantity = {}
        free_parameter_names = []
        fit_parameter_names = []

        for free_parameter_name, free_parameter_quantity in free_parameter_name2quantity.items():
            parameter_name2quantity[free_parameter_name] = free_parameter_quantity
            free_parameter_names.append(free_parameter_name)

        for fit_parameter_name, fit_parameter_quantity in fit_parameter_name2quantity.items():
            parameter_name2quantity[fit_parameter_name] = fit_parameter_quantity
            fit_parameter_names.append(fit_parameter_name)

        self.parameter_name2quantity = parameter_name2quantity
        self.free_parameter_names = free_parameter_names
        self.fit_parameter_names = fit_parameter_names

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

        self.axis_quantity_frame = AxisQuantityFrame(
            "output",
            window=self,
            get_plot_choices=get_plot_choices,
            quantity_count_per_axis=2,
            include_none=False,
            include_continuous=True,
            include_discrete=True,
            normalize_parameter_names=(),
            include_scalefactor=True,
            transform_names=transform_names,
            envelope_names=envelope_names,
            functional_names=functional_names,
            complex_names=complex_names
        )

        self.getPlotChoices = self.axis_quantity_frame.getPlotChoices

    def getParameterQuantity(self, name: str):
        """
        Get stored name for varying parameter.
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve names from
        :param name: name of parameter to retrieve quantity of
        """
        return self.parameter_name2quantity[name]

    def getFreeParameterNames(
        self,
        indicies: Union[int, Iterable[int]] = None,
    ) -> Union[str, List[str]]:
        """
        Get stored name(s) for free parameter(s).
        These parameter values may change during simulation.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve names from
        :param indicies: location(s) to retrieve parameter name(s) from
        """
        free_parameter_names = self.free_parameter_names

        def get(index: int):
            """Base method for :meth:`~Layout.SetFreeParametersWindow.getFreeParameterNames`"""
            free_parameter_name = free_parameter_names[index]
            return free_parameter_name

        all_indicies = range(len(free_parameter_names))
        return recursiveMethod(
            args=indicies,
            base_method=get,
            valid_input_types=int,
            output_type=list,
            default_args=all_indicies
        )

    def getFitParameterNames(
        self,
        indicies: Union[int, Iterable[int]] = None,
    ) -> Union[str, List[str]]:
        """
        Get stored name(s) for fit parameter(s).
        These parameters will be varied to fit the given dataset.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFitParametersWindow` to retrieve names from
        :param indicies: location(s) to retrieve parameter name(s) from
        """
        fit_parameter_names = self.fit_parameter_names

        def get(index: int):
            """Base method for :meth:`~Layout.SetFreeParametersWindow.getFitParameterNames`"""
            fit_parameter_name = fit_parameter_names[index]
            return fit_parameter_name

        all_indicies = range(len(fit_parameter_names))
        return recursiveMethod(
            args=indicies,
            base_method=get,
            valid_input_types=int,
            output_type=list,
            default_args=all_indicies
        )

    def getHeaderRow(self) -> Row:
        texts = []
        for text in ["Name", "Default Value", "Minimum", "Maximum", "Stepcount"]:
            text_element = sg.Text(
                text=text,
                size=(8, 1)  # dim
            )
            texts.append(text_element)
        return Row(window=self, elements=texts)

    def getSetParameterValuesRows(
        self,
        parameter_names: List[str] = None
    ) -> List[SetParameterValuesRow]:
        """
        Get rows allowing user to set minimum, maximum, and stepcount for parameters.

        :param self: :class:`~Layout.SetFreeParameterWindows.SetFreeParametersWindow` to retrieve rows from
        :param names: names of parameters to retrieve rows for.
            Defaults to names of free parameters.
        """
        if parameter_names is None:
            parameter_names = self.getFreeParameterNames()

        parameter_rows = []
        for parameter_name in parameter_names:
            parameter_quantity = self.getParameterQuantity(parameter_name)
            set_parameter_row = SetParameterValuesRow(
                parameter_name,
                self,
                parameter_quantity,
            )
            parameter_rows.append(set_parameter_row)

        return parameter_rows

    @storeElement
    def getChooseFitParameterElement(self, index: int) -> sg.Combo:
        """
        Get element allowing user to set parameter name changing along each axis of dataset.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParameterWindow` to retrieve element from
        :param index: index of axis corresponding to dataset (e.g. x-axis has index 0, y-axis has index 1)
        """
        fit_parameter_names = self.getFitParameterNames()
        return sg.Combo(
            values=fit_parameter_names,
            default_value=fit_parameter_names[0],
            tooltip=f"Choose free parameter along axis {index:d} in dataset",
            enable_events=False,
            disabled=False,
            size=(6, 1),  # dim
            key=f"-FREE PARAMETER NAME {index:d}-"
        )

    def getAxisQuantityFrame(self) -> AxisQuantityFrame:
        """
        Get axis-quantity frame allowing user to choose output data.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve frame from
        """
        return self.axis_quantity_frame

    @storeElement
    def getChooseDataFileRow(self) -> ChooseFileRow:
        """
        Get row allowing user to choose output-data file.

        :param self :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve row from
        """
        row_key_suffix = "choose_output_data"
        choose_output_data_row = ChooseFileRow(
            name=row_key_suffix,
            window=self
        )
        return choose_output_data_row

    def getChooseParametersRow(
        self,
        parameter_names: List[str] = None
    ) -> Row:
        """
        Get collection of elements, in row, allowing user to order which parameter varies along each axis in dataset.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParameterWindow` to retrieve row from
        :param parameter_names: names of parameters to generate row for.
            Must have same length as number of axes in dataset.
            Defaults to names of free parameters.
        """
        fit_parameter_names = self.getFitParameterNames()
        fit_parameter_count = len(fit_parameter_names)

        comboboxes = [
            self.getChooseFitParameterElement(axis_index)
            for axis_index in range(fit_parameter_count)
        ]

        row = Row(elements=comboboxes)
        return row

    def getLayout(self) -> List[List[sg.Element]]:
        header_row = self.getHeaderRow()
        layout = Layout(rows=header_row)

        fit_parameter_names = self.getFitParameterNames()
        fit_parameter_count = len(fit_parameter_names)
        free_parameter_names = self.getFreeParameterNames()

        set_parameter_value_rows = self.getSetParameterValuesRows(free_parameter_names)
        layout.addRows(set_parameter_value_rows)
        if fit_parameter_count >= 1:
            choose_fit_parameters_row = self.getChooseParametersRow(fit_parameter_names)
            layout.addRows(choose_fit_parameters_row)

            axis_quantity_frame_obj = self.getAxisQuantityFrame()
            axis_quantity_frame = axis_quantity_frame_obj.getFrame()
            frame_row = Row(elements=axis_quantity_frame)
            layout.addRows(frame_row)

            choose_output_data_row = self.getChooseDataFileRow()
            layout.addRows(choose_output_data_row)

        submit_button = sg.Submit()
        cancel_button = sg.Cancel()
        suffix_elements = [submit_button, cancel_button]
        suffix_row = Row(elements=suffix_elements)
        layout.addRows(suffix_row)

        return layout.getLayout()


class SetFreeParametersWindowRunner(
    WindowRunner,
    SetFreeParametersWindow,
    AxisQuantityWindowRunner
):
    def __init__(
        self,
        name: str,
        **kwargs
    ):
        SetFreeParametersWindow.__init__(
            self,
            name,
            self,
            **kwargs
        )
        WindowRunner.__init__(self)

        axis_quantity_frame = self.getAxisQuantityFrame()
        AxisQuantityWindowRunner.__init__(
            self,
            axis_quantity_frames=axis_quantity_frame
        )

        self.event = None
        self.free_parameter_name2range = {}
        self.fit_parameter_names_ordered = []
        self.fitdata_output_filepath = ''

        self.fit_output_axis_quantity = AxisQuantity(axis_quantity_frame)

    def getUpdatedAxisQuantity(self) -> AxisQuantity:
        """
        Get updated axis-quantity object associated with fit-output vector.

        :param self: :class:`~Layout.AxisQuantity.AxisQuantity` to retrieve axis-quantity from
        """
        axis_quantity = self.fit_output_axis_quantity
        axis_quantity.updateAttributes()
        return axis_quantity

    def getFreeParameterName2Range(
        self,
        names: Union[str, Iterable[str]] = None
    ) -> Union[List[float, float, int], Dict[str, List[float, float, int]]]:
        """
        Get info about free parameter values for simulation.
        Uses present state of window.
        Spawns popup error message if values are (1) of incorrect type or (2) missing.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindowRunner` to retrieve values from
        :param names: names of parameters to retrieve info for
        :returns: Dictionary of info for free parameters
            if :paramref:`~Layout.SetFreeParametersWindow.SetFreeParametersWindowRunner.getFreeParameterValues.names`
            is list.
            Key is name of parameter.
            Value is list of (minimum, maximum, stepcount) for parameter.
            Returns only this list
            if :paramref:`~Layout.SetFreeParametersWindow.SetFreeParametersWindowRunner.getFreeParameterValues.names`
            is str.
        """
        if isinstance(names, str):
            fit_parameter_names = self.getFitParameterNames()
            fit_parameter_count = len(fit_parameter_names)
            exists_fit_parameter = fit_parameter_count >= 1

            values = []
            if exists_fit_parameter:
                values = [None, None, None]
            valid = not exists_fit_parameter

            if valid:
                try:
                    minimum_key = self.getKey(f"free_parameter_minimum", names)
                    minimum = float(self.getValue(minimum_key))
                    values.append(minimum)
                except ValueError:
                    valid = False
                    sg.PopupError(f"Input minimum for {names:s}")

            if valid:
                try:
                    maximum_key = self.getKey(f"free_parameter_maximum", names)
                    maximum = float(self.getValue(maximum_key))
                    values.append(maximum)
                except ValueError:
                    valid = False
                    sg.PopupError(f"Input maximum for {names:s}")

            if valid:
                # noinspection PyUnboundLocalVariable
                if maximum <= minimum:
                    valid = False
                    sg.PopupError(f"Maximum must be greater than minimum for {names:s}")

            if valid:
                try:
                    stepcount_key = self.getKey(f"free_parameter_stepcount", names)
                    stepcount = float(self.getValue(stepcount_key))
                    if stepcount >= 2 and stepcount.is_integer():
                        values.append(stepcount)
                    else:
                        valid = False
                        sg.PopupError(f"Stepcount for {names:s} must be integer greater than one")
                except ValueError:
                    valid = False
                    sg.PopupError(f"Input stepcount for {names:s}")

            if valid or exists_fit_parameter:
                return values
        elif isinstance(names, list):
            values = {}
            for name in names:
                value = self.getFreeParameterName2Range(names=name)
                if value is not None:
                    values[name] = value
                else:
                    return None
            return values
        elif names is None:
            free_parameter_names = self.getFreeParameterNames()
            return self.getFreeParameterName2Range(names=free_parameter_names)
        else:
            raise RecursiveTypeError(names)

    def getFitParameterNamesOrdered(self) -> List[str]:
        """
        Get ordered list of fit-parameter names chosen by user.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve names from
        """
        fit_parameter_names = self.getFitParameterNames()
        fit_parameter_count = len(fit_parameter_names)

        fit_parameter_names_ordered = []
        for fit_parameter_index in range(fit_parameter_count):
            fit_parameter_combobox = self.getChooseFitParameterElement(fit_parameter_index)
            fit_parameter_combobox_key = getKeys(fit_parameter_combobox)
            fit_parameter_name_chosen = self.getValue(fit_parameter_combobox_key)

            if fit_parameter_name_chosen not in fit_parameter_names_ordered:
                fit_parameter_names_ordered.append(fit_parameter_name_chosen)
            else:
                sg.PopupError("Choose each fit parameter exactly once")
                return None

        return fit_parameter_names_ordered

    def runWindow(
        self
    ) -> Tuple[str, Dict[str, List[float, float, int]], List[str]]:
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_count = len(free_parameter_names)
        fit_parameter_names = self.getFitParameterNames()
        fit_parameter_count = len(fit_parameter_names)
        parameter_count = free_parameter_count + fit_parameter_count

        axis_quantity_frame = self.getAxisQuantityFrame()
        reset_button = axis_quantity_frame.getResetButton()
        reset_key = getKeys(reset_button)

        exit_keys = (sg.WIN_CLOSED, "Cancel")
        if parameter_count >= 1:
            window = self.getWindow()
            event = ''
            while event not in exit_keys:
                event, self.values = window.read()
                event: str
                self.event = event

                if event in exit_keys:
                    window.close()
                    return self
                elif event == "Submit":
                    free_parameter_name2range = self.getFreeParameterName2Range(names=free_parameter_names)
                    if free_parameter_name2range is not None:
                        self.free_parameter_name2range = free_parameter_name2range
                    else:
                        continue

                    if fit_parameter_count >= 1:
                        fit_parameter_names_ordered = self.getFitParameterNamesOrdered()
                        if fit_parameter_names_ordered is not None:
                            self.fit_parameter_names_ordered = fit_parameter_names_ordered
                        else:
                            continue

                        choose_file_row = self.getChooseDataFileRow()
                        chosen_filepath = choose_file_row.getChosenFilepath()
                        if isfile(chosen_filepath):
                            self.fitdata_output_filepath = chosen_filepath
                        else:
                            continue

                        self.fit_output_axis_quantity = self.getUpdatedAxisQuantity()

                    window.close()
                    return self
                elif event == reset_key:
                    axis_quantity_frame.reset()
                elif ccs_pre in event:
                    # axis_name = event.split(' ')[-1].replace("_AXIS", '').replace('-', '')
                    self.updatePlotChoices()
        elif parameter_count == 0:
            self.event = "Submit"
            return self
