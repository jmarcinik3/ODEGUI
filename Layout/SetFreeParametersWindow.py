from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from pint import Quantity

from CustomErrors import RecursiveTypeError
from Layout.Layout import Layout, Row, Window, WindowRunner, storeElement
from macros import formatQuantity, getTexImage, recursiveMethod


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
        fit_parameter_name2quantity: Dict[str, Quantity]
    ) -> None:
        dimensions = {
            "window": (600, 400),
        }
        Window.__init__(
            self,
            name,
            runner,
            dimensions=dimensions
        )

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
    def getChooseParameterElement(self, index: int) -> sg.Combo:
        """
        Get element allowing user to set parameter name changing along each axis of dataset.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParameterWindow` to retrieve element from
        :param index: index of axis corresponding to dataset (e.g. x-axis has index 0, y-axis has index 1)
        """
        free_parameter_names = self.getFreeParameterNames()
        return sg.Combo(
            values=free_parameter_names,
            default_value=free_parameter_names[0],
            tooltip=f"Choose free parameter along axis {index:d} in dataset",
            enable_events=True,
            disabled=False,
            size=(6, 1),  # dim
            key=f"-FREE PARAMETER NAME {index:d}-"
        )

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
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_count = len(free_parameter_names)

        comboboxes = [
            self.getChooseParameterElement(axis_index)
            for axis_index in range(free_parameter_count)
        ]

        row = Row(elements=comboboxes)
        return row

    def getLayout(self) -> List[List[sg.Element]]:
        header_row = self.getHeaderRow()
        layout = Layout(rows=header_row)

        fit_parameter_names = self.getFitParameterNames()
        fit_parameter_count = len(fit_parameter_names)
        free_parameter_names = self.getFreeParameterNames()

        if fit_parameter_count == 0:
            set_parameter_value_rows = self.getSetParameterValuesRows(free_parameter_names)
            layout.addRows(set_parameter_value_rows)
        elif fit_parameter_count >= 1:
            set_parameter_value_rows = self.getSetParameterValuesRows(fit_parameter_names)
            layout.addRows(set_parameter_value_rows)
            choose_free_parameters_row = self.getChooseParametersRow(free_parameter_names)
            layout.addRows(choose_free_parameters_row)

        submit_button = sg.Submit()
        cancel_button = sg.Cancel()
        suffix_elements = [submit_button, cancel_button]
        suffix_row = Row(elements=suffix_elements)
        layout.addRows(suffix_row)

        return layout.getLayout()


class SetFreeParametersWindowRunner(WindowRunner, SetFreeParametersWindow):
    def __init__(self, name: str, **kwargs):
        SetFreeParametersWindow.__init__(
            self,
            name,
            self,
            **kwargs
        )
        WindowRunner.__init__(self)

    def getFreeParameterValues(
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
            values = []
            valid = True

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

            if valid:
                return values
        elif isinstance(names, list):
            values = {}
            for name in names:
                value = self.getFreeParameterValues(names=name)
                if value:
                    values[name] = value
                else:
                    break
            return values
        elif names is None:
            free_parameter_names = self.getFreeParameterNames()
            return self.getFreeParameterValues(names=free_parameter_names)
        else:
            raise RecursiveTypeError(names)

    def runWindow(self) -> Tuple[str, Dict[str, List[float, float, int]]]:
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_count = len(free_parameter_names)

        exit_keys = (sg.WIN_CLOSED, "Cancel")
        if free_parameter_count >= 1:
            window = self.getWindow()
            event = ''
            while event not in exit_keys:
                event, self.values = window.read()
                if event == "Submit":
                    free_parameter_values = self.getFreeParameterValues(names=free_parameter_names)
                    if len(free_parameter_values) == len(free_parameter_names):
                        window.close()
                        return event, free_parameter_values
                else:
                    window.close()
                    return event, {}
        elif free_parameter_count == 0:
            return "Submit", {}
