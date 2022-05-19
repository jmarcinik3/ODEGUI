from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from pint import Quantity

from CustomErrors import RecursiveTypeError
from Layout.Layout import Layout, Row, Window, WindowRunner
from macros import formatQuantity, getTexImage


class SetParameterRow(Row):
    def __init__(self, name: str, window: SetFreeParametersWindow, quantity: Quantity) -> None:
        super().__init__(name=name, window=window)

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
        return self.quantity

    def getNameLabel(self) -> Union[sg.Image, sg.Text]:
        return getTexImage(self.getName())

    def getQuantityLabel(self) -> sg.Text:
        return sg.Text(
            text=formatQuantity(self.getQuantity()),
            tooltip="Present value of parameter",
            size=(10, 1),  # dim
            key=self.getKey("quantity_label", self.getName())
        )

    def getMinimumInputElement(self) -> sg.InputText:
        name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter minimum value for parameter",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_minimum", name)
        )

    def getMaximumInputElement(self) -> sg.InputText:
        name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter maximum value for parameter",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_maximum", name)
        )

    def getStepCountInputElement(self) -> sg.InputText:
        name = self.getName()

        return sg.InputText(
            default_text='',
            tooltip="Enter number of parameter values",
            size=(10, 1),  # dim
            key=self.getKey("free_parameter_stepcount", name)
        )


class SetFreeParametersWindow(Window):
    def __init__(
        self,
        name: str,
        runner: SetFreeParametersWindowRunner,
        free_parameter_quantities: Dict[str, Quantity]
    ) -> None:
        dimensions = {
            "window": (600, 700),
        }
        super().__init__(name, runner, dimensions=dimensions)

        self.free_parameter_quantities = free_parameter_quantities

    def getFreeParameterQuantity(self, name: str):
        """
        Get stored name(s) for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve names from
        :param name: name of parameter to retrieve quantity of
        """
        return self.free_parameter_quantities[name]

    def getFreeParameterNames(self, indicies: Union[int, Iterable[int]] = None) -> Union[str, Iterable[str]]:
        """
        Get stored name(s) for free parameter(s).
        The user may change the values of these parameters during simulation.

        :param self: :class:`~Layout.SetFreeParametersWindow.SetFreeParametersWindow` to retrieve names from
        :param indicies: location(s) to retrieve parameter name(s) from in collection of names
        """
        if isinstance(indicies, int):
            free_parameter_names = self.getFreeParameterNames()
            return free_parameter_names[indicies]
        elif isinstance(indicies, list):
            return [self.getFreeParameterNames(indicies=index) for index in indicies]
        elif indicies is None:
            return list(self.free_parameter_quantities.keys())
        else:
            raise TypeError("index must be int")

    def getHeaderRow(self) -> Row:
        texts = []
        for text in ["Name", "Default Value", "Minimum", "Maximum", "Stepcount"]:
            text_element = sg.Text(
                text=text,
                size=(8, 1)  # dim
            )
            texts.append(text_element)
        return Row(window=self, elements=texts)

    def getFreeParameterRows(self) -> List[Row]:
        free_parameter_rows = []
        for free_parameter_name in self.getFreeParameterNames():
            new_row = SetParameterRow(free_parameter_name, self, self.getFreeParameterQuantity(free_parameter_name))
            free_parameter_rows.append(new_row)
        return free_parameter_rows

    def getLayout(self) -> List[List[sg.Element]]:
        submit_button = sg.Submit()
        cancel_button = sg.Cancel()

        header_row = self.getHeaderRow()
        free_parameter_rows = self.getFreeParameterRows()

        layout = Layout(rows=header_row)
        layout.addRows(free_parameter_rows)
        layout.addRows(Row(elements=[submit_button, cancel_button]))
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

        self.values = None

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
            return self.getFreeParameterValues(names=self.getFreeParameterNames())
        else:
            raise RecursiveTypeError(names)

    def runWindow(self) -> Tuple[str, Dict[str, List[float, float, int]]]:
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_count = len(free_parameter_names)

        if free_parameter_count >= 1:
            window = self.getWindow()
            event = ''
            while event not in [sg.WIN_CLOSED, "Cancel"]:
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
