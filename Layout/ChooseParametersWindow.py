"""
This file contains classes relating to the choose-parameters window.
"""
from __future__ import annotations

from os.path import basename
from typing import Dict, Iterable, List, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from pint import Quantity

from Layout.Layout import Layout, Row, Window, WindowRunner
from macros import formatQuantity, getTexImage, recursiveMethod


class ChooseParameterRow(Row):
    """
    This class contains the layout for a parameter row in the choose-parameters window.
        #. Label to indicate name of parameter.
        #. Label to indicate value and unit for parameter.
        #. Checkbox allow user to overwrite (or not) parameter into model.

    :ivar quantity: quantity containing value and unit for parameter
    """

    def __init__(self, name: str, quantity: Quantity, window: ChooseParametersWindow):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParameterRow`.

        :param name: name of parameter
        :param quantity: quantity containing value and unit
        :param window: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` that row is stored in
        """
        super().__init__(name, window=window)
        self.quantity = quantity

        elements = [
            self.getParameterLabel(),
            self.getQuantityLabel(),
            self.getCheckbox()
        ]
        self.addElements(elements)

    def getQuantity(self) -> Quantity:
        """
        Get parameter quantity associated with row.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve quantity from
        """
        return self.quantity

    def getParameterLabel(self) -> Union[sg.Image, sg.Text]:
        """
        Get label for name of parameter.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve label from
        """
        kwargs = {
            "name": self.getName(),
            "size": (110, None)  # dim
        }
        return getTexImage(**kwargs)

    def getQuantityLabel(self) -> sg.Text:
        """
        Get label for parameter value and unit.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve label from
        """
        kwargs = {
            "text": formatQuantity(self.getQuantity()),
            "size": (10, None)  # dim
        }
        return sg.Text(**kwargs)

    def getCheckbox(self) -> sg.Checkbox:
        """
        Get checkbox, allowing user to overwrite (or not) parameter value in model.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve checkbox from
        """
        kwargs = {
            "text": "Overwrite?",
            "default": True,
            "key": self.getName()
        }
        return sg.Checkbox(**kwargs)


class ChooseParametersWindow(Window):
    """
    This class contains the layout for the choose-parameters window.
        #. Menu: This allows user to (un)check all parameters.
        #. Header to indicate purpose of each column.
        #. Footer with submit and cancel buttons.
        #. Row for each parameter, allowing user to choose whether (or not) to overwrite in model.

    :ivar quantites: dictionary of parameter quantities.
        Key is name of parameter.
        Value is quantity containing value and unit for parameter.
    :ivar filename: name of file that parameters were loaded from (optional)
    """

    def __init__(
            self,
            name: str,
            runner: ChooseParametersWindowRunner,
            quantities: Dict[str, Quantity],
            filename: str = ''
    ):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`.

        :param name: name of window
        :param runner: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` that window is stored in
        :param quantities: dictionary of parameter quantities.
            Key is name of parameter.
            Value is quantity containing value and unit for parameter.
        :param filename: name of file that parameters were loaded from
        """
        dimensions = {
            "window": (None, None)  # dim
        }
        super().__init__(name, runner, dimensions=dimensions)
        self.quantities = quantities
        self.filename = filename

    def getFilename(self) -> str:
        """
        Get name of file to load parameters from.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve name from
        """
        return self.filename

    def getParameters(self) -> Dict[str, Quantity]:
        """
        Get parameters stored in window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve parameters from
        :returns: Dictionary of parameter info.
            Key is name of parameter.
            Value is quantity containing value and unit for parameter.
        """
        return self.quantities

    def getParameterRows(self) -> List[ChooseParameterRow]:
        """
        Get rows, each corresponding to a parameter stored in the window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve rows from
        """
        rows = [
            ChooseParameterRow(name, quantity, self)
            for name, quantity in self.getParameters().items()
        ]
        return rows

    def getMenu(self) -> sg.Menu:
        """
        Get toolbar menu for window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve menu from
        """
        menu_definition = [
            [
                "Set",
                [
                    "Check All",
                    "Uncheck All"
                ]
            ]
        ]
        kwargs = {
            "menu_definition": menu_definition,
            "key": self.getKey("toolbar_menu")
        }
        return sg.Menu(**kwargs)

    def getHeaderRow(self) -> Row:
        """
        Get header for window.
        This includes labels, which indicate the purpose of each column in the window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve header from
        """
        filename = basename(self.getFilename())
        text = "Choose which parameters to overwrite"
        if filename != '':
            text += f" ({filename:s})"
        kwargs = {
            "text": text
        }
        row = Row(elements=sg.Text(**kwargs))
        return row

    @staticmethod
    def getFooterRow() -> Row:
        """
        Get footer for window.
        This includes a submit and cancel button.
        """
        submit_button = sg.Submit()
        cancel_button = sg.Cancel()
        row = Row(elements=[submit_button, cancel_button])
        return row

    def getParameterRowsLayout(self) -> Layout:
        """
        Get layout for scrollable section containing row for each parameter.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve section from
        """
        kwargs = {
            "layout": Layout(rows=self.getParameterRows()).getLayout(),
            "size": (None, 350),  # dim
            "scrollable": True,
            "vertical_scroll_only": True
        }
        layout = Layout(rows=Row(elements=sg.Column(**kwargs)))
        return layout

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve layout from
        """
        menu = Layout(rows=Row(elements=self.getMenu()))
        header = self.getHeaderRow()
        footer = self.getFooterRow()
        parameter_rows = self.getParameterRowsLayout()
        return menu.getLayout() + header.getLayout() + parameter_rows.getLayout() + footer.getLayout()


class ChooseParametersWindowRunner(WindowRunner):
    """
    This class runs :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`.
    This window allows the user to...
        #. Choose which parameter(s) to overwrite into model.

    """

    def __init__(self, name: str, **kwargs):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner`.

        :param name: name of window
        """
        window = ChooseParametersWindow(name, self, **kwargs)
        super().__init__(window)

    def getParameterNames(self) -> List[str]:
        """
        Get names of parameters stored in window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to retrieve names from
        """
        # noinspection PyTypeChecker
        window_object: ChooseParametersWindow = self.getWindowObject()
        quantites = window_object.getParameters()
        parameter_names = list(quantites.keys())
        return parameter_names
    
    def setChecks(self, names: Union[str, Iterable[str]], overwrite: bool) -> None:
        """
        Set all checkboxes (determining whether to overwrite parameter) to chosen value.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to set checkboxes in
        :param names: name(s) of parameter(s) to set checkboxes for
        :param overwrite: set True to set all checkboxes to True.
            Set False to set all checkboxes to False.
        """

        def set(name: str) -> None:
            checkbox = self.getElements(name)
            checkbox.update(value=overwrite)

        kwargs = {
            "base_method": set,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
        }
        return recursiveMethod(**kwargs)

    def getChosenParameters(self) -> List[str]:
        """
        Get checked parameters in window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to retrieve checked boxes from
        :returns: List of parameter names, where corresponding checkbox is checked
        """
        window = self.getWindow()
        event = ''
        while event not in [sg.WIN_CLOSED, "Cancel"]:
            event, self.values = window.read()
            print(self.getKey("toolbar_menu"), self.getElements(self.getKey("toolbar_menu")))
            menu_value = self.getValue(self.getKey("toolbar_menu"))

            parameter_names = self.getParameterNames()
            if menu_value is not None:
                if event == "Check All":
                    self.setChecks(names=parameter_names, overwrite=True)
                elif event == "Uncheck All":
                    self.setChecks(names=parameter_names, overwrite=False)
            elif event == "Submit":
                window.close()
                return [parameter_name for parameter_name in parameter_names if self.getValue(parameter_name)]

        window.close()
        return []
