"""
This file contains classes relating to the choose-parameters window.
"""
from __future__ import annotations

from os.path import basename
from typing import Iterable, List, Union

import PySimpleGUI as sg
from pint import Quantity

from Function import Parameter
from Layout.Layout import ChooseChecksWindow, Row, WindowRunner
from macros import formatQuantity, getTexImage, recursiveMethod


class ChooseParameterRow(Row):
    """
    This class contains the layout for a parameter row in the choose-parameters window.
        #. Label to indicate name of parameter.
        #. Label to indicate value and unit for parameter.
        #. Checkbox allow user to overwrite (or not) parameter into model.

    :ivar parameter: parameter object for parameter associated with row
    """

    def __init__(self, parameter: Parameter, window: ChooseParametersWindow):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParameterRow`.

        :param parameter: parameter object associated with row
        :param window: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` that row is stored in
        """
        super().__init__(parameter.getName(), window=window)
        self.parameter = parameter

        elements = [
            self.getParameterLabel(),
            self.getQuantityLabel(),
            self.getCheckbox()
        ]
        self.addElements(elements)

    def getParameter(self) -> Parameter:
        """
        Get parameter object associated with row.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve parameter from
        """
        return self.parameter

    def getQuantity(self) -> Quantity:
        """
        Get parameter quantity associated with row.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve quantity from
        """
        return self.getParameter().getQuantity()

    def getParameterLabel(self) -> Union[sg.Image, sg.Text]:
        """
        Get label for name of parameter.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve label from
        """
        return getTexImage(
            name=self.getName(),
            size=(110, None)  # dim
        )

    def getQuantityLabel(self) -> sg.Text:
        """
        Get label for parameter value and unit.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve label from
        """
        return sg.Text(
            text=formatQuantity(self.getQuantity()),
            size=(10, None)  # dim
        )

    def getCheckbox(self) -> sg.Checkbox:
        """
        Get checkbox, allowing user to overwrite (or not) parameter value in model.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParameterRow` to retrieve checkbox from
        """
        return sg.Checkbox(
            text="Overwrite?",
            default=True,
            key=self.getName()
        )


class ChooseParametersWindow(ChooseChecksWindow):
    """
    This class contains the layout for the choose-parameters window.
        #. Menu: This allows user to (un)check all parameters.
        #. Header to indicate purpose of each column.
        #. Footer with submit and cancel buttons.
        #. Row for each parameter, allowing user to choose whether (or not) to overwrite in model.

    :ivar parameters: dictionary of parameter objects.
        Key is name of parameter.
        Value is parameter object for parameter.
    :ivar filename: name of file that parameters were loaded from (optional)
    """

    def __init__(
        self,
        name: str,
        parameters: List[Parameter],
        runner: ChooseParametersWindowRunner,
        filename: str = None
    ):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`.

        :param name: name of window
        :param runner: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` that window is stored in
        :param parameters: dictionary of parameter objects.
            Key is name of parameter.
            Value is parameter object for parameter.
        :param filename: name of file that parameters were loaded from
        """
        header_text = "Choose which parameters to overwrite"
        if isinstance(filename, str):
            file_basename = basename(filename)
            header_text += f" ({file_basename:s})"

        ChooseChecksWindow.__init__(
            self,
            name=name,
            runner=runner,
            get_rows=self.getParameterRows,
            header_text=header_text
        )

        self.parameters = parameters
        self.filename = filename

    def getFilename(self) -> str:
        """
        Get name of file to load parameters from.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve name from
        """
        return self.filename

    def getParameters(self) -> List[Parameter]:
        """
        Get parameters stored in window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve parameters from
        :returns: Dictionary of parameter info.
            Key is name of parameter.
            Value is parameter object for parameter.
        """
        return self.parameters

    def getParameterRows(self) -> List[ChooseParameterRow]:
        """
        Get rows, each corresponding to a parameter stored in the window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow` to retrieve rows from
        """
        rows = [
            ChooseParameterRow(parameter, self)
            for parameter in self.getParameters()
        ]
        return rows


class ChooseParametersWindowRunner(WindowRunner, ChooseParametersWindow):
    """
    This class runs :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`.
    This window allows the user to...
        #. Choose which parameter(s) to overwrite into model.

    :ivar getParameters: pointer to :meth:`~Layout.ChooseParametersWindow.ChooseParametersWindow.getParameters`
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner`.

        :param args: required arguments for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`,
            excluding runner
        :param kwargs: additional arguments for :class:`~Layout.ChooseParametersWindow.ChooseParametersWindow`
        """
        WindowRunner.__init__(self)
        ChooseParametersWindow.__init__(
            self, 
            *args, 
            runner=self, 
            **kwargs
        )

    def getParameterNames(self) -> List[str]:
        """
        Get names of parameters stored in window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to retrieve names from
        """
        parameter_names = list(map(Parameter.getName, self.getParameters()))
        return parameter_names

    def setChecks(self, names: Union[str, Iterable[str]], checked: bool) -> None:
        """
        Set all checkboxes to chosen value.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to set checkboxes in
        :param names: name(s) of parameter(s) to set checkboxes for
        :param checked: set True to set all checkboxes to True.
            Set False to set all checkboxes to False.
        """

        def set(name: str) -> None:
            checkbox = self.getElements(name)
            checkbox.update(value=checked)

        return recursiveMethod(
            base_method=set,
            args=names,
            valid_input_types=str,
            output_type=list
        )

    def getCheckedParameterNames(self) -> List[str]:
        """
        Get currently checked parameters.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to retrieve checked boxes from
        """
        parameter_names = self.getParameterNames()
        checked_parameter_names = [parameter_name for parameter_name in parameter_names if self.getValue(parameter_name)]
        return checked_parameter_names

    def getChosenParameters(self) -> List[str]:
        """
        Get parameters chosen by user.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseParametersWindow.ChooseParametersWindowRunner` to retrieve checked boxes from
        :returns: Tuple of (event, parameter_names).
            event is name of final event, i.e. "Submit" or "Cancel",
            parameter_names is list of parameter names, where corresponding checkbox is checked,
        """
        window = self.getWindow()
        event = ''
        while event not in [sg.WIN_CLOSED, "Cancel"]:
            event, self.values = window.read()
            menu_value = self.getValue(self.getKey("toolbar_menu"))

            if menu_value is not None:
                parameter_names = self.getParameterNames()
                if event == "Check All":
                    self.setChecks(names=parameter_names, checked=True)
                elif event == "Uncheck All":
                    self.setChecks(names=parameter_names, checked=False)
            elif event == "Submit":
                checked_parameter_names = self.getCheckedParameterNames()
                window.close()
                return event, checked_parameter_names

        checked_parameter_names = self.getCheckedParameterNames()
        window.close()
        return event, checked_parameter_names
