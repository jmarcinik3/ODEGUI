"""
This file contains classes relating to the choose-variables window.
"""
from __future__ import annotations

from typing import Iterable, List, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg

from Function import Parameter
from Layout.Layout import ChooseChecksWindow, Row, WindowRunner
from macros import getTexImage, recursiveMethod


class ChooseVariableRow(Row):
    """
    This class contains the layout for a variable row in the choose-variables window.
        #. Label to indicate name of variable.
        #. Label to indicate value and unit for variable.
        #. Checkbox allow user to include variable in model.

    :ivar variable_name: name of variable associated with row
    """

    def __init__(self, variable_name: str, window: ChooseVariablesWindow):
        """
        Constructor for :class:`~Layout.ChooseVariablesWindow.ChooseParameterRow`.

        :param variable_name: name of variable associated with row
        :param window: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow` that row is stored in
        """
        super().__init__(variable_name, window=window)

        elements = [
            self.getVariableLabel(),
            self.getCheckbox()
        ]
        self.addElements(elements)

    def getVariableLabel(self) -> Union[sg.Image, sg.Text]:
        """
        Get label for name of variable.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariableRow` to retrieve label from
        """
        return getTexImage(
            name=self.getName(),
            size=(110, None)  # dim
        )

    def getCheckbox(self) -> sg.Checkbox:
        """
        Get checkbox, allowing user to include variable in model.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariableRow` to retrieve checkbox from
        """
        return sg.Checkbox(
            text="Include?",
            default=True,
            key=self.getName()
        )


class ChooseVariablesWindow(ChooseChecksWindow):
    """
    This class contains the layout for the choose-variables window.
        #. Menu: This allows user to (un)check all variables.
        #. Header to indicate purpose of each column.
        #. Footer with submit and cancel buttons.
        #. Row for each variable, allowing user to choose whether (or not) to include in model.

    :ivar variable_names: variable names included in window
    """

    def __init__(
            self,
            name: str,
            variable_names: str,
            runner: ChooseVariablesWindowRunner
    ):
        """
        Constructor for :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow`.

        :param name: name of window
        :param runner: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindowRunner` that window is stored in
        :param variable_names: variable names included in window
        """
        header_text = "Choose which variables to include in model"

        super().__init__(
            name,
            runner,
            get_rows=self.getVariableRows,
            header_text=header_text
        )

        self.variable_names = variable_names

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables stored in window.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow` to retrieve names from
        """
        return self.variable_names

    def getVariableRows(self) -> List[ChooseVariableRow]:
        """
        Get rows, each corresponding to a variable stored in the window.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow` to retrieve rows from
        """
        rows = [
            ChooseVariableRow(variable_name, self)
            for variable_name in self.getVariableNames()
        ]
        return rows


class ChooseVariablesWindowRunner(WindowRunner):
    """
    This class runs :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow`.
    This window allows the user to...
        #. Choose which variables to include in model.

    :ivar getVariableNames: pointer to :meth:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow.getVariableNames`
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindowRunner`.

        :param args: required arguments for :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow`,
            excluding runner
        :param kwargs: additional arguments for :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindow`
        """
        window_object = ChooseVariablesWindow(*args, runner=self, **kwargs)
        super().__init__(window_object)

        self.getVariableNames = window_object.getVariableNames

    def setChecks(self, names: Union[str, Iterable[str]], checked: bool) -> None:
        """
        Set all checkboxes to chosen value.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindowRunner` to set checkboxes in
        :param names: name(s) of variable(s) to set checkboxes for
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

    def getCheckedVariableNames(self) -> List[str]:
        """
        Get currently checked variables.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindowRunner` to retrieve checked boxes from
        """
        variable_names = self.getVariableNames()
        checked_variable_names = [variable_name for variable_name in variable_names if self.getValue(variable_name)]
        return checked_variable_names

    def getChosenVariables(self) -> List[str]:
        """
        Get checked variables in window.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseVariablesWindow.ChooseVariablesWindowRunner` to retrieve checked boxes from
        :returns: Tuple of (event, variable_names).
            event is name of final event, i.e. "Submit" or "Cancel",
            variable_names is list of variable names, where corresponding checkbox is checked,
        """
        window = self.getWindow()
        event = ''
        while event not in [sg.WIN_CLOSED, "Cancel"]:
            event, self.values = window.read()
            menu_value = self.getValue(self.getKey("toolbar_menu"))

            variable_names = self.getVariableNames()
            if menu_value is not None:
                if event == "Check All":
                    self.setChecks(names=variable_names, checked=True)
                elif event == "Uncheck All":
                    self.setChecks(names=variable_names, checked=False)
            elif event == "Submit":
                checked_variable_names = self.getCheckedVariableNames()
                window.close()
                return event, checked_variable_names

        checked_variable_names = self.getCheckedVariableNames()
        window.close()
        return event, checked_variable_names
