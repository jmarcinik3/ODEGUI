from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
import yaml
from numpy import array
from numpy import ndarray
from pint import Quantity

import YML
from CustomErrors import RecursiveTypeError
from Function import Independent, Model
from Layout.ChooseParametersWindow import ChooseParametersWindowRunner
from Layout.Layout import Element, Layout, Row, Tab, TabGroup, TabRow, TabbedWindow, WindowRunner, generateCollapsableSection
from Layout.SetFreeParametersWindow import SetFreeParametersWindowRunner
from Layout.SimulationWindow import SimulationWindowRunner
from macros import formatQuantity, getTexImage


class TimeEvolutionVariableRow(TabRow):
    """
    Row to set time-evolution properties for variable.
    This contains
        #. Label of variable name
        #. Combobox to choose time-evolution type
        #. Input field to set initial value
        #. Checkbox to set initial condition as equilibrium
    """

    def __init__(self, name: str, tab: TimeEvolutionTab) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.TimeEvolutionVariableRow`.

        :param name: name of variable
        :param tab: tab that row is stored in
        """
        super().__init__(name, tab)

        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        window_object.addVariableNames(name)

        elements = [
            self.getRowLabel(),
            self.getTimeEvolutionTypeElement(),
            self.getInitialConditionElement(),
            self.getInitialEquilibriumElement()
        ]
        # if "Equilibrium" in self.getTimeEvolutionTypes(): elements.append(self.getCheckbox())
        self.addElements(elements)

    def getInitialCondition(self) -> float:
        """
        Get default initial condition for variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve initial condition from
        """
        return self.getTab().getInitialConditions(self.getName())

    def getTimeEvolutionTypes(self, **kwargs) -> Union[str, List[str]]:
        """
        Get possible time-evolution types for variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve time-evolution types from
        :param kwargs: additional arguments to pass into :meth:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes`
        """
        tab: TimeEvolutionTab = self.getTab()
        return tab.getTimeEvolutionTypes(self.getName(), **kwargs)

    def getTimeEvolutionTypeElement(self) -> sg.InputCombo:
        """
        Get element allowing user to set time-evolution type.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve element from
        """
        kwargs = {
            "values": self.getTimeEvolutionTypes(),
            "default_value": self.getTimeEvolutionTypes(index=0),
            "enable_events": True,
            "size": self.getDimensions(name="evolution_type_combobox"),
            "key": self.getKey("time_evolution_type", self.getName())
        }
        return sg.InputCombo(**kwargs)

    def getRowLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get element to label time-evolution row by variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve label from
        """
        kwargs = {
            "name": self.getName(),
            "size": self.getDimensions(name="variable_label")
        }
        return getTexImage(**kwargs)

    def getInitialConditionElement(self) -> sg.InputText:
        """
        Get element allowing user to input initial value.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve element from
        """
        kwargs = {
            "default_text": self.getInitialCondition(),
            "size": self.getDimensions(name="initial_condition_input_field"),
            "key": self.getKey("initial_condition_value", self.getName())
        }
        return sg.InputText(**kwargs)

    def getInitialEquilibriumElement(self) -> sg.Checkbox:
        """
        Get element allowing user to choose whether or not variable begins simulation in equilibrium.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionVariableRow` to retrieve element from
        """
        kwargs = {
            "text": "Equilibrium",
            "default": False,
            "enable_events": True,
            "disabled": True,
            "size": self.getDimensions(name="initial_equilibrium_checkbox"),
            "key": self.getKey("initial_condition_equilibrium", self.getName())
        }
        return sg.Checkbox(**kwargs)


class TimeEvolutionTab(Tab):
    """
    Tab to organize time-evolution rows for variables.
    This contains
        #. Header of labels to indicate purpose of each element in variable row
        #. :class:`~Layout.MainWindow.TimeEvolutionVariableRow` for each variable in tab
    
    :ivar time_evolution_types: dictionary of time-evolution types for variables.
        Key is name of variable.
        Value is list of time-evolution types for variable.
    :ivar initial_conditions: dictionary of initial values for variables.
        Key is name of variable.
        Value is initial value for variable.
    :ivar variable_rows: list of :class:`~Layout.MainWindow.TimeEvolutionRow`, one for each variable in tab
    """

    def __init__(self, name: str, window: MainWindow, variable_names: List[str]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.TimeEvolutionTab`.
        
        :param name: name of tab
        :param window: window that tab is stored in
        :param variable_names: names of variables included in tab
        """
        super().__init__(name, window)

        self.time_evolution_types = YML.readStates("time_evolution_types")
        self.initial_conditions = YML.readStates("initial_condition")

        self.variable_rows = []
        self.addVariableRows(variable_names)

    def addVariableRows(
            self, names: Union[str, List[str]]
    ) -> Union[TimeEvolutionVariableRow, List[TimeEvolutionVariableRow]]:
        """
        Add rows corresponding to variable names.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to add rows to
        :param names: name(s) of variable(s) to add rows for
        :returns: New row added if names is str.
            List of new rows if names is list.
        """
        if isinstance(names, str):
            new_row = TimeEvolutionVariableRow(names, self)
            self.variable_rows.append(new_row)
            return new_row
        elif isinstance(names, list):
            return [self.addVariableRows(name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getVariableRows(self) -> List[TimeEvolutionVariableRow]:
        """
        Get variable rows added to tab.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve rows from
        """
        return self.variable_rows

    def getInitialConditions(self, names: str = None) -> Union[float, Dict[str, float]]:
        """
        Get default initial condition for variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve initial condition from
        :param names: name(s) of variable to retrieve initial conditions for
        """
        if isinstance(names, str):
            initial_conditions = self.initial_conditions
            return initial_conditions[names]
        elif isinstance(names, list):
            return {name: self.getInitialConditions(names=name) for name in names}
        elif names is None:
            return self.initial_conditions

    def getTimeEvolutionTypes(self, name: str, index: int = None) -> Union[str, List[str]]:
        """
        Get default initial condition for variable.

        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve initial condition from
        :param name: name of variable to retrieve time-evolution type for
        :param index: index of time-evolution type to retrieve within collection of time-evolution types for variable
        :returns: Time-evolution type at index if :paramref:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes.index` is int.
            All time-evolution types for variable if :paramref:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes.index` is None.
        """
        if isinstance(index, int):
            time_evolution_types = self.getTimeEvolutionTypes(name=name)
            return time_evolution_types[index]
        elif index is None:
            return self.time_evolution_types[name]
        else:
            raise TypeError("index must be int")

    def getHeaderRow(self) -> Row:
        """
        Get header row for single time-evolution tab.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve row from
        """
        row = Row(window=self.getWindowObject())
        kwargs = {
            "text": "Variable",
            "size": self.getDimensions(name="variable_text"),
            "justification": "center"
        }
        row.addElements(sg.Text(**kwargs))
        kwargs = {
            "text": "Evolution Type",
            "size": self.getDimensions(name="evolution_type_text"),
            "justification": "center"
        }
        row.addElements(sg.Text(**kwargs))
        kwargs = {
            "text": "Initial Condition",
            "size": self.getDimensions(name="initial_condition_text"),
            "justification": "center"
        }
        row.addElements(sg.Text(**kwargs))
        return row

    def getAsColumn(self) -> sg.Column:
        """
        Get time-evolution tab as an column object.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve as column
        """
        header_row = self.getHeaderRow()
        rows = [row.getRow() for row in self.getVariableRows()]
        layout = header_row.getLayout() + rows
        kwargs = {
            "layout": layout,
            "size": self.getDimensions(name="time_evolution_tab"),
            "scrollable": False,
            "vertical_scroll_only": True
        }
        return sg.Column(**kwargs)

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for time-evolution tab.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to retrieve layout from
        """
        return [[self.getAsColumn()]]


class TimeEvolutionTabGroup(TabGroup):
    """
    Tabgroup for time-evolution tabs.
    This contains
        #. :class:`~Layout.MainWindow.TimeEvolutionTab` for each group of variables
    """

    def __init__(self, name: str, window: MainWindow, blueprint: dict) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.TimeEvolutionTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprint: dictionary dictating how to set up each tab within tab group.
            Key is name of tab in tab group.
            Value is names of variables within this tab.
        """
        tabs = []
        append_tab = tabs.append
        tab_names = blueprint.keys()
        for tab_name in tab_names:
            variable_names = blueprint[tab_name]
            append_tab(TimeEvolutionTab(tab_name, window, variable_names))
        super().__init__(tabs, name=name)


class ParameterRow(TabRow):
    """
    Row to set properties for parameter.
    This contains
        #. Label of parameter name
        #. Label for present parameter value and unit
        #. Input field to set parameter value
        #. Combobox to choose parameter type
    """

    def __init__(
            self, name: str, section: ParameterSection, parameter_types: Tuple[str] = ("Constant", "Free")
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterRow`.

        :param name: name of tab group
        :param section: :class:`~Layout.MainWindow.ParameterSection` that row is stored in
        :param parameter_types: collection of types that each parameter can take on
        """
        super().__init__(name, section.getTab())
        self.section = section
        self.parameter_types = parameter_types

        elements = [
            self.getParameterLabel(),
            self.getQuantityLabel(),
            self.getValueInputElement(),
            self.getParameterTypeElement()
        ]
        self.addElements(elements)

    def getSection(self) -> ParameterSection:
        """
        Get parameter section containing row.
        
        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve section for
        """
        return self.section

    def getParameterTypes(self, index: int = None) -> Tuple[str]:
        """
        Get possible types for parameter associated with row.
        
        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve types from
        :param index: index of parameter type to retrieve within collection of parameter types
        :returns: Parameter type at index if :paramref:`~Layout.MainWindow.ParameterRow.getParameterTypes.index` is int.
            All parameter types for parameter if :paramref:`~Layout.MainWindow.ParameterRow.getParameterTypes.index` is None.
        """
        if isinstance(index, int):
            parameter_types = self.getParameterTypes()
            return parameter_types[index]
        elif index is None:
            return self.parameter_types
        else:
            raise TypeError("index must be int")

    def getQuantity(self) -> Quantity:
        """
        Get quantity for parameter.
        This contains value and unit.
        
        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve quantity from
        """
        # noinspection PyTypeChecker
        runner: MainWindowRunner = self.getWindowRunner()
        name = self.getName()
        return runner.getParameters(name)

    def getParameterLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get label to indicate parameter name.
        
        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve label from
        """
        kwargs = {
            "name": self.getName(),
            "size": self.getDimensions(name="parameter_label")
        }
        return getTexImage(**kwargs)

    def getQuantityLabel(self) -> sg.Text:
        """
        Get label to indicate quantity value and unit.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve label from
        """
        kwargs = {
            "text": formatQuantity(self.getQuantity()),
            "tooltip": "Present value of parameter",
            "size": self.getDimensions(name="parameter_value_label"),
            "key": self.getKey("quantity_label", self.getName())
        }
        return sg.Text(**kwargs)

    def getValueInputElement(self) -> sg.InputText:
        """
        Get field to allow user to set parameter value.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve element from
        """
        kwargs = {
            "default_text": '',
            "tooltip": "If 'Constant': enter new parameter value, then click button to update. " + "If 'Sweep': this input field will be ignored." + "Displayed units are preserved",
            "size": self.getDimensions(name="parameter_value_input_field"),
            "key": self.getKey("parameter_field", self.getName())
        }
        return sg.InputText(**kwargs)

    def getParameterTypeElement(self) -> sg.InputCombo:
        """
        Get field to allow user input to set parameter type.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve element from
        """
        kwargs = {
            "values": self.getParameterTypes(),
            "default_value": self.getParameterTypes(0),
            "enable_events": True,
            "size": self.getDimensions(name="parameter_type_combobox"),
            "key": self.getKey("parameter_type", self.getName())
        }
        return sg.InputCombo(**kwargs)


class ParameterSection(Element):
    """
    Section to organize parameter rows for parameters.
    This contains
        #. Header to indicate name of parameter section
        #. collapsable section of :class:`~Layout.MainWindow.ParameterRow` for each parameter in section

    :ivar name: name of section
    :ivar tab: tab that section is stored in
    :ivar parameters_rows: list of :class:`~Layout.MainWindow.ParameterRow`, one for each parameter in section
    """

    def __init__(self, name: str, tab: ParameterTab, parameter_names: List[str]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterSection`.

        :param name: name of tab
        :param tab: tab that section in stored in
        :param parameter_names: names of parameters included in section
        """
        super().__init__(tab.getWindowObject())
        self.name = name
        self.tab = tab

        self.parameter_rows = []
        self.addParameterRows(parameter_names)

    def addParameterRows(self, names: Union[str, List[str]]) -> Union[ParameterRow, List[ParameterRow]]:
        """
        Add rows corresponding to parameter names.

        :param self: :class:`~Layout.MainWindow.ParameterSection` to add rows to
        :param names: name(s) of parameter(s) to add rows for
        :returns: New row added if names is str.
            List of new rows if names is list.
        """
        if isinstance(names, str):
            new_row = ParameterRow(names, self)
            self.parameter_rows.append(new_row)
            return new_row
        elif isinstance(names, list):
            return [self.addParameterRows(name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getParameterRows(self) -> List[ParameterRow]:
        """
        Get parameter rows added to section.

        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve rows from
        """
        return self.parameter_rows

    def getName(self) -> str:
        """
        Get name of section.
        
        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve name from
        """
        return self.name

    def getTab(self) -> ParameterTab:
        """
        Get parameter tab that section is contained within.
        
        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve tab from
        """
        return self.tab

    def getHeaderLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get label in header to indicate section of parameters.
        
        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve label from
        """
        name = self.getName()
        kwargs = {
            "enable_events": True,
            "size": self.getDimensions(name="parameter_header_label"),
            "key": self.getWindowObject().getKey("parameter_subgroup_label", name)
        }
        return getTexImage(name, **kwargs)

    def getHeaderRow(self) -> Row:
        """
        Get header row to indicate section of parameters.
        
        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve row from
        """
        row = Row(window=self.getWindowObject())
        row.addElements(self.getHeaderLabel())
        row.addElements(sg.HorizontalSeparator())
        return row

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for parameter section.

        :param self: :class:`~Layout.MainWindow.ParameterSection` to retrieve layout from
        """
        header_row = self.getHeaderRow()
        kwargs = {
            "layout": [row.getRow() for row in self.getParameterRows()],
            "size": self.getDimensions(name="parameter_section"),
            "key": self.getWindowObject().getKey("parameter_subgroup_section", self.getName())
        }
        collapsable_section = generateCollapsableSection(**kwargs)

        layout = Layout()
        layout.addRows(rows=header_row)
        layout.addRows(rows=Row(window=self.getWindowObject(), elements=collapsable_section))
        return layout.getLayout()


class ParameterTab(Tab):
    """
    Tab to organize parameter sections.
    This contains
        #. :class:`~Layout.MainWindow.ParameterSection` for each section of parameters
        
    :ivar sections: list of :class:`~Layout.MainWindow.ParameterSection`, one for each section in tab
    """

    def __init__(self, name: str, window: MainWindow, blueprint: dict) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterTab`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprint: dictionary dictating how to set up each section within tab.
            Key is name of section in tab.
            Value is names of parameters within this section.
        """
        super().__init__(name, window)

        self.sections = []
        if isinstance(blueprint, list):
            blueprint = {
                name: blueprint
            }
        for section_name, section_parameters in blueprint.items():
            self.addSection(section_name, section_parameters)

    def addSection(self, name: str, parameter_names: List[str]) -> ParameterSection:
        """
        Add section of parameters into tab.

        :param self: :class:`~Layout.MainWindow.ParameterTab` to add rows to
        :param name: name of section containing parameters
        :param parameter_names: name(s) of parameter(s) within section
        :returns: New section added.
        """
        new_section = ParameterSection(name, self, parameter_names)
        self.sections.append(new_section)
        return new_section

    def getSections(self) -> List[ParameterSection]:
        """
        Get sections added to tab.

        :param self: :class:`~Layout.MainWindow.ParameterTab` to retrieve section from
        """
        return self.sections

    def getAsColumn(self) -> sg.Column:
        """
        Get scrollable element with tab layout.
        
        :param self: :class:`~Layout.MainWindow.ParameterTab` to retrieve element from
        """
        layout = [[]]
        section_layouts = [section.getLayout() for section in self.getSections()]
        for section_layout in section_layouts:
            layout += section_layout
        kwargs = {
            "layout": layout,
            "size": self.getDimensions(name="parameter_tab"),
            "scrollable": True,
            "vertical_scroll_only": True
        }
        return sg.Column(**kwargs)

    def getLayout(self) -> List[List[sg.Column]]:
        """
        Get layout for tab.
        
        :param self: :class:`~Layout.MainWindow.ParameterTab` to retrieve layout from
        """
        return [[self.getAsColumn()]]


class ParameterTabGroup(TabGroup):
    """
    Tabgroup for parameter tabs.
    This contains
        #. :class:`~Layout.MainWindow.ParameterTab` for each group of parameters
    """

    def __init__(self, name: str, window: MainWindow, blueprint: dict, suffix_layout: Layout) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprint: dictionary dictating how to set up each tab within tab group.
            Key is name of tab.
            Value is in dictionary dictating how to set up sections within this tab.
        """
        tabs = []
        tab_names = blueprint.keys()
        for tab_name in tab_names:
            tab_blueprint = blueprint[tab_name]
            new_tab = ParameterTab(tab_name, window, tab_blueprint)
            tabs.append(new_tab)
        super().__init__(tabs, name=name, suffix_layout=suffix_layout)


class FunctionRow(TabRow):
    def __init__(self, name: str, tab: FunctionTab) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionRow`.

        :param name: name of function
        :param tab: tab that row is stored in
        """
        super().__init__(name, tab)

        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        window_object.addFunctionNames(name)

        elements = [self.getRowLabel(), self.getFormLabel()]
        # if "Equilibrium" in self.getTimeEvolutionTypes(): elements.append(self.getCheckbox())
        self.addElements(elements)

    def getRowLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get element to label function row by function name.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve label from
        """
        kwargs = {
            "name": self.getName(),
            "size": self.getDimensions(name="function_label")
        }
        return getTexImage(**kwargs)

    def getFormLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get element to display form of function.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve label from
        """
        for filename_temp in ["equations/Bormuth2014.yml", "equations/soma_eqs.yml", "equations/hb-soma_eqs.yml"]:
            function_info = yaml.load(open(filename_temp, 'r'), Loader=yaml.Loader)
            try:
                form = function_info[self.getName()]["form"]
            except KeyError:
                try:
                    form = function_info[self.getName()]["pieces"]
                except KeyError:
                    pass
        kwargs = {
            "name": form,
            "size": self.getDimensions(name="form_label")
        }
        return getTexImage(**kwargs)


class FunctionTab(Tab):
    def __init__(self, name: str, window: MainWindow, function_names: List[str]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionTab`.
    
        :param name: name of tab
        :param window: window that tab is stored in
        :param function_names: names of functions included in tab
        """
        super().__init__(name, window)

        self.function_rows = []
        self.addFunctionRows(function_names)

    def addFunctionRows(self, names: Union[str, List[str]]) -> Union[FunctionRow, List[FunctionRow]]:
        """
        Add rows corresponding to function names.
    
        :param self: :class:`~Layout.MainWindow.FunctionTab` to add rows to
        :param names: name(s) of function(s) to add rows for
        :returns: New row added if names is str.
            List of new rows if names is list.
        """
        if isinstance(names, str):
            new_row = FunctionRow(names, self)
            self.function_rows.append(new_row)
            return new_row
        elif isinstance(names, list):
            return [self.addFunctionRows(name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getFunctionRows(self) -> List[FunctionRow]:
        """
        Get variable rows added to tab.
    
        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve rows from
        """
        return self.function_rows

    def getHeaderRow(self) -> Row:
        """
        Get header row for single function tab.
    
        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve row from
        """
        row = Row(window=self.getWindowObject())
        texts = ["Function", "Form"]
        dimension_keys = [f"function_header_row_{string:s}" for string in ["function_label", "form_label"]]
        add_element = row.addElements
        for index in range(len(texts)):
            kwargs = {
                "text": texts[index],
                "size": self.getDimensions(name=dimension_keys[index]),
                "justification": "left"
            }
            add_element(sg.Text(**kwargs))
        return row

    def getAsColumn(self) -> sg.Column:
        """
        Get function tab as an column object.
    
        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve as column
        """
        header_row = self.getHeaderRow()
        rows = [row.getRow() for row in self.getFunctionRows()]
        layout = header_row.getLayout() + rows
        kwargs = {
            "layout": layout,
            "size": self.getDimensions(name="function_tab"),
            "scrollable": True,
            "vertical_scroll_only": True
        }
        return sg.Column(**kwargs)

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for function tab.
    
        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve layout from
        """
        return [[self.getAsColumn()]]


class FunctionTabGroup(TabGroup):
    def __init__(self, name: str, window: MainWindow, blueprint: dict) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprint: dictionary dictating how to set up each tab within tab group.
            Key is name of tab in tab group.
            Value is names of functions within this tab.
        """
        tabs = []
        append_tab = tabs.append
        tab_names = blueprint.keys()
        for tab_name in tab_names:
            function_names = blueprint[tab_name]
            append_tab(FunctionTab(tab_name, window, function_names))
        super().__init__(tabs, name=name)


class MainWindow(TabbedWindow):
    """
    Primary window to run program.
    This contains
        #. :class:`~Layout.MainWindow.TimeEvolutionTabGroup` to allow user to set time-evolution properties for each variable
        #. :class:`~Layout.MainWindow.ParameterTabGroup` to allow user to set properties for each parameter
    """

    def __init__(self, name: str, runner: MainWindowRunner, blueprint: dict) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.MainWindow`.
        
        :param name: title of window
        :param runner: :class:`~Layout.MainWindow.MainWindowRunner` that window is stored in
        :param blueprint: dictionary indicating how to set up elements in window
        """
        dimensions = {
            "window": YML.getDimensions(["main_window", "window"]),
            "time_evolution_tab": YML.getDimensions(["main_window", "time_evolution_tab", "tab"]),
            "evolution_type_text": YML.getDimensions(
                ["main_window", "time_evolution_tab", "header_row", "evolution_type_text"]
            ),
            "variable_text": YML.getDimensions(["main_window", "time_evolution_tab", "header_row", "variable_text"]),
            "initial_condition_text": YML.getDimensions(
                ["main_window", "time_evolution_tab", "header_row", "initial_condition_text"]
            ),
            "evolution_type_combobox": YML.getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "evolution_type_combobox"]
            ),
            "variable_label": YML.getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "variable_label"]
            ),
            "initial_condition_input_field": YML.getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "initial_condition_input_field"]
            ),
            "initial_equilibrium_checkbox": YML.getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "initial_equilibrium_checkbox"]
            ),
            "parameter_tab": YML.getDimensions(["main_window", "parameter_tab", "tab"]),
            "parameter_section": YML.getDimensions(["main_window", "parameter_tab", "parameter_section"]),
            "parameter_header_label": YML.getDimensions(
                ["main_window", "parameter_tab", "header_row", "section_label"]
            ),
            "parameter_label": YML.getDimensions(["main_window", "parameter_tab", "parameter_row", "parameter_label"]),
            "parameter_value_label": YML.getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_value_label"]
            ),
            "parameter_value_input_field": YML.getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_value_input_field"]
            ),
            "parameter_type_combobox": YML.getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_type_combobox"]
            ),
            "function_tab": YML.getDimensions(["main_window", "function_tab", "tab"]),
            "function_label": YML.getDimensions(["main_window", "function_tab", "function_row", "function_label"]),
            "form_label": YML.getDimensions(["main_window", "function_tab", "function_row", "form_label"]),
            "function_header_row_function_label": YML.getDimensions(
                ["main_window", "function_tab", "header_row", "function_label"]
            ),
            "function_header_row_form_label": YML.getDimensions(
                ["main_window", "function_tab", "header_row", "form_label"]
            )
        }
        super().__init__(name, runner, dimensions=dimensions)

        self.variable_names = []
        self.function_names = []
        self.blueprint = blueprint

        tabs = [
            self.getTimeEvolutionTypeTab(),
            self.getParameterInputTab(),
            # self.getFunctionTab()
        ]
        self.addTabs(tabs)

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables included in window.
        
        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve names from
        """
        return self.variable_names

    def addVariableNames(self, names: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Add names of variables included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to add names into
        :param names: name(s) of variable(s) to add
        :returns: Name of new variable added if :paramref:`~Layout.MainWindow.MainWindow.addVariableNames.names` is str.
            List of new variables added if :paramref:`~Layout.MainWindow.MainWindow.addVariableNames.names` is list.
        """
        if isinstance(names, str):
            new_variable_name = names
            self.variable_names.append(names)
            return new_variable_name
        elif isinstance(names, list):
            return [self.addVariableNames(names=name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getFunctionNames(self) -> List[str]:
        """
        Get names of functions included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve names from
        """
        return self.function_names

    def addFunctionNames(self, names: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Add names of functions included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to add names into
        :param names: name(s) of function(s) to add
        :returns: Name of new function added if :paramref:`~Layout.MainWindow.MainWindow.addVFunctionNames.names` is str.
            List of new functions added if :paramref:`~Layout.MainWindow.MainWindow.addFunctionNames.names` is list.
        """
        if isinstance(names, str):
            new_function_name = names
            self.function_names.append(names)
            return new_function_name
        elif isinstance(names, list):
            return [self.addFunctionNames(names=name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getBlueprints(self, tab_name: str = None) -> dict:
        """
        Get blueprint for specified tab.
        Returns all b dictionary of all blueprints by default.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve blueprints from
        :param tab_name: name of tab to get blueprint for
        """
        if tab_name is None:
            return self.blueprint
        else:
            return self.blueprint[tab_name]

    def getMenu(self) -> sg.Menu:
        """
        Get toolbar menu for window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve menu from
        """
        menu_definition = [
            [
                "Import",
                [
                    "Parameters::import",
                    "Time-Evolution Types::import"
                ]
            ], [
                "Set",
                [
                    "Parameter Types to...",
                    [
                        "Constant::set_parameter_types_to",
                        "Free::set_parameter_types_to"
                    ],
                    "Time-Evolution Types to...",
                    [
                        "Temporal::set_time_evolution_types_to",
                        "Equilibrium::set_time_evolution_types_to",
                        "Constant::set_time_evolution_types_to",
                        "Function::set_time_evolution_types_to"
                    ]
                ]
            ]
        ]
        kwargs = {
            "menu_definition": menu_definition,
            "key": self.getKey("toolbar_menu")
        }
        return sg.Menu(**kwargs)

    def getOpenSimulationButton(self) -> sg.Button:
        """
        Get button to open :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve button from
        """
        kwargs = {
            "button_text": "Open Simulation",
            "key": self.getKey("open_simulation")
        }
        return sg.Button(**kwargs)

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve layout form
        """
        menu = self.getMenu()
        open_simulation_button = self.getOpenSimulationButton()
        prefix_layout = Layout(rows=Row(window=self, elements=menu))
        suffix_layout = Layout(rows=Row(window=self, elements=open_simulation_button))
        tabgroup = TabGroup(self.getTabs())
        return prefix_layout.getLayout() + tabgroup.getLayout() + suffix_layout.getLayout()

    def getTimeEvolutionTypeTab(self) -> sg.Tab:
        """
        Get time-evolution tabgroup as tab, containing tab for each variable group
        
        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve tabgroup from
        """
        tabgroup = TimeEvolutionTabGroup("Time Evolution", self, self.getBlueprints("time_evolution"))
        tab = tabgroup.getAsTab()
        return tab

    def getParameterInputTab(self) -> sg.Tab:
        """
        Get time-evolution tabgroup as tab, containing tab for each parameter group
        
        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve tabgroup from
        """
        common = Layout(rows=Row(elements=sg.Button("Update Parameters")))
        tabgroup = ParameterTabGroup("Parameters", self, self.getBlueprints("parameter_input"), common)
        tab = tabgroup.getAsTab()
        return tab

    def getFunctionTab(self) -> sg.Tab:
        """
        Get function tabgroup as tab, containing tab for each function group

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve tabgroup from
        """
        tabgroup = FunctionTabGroup("Function", self, self.getBlueprints("functions"))
        tab = tabgroup.getAsTab()
        return tab


class MainWindowRunner(WindowRunner):
    """
    Runner for :class:`~Layout.MainWindow.MainWindow`.
    
    :ivar function_ymls: filename(s) of YML files, containing information about functions for model
    :ivar parameters: dictionary of parameter quantities.
        Key is name of parameter.
        Value is quantity containing value and unit for parameter.
    """

    def __init__(
            self,
            name: str,
            parameter_filenames: Union[str, List[str]],
            function_filenames: Union[str, List[str]],
            time_evolution_filename: str,
            parameter_input_filename: str,
            function_filename: str
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.MainWindowRunner`.
        
        :param name: name of window
        :param parameter_filenames: name(s) for file(s), containing information about parameters in model
        :param function_filenames: name(s) for file(s), containing information about functions in model
        :param time_evolution_filename: name of file containing layout for time-evolution tab
        :param parameter_input_filename: name of file containing layout for parameter-input tab
        """
        self.function_ymls = function_filenames
        self.parameters = YML.readParameters(parameter_filenames)

        blueprints = {
            "time_evolution": YML.readLayout(time_evolution_filename),
            "parameter_input": YML.readLayout(parameter_input_filename),
            "functions": YML.readLayout(function_filename)
        }
        window = MainWindow("Hair Bundle/Soma Model", self, blueprints)
        super().__init__(name, window)

    def runWindow(self) -> None:
        window = self.getWindow()
        while True:
            event, self.values = window.read()

            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            psl_pre = self.getPrefix("parameter_subgroup_label")
            tet_pre = self.getPrefix("time_evolution_type")
            ice_pre = self.getPrefix("initial_condition_equilibrium")
            menu_value = self.getValue(self.getKey("toolbar_menu"))

            if menu_value is not None:
                if event == "Parameters::import":
                    self.loadParametersFromFile()
                elif "::set_time_evolution_types_to" in event:
                    time_evolution_type = event.replace("::set_time_evolution_types_to", '')
                    self.setElementsWithPrefix("time_evolution_type", time_evolution_type)
                elif "::set_parameter_types_to" in event:
                    parameter_type = event.replace("::set_parameter_types_to", '')
                    self.setElementsWithPrefix("parameter_type", parameter_type)
            elif psl_pre in event:
                key = event.replace(psl_pre, self.getPrefix("parameter_subgroup_section"))
                self.toggleVisibility(key)
            elif tet_pre in event or ice_pre in event:
                prefix = "time_evolution_type" if tet_pre in event else "initial_condition_equilibrium"
                variable_name = self.getVariableNameFromElementKey(event, prefix)
                self.changeTimeEvolution(variable_name)
            elif event == "Update Parameters":
                self.updateParameters()
            elif event == self.getKey("open_simulation"):
                self.openSimulationWindow()
        window.close()

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables in window.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve names from
        """
        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        return window_object.getVariableNames()

    def getParameterNames(self, species: Union[str, List[str]] = None) -> List[str]:
        """
        Get name(s) of parameter(s) in model.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve names from
        :param species: species of parameters to retrieve, acts as optional filter.
        """
        if isinstance(species, str):
            filtered_parameter_names = []
            for parameter_name in self.getParameterNames():
                combobox_key = self.getKey("parameter_type", parameter_name)
                parameter_type = self.getValue(combobox_key)
                if parameter_type == species:
                    filtered_parameter_names.append(parameter_name)
            return filtered_parameter_names
        elif isinstance(species, list):
            filtered_parameter_names = []
            for specie in species:
                filtered_parameter_names.extend(self.getParameterNames(species=specie))
            return filtered_parameter_names
        elif species is None:
            key_list = self.getKeyList(prefixes="parameter_type")
            key_prefix = self.getPrefix(prefix="parameter_type", with_separator=True)
            parameter_names = [key.replace(key_prefix, '') for key in key_list]
            return parameter_names
        else:
            raise RecursiveTypeError(species)

    def getPlotChoices(self, model: Model = None) -> Dict[str, List[str]]:
        """
        Get names of variable/functions to analyze for plot.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve choices from
        :param model: model to retrieve choices from.
            Default to model associated with window.
        :returns: Dictionary of plot choices.
            Key is species of plot choice (e.g. "Variable", "Function", "Parameter").
            Value is list of names for plot choices.
        """
        if model is None:
            model = self.getModel()
        plot_choices = {
            "Variable": ['t'] + model.getDerivativeVariables(return_type=str),
            "Function": [function.getName() for function in model.getFunctions(filter_type=Independent)],
            "Parameter": self.getParameterNames(species="Free")
        }
        return plot_choices

    def getModel(self) -> Model:
        """
        Get model for window.
        Uses present state of window.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve model from
        """
        parameters = {name: self.getParameters(names=name) for name in self.getParameterNames()}
        model = Model(parameters=parameters)
        model.loadFunctionsFromFiles(self.getFunctionYMLs())
        for derivative in model.getDerivatives():
            variable_name = derivative.getVariable(return_type=str)
            derivative.setTimeEvolutionType(self.getTimeEvolutionTypes(names=variable_name))
            derivative.setInitialCondition(self.getInitialConditions(names=variable_name))
        return model

    def setElementsWithPrefix(self, prefix: str, value: str) -> None:
        """
        Set elements with same function to certain value.
        For example, set all elements that allow user input for time-evolution type to 'Temporal'.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to set elements in
        :param prefix: prefix that elements to change value have in common
        :param value: value to set elements to
        """
        elements = self.getElements(self.getKeyList(prefix))
        for element in elements:
            if value in vars(element)["Values"]:
                element.update(value)

    def getInitialConditions(self, names: Union[str, List[str]] = None) -> Union[float, ndarray, str]:
        """
        Get initial conditions for variables.
        Uses present state of window.
        
        __Recursion Base__
            get initial condition value for single variable: names [str]
        
        :param self: `~Layout.MainWindow.MainWindowRunner` to retrieve conditions from
        :param names: name(s) of variables to retrieve conditions for
        :returns: Initial value if variable does not begin in equilibrium.
            "Equilibrium" if variable begins in equilibrium.
        """
        if isinstance(names, str):
            time_evolution_type = self.getValue(self.getKey("time_evolution_type", names))
            is_equilibrium = time_evolution_type == "Equilibrium"
            is_initial_equilibrium = self.getValue(
                self.getKey("initial_condition_equilibrium", names)
            ) or is_equilibrium

            if is_initial_equilibrium:
                return "Equilibrium"
            else:
                input_field_key = self.getKey("initial_condition_value", names)
                try:
                    value = float(self.getValue(input_field_key))
                except AttributeError:
                    value = float(YML.getStates("initial_condition", names))
                return value
        elif isinstance(names, list):
            return array(self.getInitialConditions(names=name) for name in names)
        else:
            raise RecursiveTypeError(names)

    def getTimeEvolutionTypes(self, names: Union[str, List[str]] = None) -> Union[str, List[str]]:
        """
        Get selected time-evolution type for variable.
        Get default time-evolution type if none selected.
        Uses present state of window.
        
        __Recursion Base__
            get time-evolution selection for single variable: names [str]
        
        :param self: `~Layout.MainWindow.MainWindowRunner` to retrieve time-evolution types from
        :param names: name(s) of variables to retrieve time-evolution types for
        """
        if isinstance(names, str):
            key = self.getKey("time_evolution_type", names)
            try:
                return self.getValue(key)
            except AttributeError:
                return YML.getStates("time_evolution_types", names)
        elif isinstance(names, list):
            return [self.getTimeEvolutionTypes(names=name) for name in names]
        else:
            raise RecursiveTypeError(names)

    def getVariableNameFromElementKey(self, key: str, prefix: str) -> Optional[str]:
        """
        Get name of variable associated with element.

        :param self: :class:`~Layout.SimulationWindow.SimulationWindowRunner` to search for key in
        :param key: key for element
        :param prefix: prefix codename of key
        :returns: Name of axis associated with element if key is found.
            Returns None if key is not found.
        """
        for variable_name in self.getVariableNames():
            element_key = self.getKey(prefix, variable_name)
            if key == element_key:
                return variable_name
        return None

    def changeTimeEvolution(self, name: str) -> None:
        """
        Disable input field for initial conditions if time-evolution type is set to "Equilibrium" or if initial condition is set to equilibrium.
        Disable checkbox if time-evolution type is set to "Equilibrium".
        Enable elements otherwise.
        
        :param self: `~Layout.MainWindow.MainWindowRunner` to retrieve elements from
        :param name: name of variable associated with elements
        """
        time_evolution_type = self.getValue(self.getKey("time_evolution_type", name))
        is_equilibrium = time_evolution_type == "Equilibrium"
        input_field = self.getElements(self.getKey("initial_condition_value", name))

        checkbox_key = self.getKey("initial_condition_equilibrium", name)
        checkbox = self.getElements(checkbox_key)
        is_initial_equilibrium = self.getValue(checkbox_key)

        input_field.update(disabled=is_equilibrium or is_initial_equilibrium)
        checkbox.update(disabled=is_equilibrium)

    def getFunctionYMLs(self) -> List[str]:
        """
        Get YML filenames to add functions to model.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filenames from
        """
        function_ymls = self.function_ymls
        if isinstance(function_ymls, str):
            return [function_ymls]
        elif isinstance(function_ymls, list):
            return function_ymls
        else:
            raise TypeError("function_ymls must be of type str or list")

    def getParameters(
            self, names: Union[str, List[str]] = None, form: str = "quantity"
    ) -> Optional[Union[Quantity, Dict[str, Quantity]]]:
        """
        Get parameter(s), including value and unit
        
        __Recursion Base__
            return quantity, value, or unit of single parameter: names [str]
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retreive.
            Defaults to all parameters.
        :param form: form of quantity to retrieve. Each desired option must be a substring of this argument.
            "quantity": retrieve value and unit
            "unit": retrieve only unit
            "value": retrieve only value
            "base" prefix or suffix: convert to base units (SI)
            
        :returns: Quantity for single parameter if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is str
            List of quantities if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is list.
            None if parameter is not included in window.
        """
        if isinstance(names, str):
            parameters = self.parameters
            if names in parameters.keys():
                quantity = self.parameters[names]
                if "base" in form:
                    quantity = quantity.to_base_units()
                if "quantity" in form:
                    return quantity
                elif "value" in form:
                    return quantity.magnitude
                elif "unit" in form:
                    return quantity.units
                raise ValueError("Invalid entry for form input")
            else:
                return None
        elif isinstance(names, list):
            return {name: self.getParameters(names=name, form=form) for name in names}
        elif names is None:
            return self.getParameters(names=self.getParameterNames(), form=form)
        else:
            raise RecursiveTypeError(names)

    def setParameters(self, name: str, quantity: Quantity) -> None:
        """
        Set or overwrite parameter quantity stored in runner.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store quantity in
        :param name: name of parameter to set/overwrite
        :param quantity: quantity to set parameter as
        """
        old_quantity = self.getParameters(names=name)
        if isinstance(quantity, type(old_quantity)):
            self.parameters[name] = quantity
            self.updateParameterLabels(name)
        elif old_quantity is None:
            pass
        else:
            raise TypeError("new quantity must be same class as old quantity")

    def getParameterFields(self, names: Union[str, List[str]] = None) -> Union[str, List[str]]:
        """
        Get values for parameter-value input-fields.
        Uses present state of window.
        
        __Recursion Base__
            get field value for single parameter: names [str]
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve values from
        :param names: name(s) of parameter(s) to retrieve values for.
            Defaults to all parameters.
        """
        if isinstance(names, str):
            key = self.getKey("parameter_field", names)
            field_value = self.getValue(key)
            return field_value
        elif isinstance(names, list):
            return [self.getParameterFields(names=name) for name in names]
        elif names is None:
            return self.getParameterFields(names=self.getParameterNames())
        else:
            raise RecursiveTypeError(names)

    def updateParameters(self, names: Union[str, List[str]] = None) -> None:
        """
        Update parameter quantity(s) stored in runner.
        Uses present state of window.
        
        __Recursion Base__
            update quantity associated with single parameter: names [str]
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to update parameters in
        :param names: name(s) of parameter(s) to update.
            Defaults to all parameters.
        """
        if isinstance(names, str):
            field_value = self.getParameterFields(names=names)
            if field_value != '':
                old_unit, new_value = self.getParameters(names=names, form="unit"), float(field_value)
                new_quantity = new_value * old_unit
                self.setParameters(names, new_quantity)
        elif isinstance(names, list):
            for name in names:
                self.updateParameters(names=name)
        elif names is None:
            self.updateParameters(names=self.getParameterNames())
        else:
            raise TypeError("names input be str or list")

    def updateParameterLabels(self, names: Union[str, List[str]] = None) -> None:
        """
        Update quantity (value and unit) label for parameter.
        
        __Recursion Base__
            update label for single parameter: names [str]
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to update label in
        :param names: name(s) of parameter(s) to update labels for
        """
        if isinstance(names, str):
            label_key = self.getKey("quantity_label", names)
            label_element = self.getElements(label_key)
            new_label = formatQuantity(self.getParameters(names=names))
            label_element.update(new_label)
        elif isinstance(names, list):
            for name in names:
                self.updateParameterLabels(names=name)
        elif names is None:
            self.updateParameterLabels(names=self.getParameterFields())
        else:
            raise RecursiveTypeError(names)

    def loadParametersFromFile(self, filenames: str = None, choose_parameters: bool = True) -> None:
        """
        Load and stored parameter quantities from file.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store quantities in
        :param filenames: name(s) of file(s) to load parameters from
        :param choose_parameters: set True to allow user to choose which parameters to actually load.
            Set False to automatically load all parameters from file(s)
        """
        if filenames is None:
            file_types = (("YML", "*.yml"), ("YAML", "*.yaml"), ("Plain Text", "*.txt"), ("ALL Files", "*.*"),)
            kwargs = {
                "message": "Enter Filename to Load",
                "title": "Load Parameters",
                "file_types": file_types,
                "multiple_files": True
            }
            filenames = sg.PopupGetFile(**kwargs)
            if filenames is None:
                return None
            elif filenames is not None:
                filenames = filenames.split(';')
        elif isinstance(filenames, str):
            filenames = [filenames]

        load_quantities = {
            name: quantity
            for filename in filenames
            for name, quantity in YML.readParameters(filename).items()
        }

        if choose_parameters:
            runner = ChooseParametersWindowRunner("Choose Parameters to Load", quantities=load_quantities)
            parameter_names = runner.getChosenParameters()
        else:
            parameter_names = self.getParameterNames()

        for name, quantity in load_quantities.items():
            if name in parameter_names:
                self.setParameters(name, quantity)

    def getFreeParameterValues(self) -> Tuple[str, Dict[str, Tuple[float, float, int, Quantity]]]:
        """
        Open window allowing user to set minimum, maximum, and step count for each selected free parameter.
        Uses present state of window.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve free parameters from
        """
        free_parameter_names = self.getParameterNames(species="Free")
        free_parameter_quantities = self.getParameters(names=free_parameter_names)

        kwargs = {
            "name": "Set Values for Free Parameters",
            "free_parameter_quantities": free_parameter_quantities
        }
        set_free_parameters_window = SetFreeParametersWindowRunner(**kwargs)
        event, free_parameter_values = set_free_parameters_window.runWindow()
        print(event, free_parameter_values)
        if event == "Submit":
            free_parameter_values = {free_parameter_name: tuple(
                [*free_parameter_values[free_parameter_name], free_parameter_quantities[free_parameter_name]]
            ) for free_parameter_name in free_parameter_names}
        return event, free_parameter_values

    def openSimulationWindow(self) -> None:
        """
        Open simulation window allowing user to run and analyze model.
        Uses present state of window.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to call :class:`~Layout.SimulationWindow.SimulationWindowRunner`
        """
        event, free_parameter_values = self.getFreeParameterValues()
        if event == "Submit":
            model = self.getModel()
            kwargs = {
                "name": "Run Simulation for Model",
                "model": model,
                "free_parameter_values": free_parameter_values,
                "plot_choices": self.getPlotChoices(model=model)
            }
            simulation_window = SimulationWindowRunner(**kwargs)
            simulation_window.runWindow()
