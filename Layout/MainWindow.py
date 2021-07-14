from __future__ import annotations

from os.path import dirname, join
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from numpy import ndarray
from pint import Quantity

from CustomErrors import RecursiveTypeError
from Function import Function, Independent, Model, Parameter, readFunctions, readParameters
from Layout.ChooseParametersWindow import ChooseParametersWindowRunner
from Layout.Layout import Element, Layout, Row, Tab, TabGroup, TabRow, TabbedWindow, WindowRunner, generateCollapsableSection
from Layout.SetFreeParametersWindow import SetFreeParametersWindowRunner
from Layout.SimulationWindow import SimulationWindowRunner
from YML import getDimensions, getStates, readLayout, readStates
from macros import expression2png, formatQuantity, getTexImage, recursiveMethod, unique

tet_types = ("Temporal", "Equilibrium", "Constant", "Function")
p_types = ("Constant", "Free")


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

    def getTimeEvolutionTypes(self, **kwargs) -> Union[str, Iterable[str]]:
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

        self.time_evolution_types = readStates("time_evolution_types")
        self.initial_conditions = readStates("initial_condition")

        self.variable_rows = []
        self.addVariableRows(variable_names)

    def addVariableRows(
            self, names: Union[str, Iterable[str]]
    ) -> Union[TimeEvolutionVariableRow, Iterable[TimeEvolutionVariableRow]]:
        """
        Add rows corresponding to variable names.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionTab` to add rows to
        :param names: name(s) of variable(s) to add rows for
        :returns: New row added if names is str.
            List of new rows if names is list.
        """

        def get(name: str) -> TimeEvolutionVariableRow:
            new_row = TimeEvolutionVariableRow(name, self)
            self.variable_rows.append(new_row)
            return new_row

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

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

        def get(name: str) -> float:
            return self.initial_conditions[name]

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": dict,
            "default_args": list(self.initial_conditions.keys())
        }
        return recursiveMethod(**kwargs)

    def getTimeEvolutionTypes(self, name: str, index: int = None) -> Union[str, Iterable[str]]:
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
        #. Combobox to choose file to load parameter from
    """

    def __init__(
            self,
            name: str,
            section: ParameterSection,
            parameters: List[Parameter],
            parameter_types: Tuple[str] = ("Constant", "Free")
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterRow`.

        :param name: name of tab group
        :param section: :class:`~Layout.MainWindow.ParameterSection` that row is stored in
        :param parameters: dictionary of filepaths for file containing parameter.
            Key is name of filestem for file.
            Value is filepath for file.
        :param parameter_types: collection of types that each parameter can take on
        """
        super().__init__(name, section.getTab())
        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        window_object.addParameterNames(names=name)

        self.parameters = parameters
        self.section = section
        self.parameter_types = parameter_types

        elements = [
            self.getNameLabel(),
            self.getQuantityLabel(),
            self.getValueInputElement(),
            self.getParameterTypeElement(),
            self.getChooseFileElement(),
            self.getCustomCheckbox()
        ]
        self.addElements(elements)

    def getStems(self) -> List[str]:
        """
        Get filestems for files containing parameter.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve filestems from
        """
        return [parameter.getStem() for parameter in self.getParameters()]

    def getQuantities(self) -> List[Quantity]:
        """
        Get filepaths for files containing parameter.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve filepaths from
        """
        return [parameter.getQuantity() for parameter in self.getParameters()]

    def getParameters(self) -> List[Parameter]:
        """
        Get parameters for row.

        :param self: `~Layout.MainWindow.ParameterRow` to retrieve parameters from
        """
        return self.parameters

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

    def getCustomCheckbox(self) -> sg.Checkbox:
        """
        Get checkbox allowing user to use custom value for parameter.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve checkbox from
        """
        name = self.getName()
        kwargs = {
            "text": "Custom",
            "default": False,
            "tooltip": f"Checked if parameter {name:s} is from custom value (i.e. input field)",
            "enable_events": False,
            "disabled": True,
            "size": (None, None),  # dim
            "key": self.getKey("custom_parameter", name)
        }
        return sg.Checkbox(**kwargs)

    def getChooseFileElement(self) -> sg.Combo:
        """
        Get element allowing user to choose which file to load parameter from.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve element from
        """
        filestems = self.getStems()
        stem_count = len(filestems)
        kwargs = {
            "values": filestems,
            "default_value": filestems[-1],
            "tooltip": f"Choose file to load parameter {self.getName():s} from",
            "enable_events": True,
            "disabled": stem_count == 1,
            "size": self.getDimensions(name="parameter_stem_combobox"),
            "key": self.getKey("parameter_filestem", self.getName())
        }
        return sg.Combo(**kwargs)

    def getNameLabel(self) -> Union[sg.Text, sg.Image]:
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
        quantity = self.getQuantities()[-1]
        kwargs = {
            "text": formatQuantity(quantity),
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

    def __init__(self, name: str, tab: ParameterTab, name2params: Dict[str, List[Parameter]]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterSection`.

        :param name: name of tab
        :param tab: tab that section in stored in
        :param name2params: dictionary dictating how to set up each row within section.
            Key is name of parameter in section.
            Value is dictionary from stems to paths for parameter.
                Key is filestems for files containing parameter.
                Value is filepaths for files.
        """
        super().__init__(tab.getWindowObject())
        self.name = name
        self.tab = tab

        self.parameter_rows = []
        for parameter_name, parameters in name2params.items():
            self.parameter_rows.append(ParameterRow(parameter_name, self, parameters))

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

    def __init__(self, name: str, window: MainWindow, blueprints: Tuple[Dict[str, List[str]], dict]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterTab`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprints: Tuple of blueprints.
            First element is dictionary dictating how to set up each section within tab.
            Key is name of section in tab.
            Value is names of parameters within this section.
            Second element is 2-level dictionary of parameters to filepaths.
            First key is name of parameter.
            Second key is filestems for files containing parameter.
            Value is filepaths for files.
        """
        super().__init__(name, window)
        layout_blueprint, name2params = blueprints

        if isinstance(layout_blueprint, list):
            layout_blueprint = {
                name: layout_blueprint
            }

        self.sections = []
        for section_name, parameter_names in layout_blueprint.items():
            name2params_sub = {
                parameter_name: name2params[parameter_name]
                for parameter_name in parameter_names
            }
            self.sections.append(ParameterSection(section_name, self, name2params_sub))

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

    def __init__(self, name: str, window: MainWindow, blueprints: Tuple[dict, dict], suffix_layout: Layout) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.ParameterTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprints: Tuple of blueprints.
            First element is dictionary dictating how to set up each tab within tab group.
            Key is name of tab in tab group.
            Value is names of parameters within this tab.
            Second element is 2-level dictionary of parameters to filepaths.
            First key is name of parameter.
            Second key is filestems for files containing parameter.
            Value is filepaths for files.
        :param suffix_layout: layout shared by all tabs in tab group
        """
        tabgroup_blueprint, name2params = blueprints

        tabs = []
        tab_names = tabgroup_blueprint.keys()
        for tab_name in tab_names:
            tab_blueprint = tabgroup_blueprint[tab_name]
            if isinstance(tab_blueprint, list):
                tab_blueprint = {
                    tab_name: tab_blueprint
                }
            parameter_namess = []
            for parameter_names in tab_blueprint.values():
                parameter_namess.extend(parameter_names)
            name2params_sub = {
                parameter_name: name2params[parameter_name]
                for parameter_name in parameter_namess
            }
            tabs.append(ParameterTab(tab_name, window, (tab_blueprint, name2params_sub)))
        super().__init__(tabs, name=name, suffix_layout=suffix_layout)


class FunctionRow(TabRow):
    """
    Row to display function info.

    This contains
        #. Label for name of function
        #. Label for expression of function
        #. Combobox to choose filestem to load function from
    """

    def __init__(self, name: str, tab: FunctionTab, functions: List[Function]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionRow`.

        :param name: name of function
        :param tab: tab that row is stored in
        :param functions: dictionary from of functions.
            Key is filestem for file.
            Value is function object loaded from file.
        """
        super().__init__(name, tab)
        self.image_folder = "tex_eq"
        self.functions = functions

        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        window_object.addFunctionNames(name)

        elements = [
            self.getRowLabel(),
            self.getExpressionLabel(),
            self.getChooseFileElement()
        ]
        self.addElements(elements)

    def getImageFoldername(self, filestem: str) -> str:
        """
        Get folderpath to save or load image from.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve folderpath from
        :param filestem: stem of file to retrieve folderpath for
        """
        top_folder = self.image_folder
        folderpath = join(top_folder, filestem)
        return folderpath

    def generatePngExpressions(self) -> List[str]:
        """
        Generate PNG images for each function corresponding to filestem.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to generate images for
        :returns: List of filepaths for generate images.
        """
        functions = self.getFunctions()
        filestems = self.getStems()
        name = self.getName()

        image_filepaths = []
        for index in range(len(functions)):
            function = functions[index]
            filestem = filestems[index]
            kwargs = {
                "name": name,
                "expression": function.getExpression(generations=0),
                "var2tex": "var2tex.yml",
                "folder": self.getImageFoldername(filestem),
                "filename": f"{name:s}.png"
            }
            image_filepaths.append(expression2png(**kwargs))
        return image_filepaths

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

    def getStems(self) -> List[str]:
        """
        Get filestems for files containing function.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve filestems from
        """
        return [function.getStem() for function in self.getFunctions()]

    def getFunctions(self) -> List[Function]:
        """
        Get :class:`~Function.Function` in order of filestems.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve filepaths from
        """
        return self.functions

    def getChooseFileElement(self) -> sg.Combo:
        """
        Get element allowing user to choose which file to load function.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve element from
        """
        filestems = self.getStems()
        stem_count = len(filestems)
        kwargs = {
            "values": filestems,
            "default_value": filestems[-1],
            "tooltip": f"Choose file to load function {self.getName():s} from",
            "enable_events": True,
            "disabled": stem_count == 1,
            "size": self.getDimensions(name="function_stem_combobox"),
            "key": self.getKey("function_filestem", self.getName())
        }
        return sg.Combo(**kwargs)

    def getExpressionLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get element to display expression of function.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve label from
        """
        name = self.getName()
        image_filepath = self.generatePngExpressions()[-1]
        image_folder = dirname(image_filepath)

        kwargs = {
            "name": name,
            "size": self.getDimensions(name="expression_label"),
            "tex_folder": image_folder,
            "key": self.getKey("function_expression", name)
        }
        return getTexImage(**kwargs)


class FunctionTab(Tab):
    """
    Tab to organize rows for functions.

    This contains
        #. Header of labels to indicate purpose of each element in tab columns
        #. :class:`~Layout.MainWindow.FunctionRow` for each function in tab

    :ivar function_rows: list of :class:`~Layout.MainWindow.FunctionRow`, one for each function in tab
    """

    def __init__(
            self,
            name: str,
            window: MainWindow,
            blueprint: Dict[str, List[Function]]
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionTab`.

        :param name: name of tab
        :param window: window that tab is stored in
        :param blueprint: 2-level dictionary dictating which rows to add into tab.
            First key is name of function to add.
            Second key is filestems for files containing function.
            Value is function generated from file.
        """
        super().__init__(name, window)
        function_names = list(blueprint.keys())
        self.function_rows = [
            FunctionRow(function_name, self, blueprint[function_name])
            for function_name in function_names
        ]

    def getFunctionRows(self) -> List[FunctionRow]:
        """
        Get function rows stored in tab.

        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve rows from
        """
        return self.function_rows

    def getHeaderRow(self) -> Row:
        """
        Get header row for single function tab.

        :param self: :class:`~Layout.MainWindow.FunctionTab` to retrieve row from
        """
        row = Row(window=self.getWindowObject())
        texts = ["Function", "Expression"]
        dimension_keys = [f"function_header_row_{string:s}" for string in ["function_label", "expression_label"]]
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
    """
   Tabgroup for function tabs.

   This contains
       #. :class:`~Layout.MainWindow.FunctionTab` for each group of functions
   """

    def __init__(self, name: str, window: MainWindow, blueprints: Tuple[Dict[str, List[str]], dict]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionTabGroup`.

        :param name: name of tab group
        :param window: :class:`~Layout.MainWindow.MainWindow` that tab group is stored in
        :param blueprints: Tuple of blueprints.
            First element is dictionary dictating how to set up each tab within tab group.
            Key is name of tab in tab group.
            Value is names of functions within this tab.
            Second element is 2-level dictionary of functions to filepaths.
            First key is name of function.
            Second key is filestems for files containing function.
            Value is filepaths for files.
        """
        tabgroup_blueprint, name2funcs = blueprints
        tabs = []
        tab_names = tabgroup_blueprint.keys()
        for tab_name in tab_names:
            function_names = tabgroup_blueprint[tab_name]
            tab_blueprint = {
                function_name: name2funcs[function_name]
                for function_name in function_names
            }
            tabs.append(FunctionTab(tab_name, window, tab_blueprint))
        super().__init__(tabs, name=name)


class MainWindow(TabbedWindow):
    """
    Primary window to run program.

    This contains
        #. :class:`~Layout.MainWindow.TimeEvolutionTabGroup` to allow user to set time-evolution properties for each variable
        #. :class:`~Layout.MainWindow.ParameterTabGroup` to allow user to set properties for each parameter
    """

    def __init__(
            self,
            name: str,
            runner: MainWindowRunner,
            blueprint: dict,
            stem2name2obj: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.MainWindow`.

        :param name: title of window
        :param runner: :class:`~Layout.MainWindow.MainWindowRunner` that window is stored in
        :param blueprint: dictionary indicating how to set up elements in window
        :param stem2name2obj: dictionary of filepaths.
            Key is species of quantity ("functions", "parameters").
            Value is list of filepaths containing this species of quantity.
        """
        dimensions = {
            "window": getDimensions(["main_window", "window"]),
            "time_evolution_tab": getDimensions(["main_window", "time_evolution_tab", "tab"]),
            "evolution_type_text": getDimensions(
                ["main_window", "time_evolution_tab", "header_row", "evolution_type_text"]
            ),
            "variable_text": getDimensions(["main_window", "time_evolution_tab", "header_row", "variable_text"]),
            "initial_condition_text": getDimensions(
                ["main_window", "time_evolution_tab", "header_row", "initial_condition_text"]
            ),
            "evolution_type_combobox": getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "evolution_type_combobox"]
            ),
            "variable_label": getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "variable_label"]
            ),
            "initial_condition_input_field": getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "initial_condition_input_field"]
            ),
            "initial_equilibrium_checkbox": getDimensions(
                ["main_window", "time_evolution_tab", "variable_row", "initial_equilibrium_checkbox"]
            ),
            "parameter_tab": getDimensions(["main_window", "parameter_tab", "tab"]),
            "parameter_section": getDimensions(["main_window", "parameter_tab", "parameter_section"]),
            "parameter_header_label": getDimensions(
                ["main_window", "parameter_tab", "header_row", "section_label"]
            ),
            "parameter_label": getDimensions(["main_window", "parameter_tab", "parameter_row", "parameter_label"]),
            "parameter_value_label": getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_value_label"]
            ),
            "parameter_value_input_field": getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_value_input_field"]
            ),
            "parameter_type_combobox": getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_type_combobox"]
            ),
            "parameter_stem_combobox": getDimensions(
                ["main_window", "parameter_tab", "parameter_row", "parameter_stem_combobox"]
            ),
            "function_tab": getDimensions(["main_window", "function_tab", "tab"]),
            "function_label": getDimensions(["main_window", "function_tab", "function_row", "function_label"]),
            "expression_label": getDimensions(["main_window", "function_tab", "function_row", "expression_label"]),
            "function_stem_combobox": getDimensions(
                ["main_window", "function_tab", "function_row", "function_stem_combobox"]
            ),
            "function_header_row_function_label": getDimensions(
                ["main_window", "function_tab", "header_row", "function_label"]
            ),
            "function_header_row_expression_label": getDimensions(
                ["main_window", "function_tab", "header_row", "expression_label"]
            )
        }
        super().__init__(name, runner, dimensions=dimensions)

        self.stem2name2func = stem2name2obj["functions"]
        self.stem2name2param = stem2name2obj["parameters"]

        self.variable_names = []
        self.function_names = []
        self.parameter_names = []
        self.blueprint = blueprint

        tabs = [
            self.getTimeEvolutionTypeTab(),
            self.getParameterTab(),
            self.getFunctionTab()
        ]
        self.addTabs(tabs)

    def getName2Functions(self) -> Dict[str, Dict[str, Function]]:
        """
        Get dictionary from function name to filestem to function object.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve dictionary from
        """
        stem2name2func = self.stem2name2func
        name2stem2func = {}
        for stem, name2func in stem2name2func.items():
            for name, func in name2func.items():
                if name not in name2stem2func:
                    name2stem2func[name] = {}
                name2stem2func[name][stem] = func
        name2funcs = {name: list(name2stem2func[name].values()) for name in name2stem2func.keys()}
        return name2funcs

    def getName2Parameters(self) -> Dict[str, Dict[str, Quantity]]:
        """
        Get dictionary from parameter name to filestem to quantity object.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve dictionary from
        """
        stem2name2param = self.stem2name2param
        name2stem2param = {}
        for stem, name2quant in stem2name2param.items():
            for name, quant in name2quant.items():
                if name not in name2stem2param:
                    name2stem2param[name] = {}
                name2stem2param[name][stem] = quant
        name2params = {name: list(name2stem2param[name].values()) for name in name2stem2param.keys()}
        return name2params

    def getVariableNames(self) -> List[str]:
        """
        Get names of variables included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve names from
        """
        return self.variable_names

    def addVariableNames(self, names: Union[str, Iterable[str]]) -> Union[str, Iterable[str]]:
        """
        Add names of variables included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to add names into
        :param names: name(s) of variable(s) to add
        :returns: Name of new variable added if :paramref:`~Layout.MainWindow.MainWindow.addVariableNames.names` is str.
            List of new variables added if :paramref:`~Layout.MainWindow.MainWindow.addVariableNames.names` is list.
        """

        def add(name: str) -> str:
            new_variable_name = name
            self.variable_names.append(name)
            return new_variable_name

        kwargs = {
            "base_method": add,
            "args": names,
            "valid_input_types": str,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

    def getFunctionNames(self) -> List[str]:
        """
        Get names of functions included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve names from
        """
        return self.function_names

    def addFunctionNames(self, names: Union[str, Iterable[str]]) -> Union[str, Iterable[str]]:
        """
        Add names of functions included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to add names into
        :param names: name(s) of function(s) to add
        :returns: Name of new function added
            if :paramref:`~Layout.MainWindow.MainWindow.addFunctionNames.names` is str.
            List of new functions added if :paramref:`~Layout.MainWindow.MainWindow.addFunctionNames.names` is list.
        """

        def add(name: str) -> str:
            new_function_name = name
            self.function_names.append(name)
            return new_function_name

        kwargs = {
            "base_method": add,
            "args": names,
            "valid_input_types": str,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

    def getParameterNames(self) -> List[str]:
        """
        Get names of parameters included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve names from
        """
        return self.parameter_names

    def addParameterNames(self, names: Union[str, Iterable[str]]) -> Union[str, Iterable[str]]:
        """
        Add names of parameters included in window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to add names into
        :param names: name(s) of parameter(s) to add
        :returns: Name of new parameter added if
            :paramref:`~Layout.MainWindow.MainWindow.addParameterNames.names` is str.
            List of new parameters added if :paramref:`~Layout.MainWindow.MainWindow.addParameterNames.names` is list.
        """

        def add(name: str) -> str:
            new_parameter_name = name
            self.parameter_names.append(name)
            return new_parameter_name

        kwargs = {
            "base_method": add,
            "args": names,
            "valid_input_types": str,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

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
        param_stems = tuple(self.stem2name2param.keys())
        param_stems_keyed = [
            stem + "::set_parameter_filestems_to"
            for stem in param_stems
        ]
        param_types_keyed = [
            ptype + "::set_parameter_types_to"
            for ptype in p_types
        ]
        func_stems = tuple(self.stem2name2func.keys())
        func_stems_keyed = [
            stem + "::set_function_filestems_to"
            for stem in func_stems
        ]
        tet_types_keyed = [
            tet_type + "::set_time_evolution_types_to"
            for tet_type in tet_types
        ]
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
                    "Parameter",
                    [
                        "Filestems",
                        param_stems_keyed,
                        "Types",
                        param_types_keyed
                    ],
                    "Time-Evolution Types to...",
                    tet_types_keyed,
                    "Function Filestems",
                    func_stems_keyed
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

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve layout from
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

    def getParameterTab(self) -> sg.Tab:
        """
        Get time-evolution tabgroup as tab, containing tab for each parameter group

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve tabgroup from
        """
        common = Layout(rows=Row(elements=sg.Button("Update Parameters")))
        blueprints = (self.getBlueprints("parameters"), self.getName2Parameters())
        tabgroup = ParameterTabGroup("Parameters", self, blueprints, common)
        tab = tabgroup.getAsTab()
        return tab

    def getFunctionTab(self) -> sg.Tab:
        """
        Get function tabgroup as tab, containing tab for each function group

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve tabgroup from
        """
        blueprints = (self.getBlueprints("functions"), self.getName2Functions())
        tabgroup = FunctionTabGroup("Function", self, blueprints)
        tab = tabgroup.getAsTab()
        return tab


class MainWindowRunner(WindowRunner):
    """
    Runner for :class:`~Layout.MainWindow.MainWindow`.

    :ivar function_ymls: filename(s) of YML files, containing info about functions for model
    :ivar parameters: dictionary of parameter quantities.
        Key is name of parameter.
        Value is quantity containing value and unit for parameter.
    """

    def __init__(
            self,
            name: str,
            parameter_filepaths: Union[str, Iterable[str]],
            function_filepaths: Union[str, Iterable[str]],
            time_evolution_layout: str,
            parameter_layout: str,
            function_layout: str
    ) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.MainWindowRunner`.

        :param name: name of window
        :param parameter_filepaths: name(s) for file(s), containing info about parameters in model
        :param function_filepaths: name(s) for file(s), containing info about functions in model
        :param time_evolution_layout: name of file containing layout for time-evolution tab
        :param parameter_layout: name of file containing layout for parameter-input tab
        :param function_layout: name of file containing layout for function tab
        """
        self.custom_param = {}
        self.stem2name2param = {
            Path(filepath).stem: readParameters(filepath)
            for filepath in parameter_filepaths
        }
        self.stem2name2func = {
            Path(filepath).stem: readFunctions(filepath)
            for filepath in function_filepaths
        }

        blueprints = {
            "time_evolution": readLayout(time_evolution_layout),
            "parameters": readLayout(parameter_layout),
            "functions": readLayout(function_layout),
        }
        stem2name2obj = {
            "functions": self.stem2name2func,
            "parameters": self.stem2name2param
        }
        window = MainWindow(name, self, blueprints, stem2name2obj)
        super().__init__(window)

        self.getVariableNames = window.getVariableNames
        self.getFunctionNames = window.getFunctionNames
        # self.getParameterNames = window.getParameterNames

    def runWindow(self) -> None:
        window = self.getWindow()
        while True:
            event, self.values = window.read()
            print(event)
            if event in [sg.WIN_CLOSED, event == 'Exit']:
                break
            psl_pre = self.getPrefix("parameter_subgroup_label")
            tet_pre = self.getPrefix("time_evolution_type")
            ice_pre = self.getPrefix("initial_condition_equilibrium")
            ff_pre = self.getPrefix("function_filestem")
            pf_pre = self.getPrefix("parameter_filestem")
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
                elif "::set_parameter_filestems_to" in event:
                    parameter_filestem = event.replace("::set_parameter_filestems_to", '')
                    self.setElementsWithPrefix("parameter_filestem", parameter_filestem)
                elif "::set_function_filestems_to" in event:
                    function_filestem = event.replace("::set_function_filestems_to", '')
                    self.setElementsWithPrefix("function_filestem", function_filestem)
            elif psl_pre in event:
                key = event.replace(psl_pre, self.getPrefix("parameter_subgroup_section"))
                self.toggleVisibility(key)
            elif tet_pre in event or ice_pre in event:
                prefix = "time_evolution_type" if tet_pre in event else "initial_condition_equilibrium"
                variable_name = self.getVariableNameFromElementKey(event, prefix)
                self.changeTimeEvolution(variable_name)
            elif ff_pre in event:
                ff_pre_sep = self.getPrefix("function_filestem", with_separator=True)
                function_name = event.replace(ff_pre_sep, '')
                self.updateFunctionExpressions(names=function_name)
            elif pf_pre in event:
                pf_pre_sep = self.getPrefix("parameter_filestem", with_separator=True)
                parameter_name = event.replace(pf_pre_sep, '')
                self.updateParametersFromStems(names=parameter_name)
            elif event == "Update Parameters":
                self.updateParametersFromFields()
            elif event == self.getKey("open_simulation"):
                self.openSimulationWindow()
        window.close()

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
            "Variable": ['t'] + model.getVariables(return_type=str),
            "Function": [function.getName() for function in model.getFunctions(filter_type=Independent)],
            "Parameter": self.getParameterNames(parameter_types="Free")
        }
        return plot_choices

    def getModel(self) -> Model:
        """
        Get model for window.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve model from
        """
        model = Model(functions=self.getFunctions(), parameters=self.getParameters())
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
        window = self.getWindow()
        keys = self.getKeyList(prefix)
        for key in keys:
            element = self.getElements(key)
            if value in vars(element)["Values"]:
                element.update(value)
                window.write_event_value(key, value)

    def getInitialConditions(self, names: Union[str, Iterable[str]] = None) -> Union[float, ndarray, str]:
        """
        Get initial conditions for variables.
        Uses present state of window.

        __Recursion Base__
            get initial condition value for single variable: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve conditions from
        :param names: name(s) of variable(s) to retrieve conditions for.
            Defaults to all variables.
        :returns: Initial value if variable does not begin in equilibrium.
            "Equilibrium" if variable begins in equilibrium.
        """

        def get(name: str) -> Union[float, str]:
            time_evolution_type = self.getValue(self.getKey("time_evolution_type", name))
            is_equilibrium = time_evolution_type == "Equilibrium"
            is_initial_equilibrium = self.getValue(self.getKey("initial_condition_equilibrium", name)) or is_equilibrium

            if is_initial_equilibrium:
                return "Equilibrium"
            else:
                input_field_key = self.getKey("initial_condition_value", name)
                try:
                    value = float(self.getValue(input_field_key))
                except AttributeError:
                    value = float(getStates("initial_condition", name))
                return value

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getVariableNames()
        }
        return recursiveMethod(**kwargs)

    def getTimeEvolutionTypes(self, names: Union[str, Iterable[str]] = None) -> Union[str, Iterable[str]]:
        """
        Get selected time-evolution type for variable.
        Get default time-evolution type if none selected.
        Uses present state of window.

        __Recursion Base__
            get time-evolution selection for single variable: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve time-evolution types from
        :param names: name(s) of variable(s) to retrieve time-evolution types for.
            Defaults to all variables.
        """

        def get(name: str) -> str:
            key = self.getKey("time_evolution_type", name)
            try:
                return self.getValue(key)
            except AttributeError:
                return getStates("time_evolution_types", name)

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getVariableNames()
        }
        return recursiveMethod(**kwargs)

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

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve elements from
        :param name: name of variable associated with elements
        """
        time_evolution_type = self.getValue(self.getKey("time_evolution_type", name))
        is_equilibrium = time_evolution_type == "Equilibrium"
        input_field_key = self.getKey("initial_condition_value", name)
        input_field = self.getElements(input_field_key)

        checkbox_key = self.getKey("initial_condition_equilibrium", name)
        checkbox = self.getElements(checkbox_key)
        is_initial_equilibrium = self.getValue(checkbox_key)

        is_either_equilibrium = is_equilibrium or is_initial_equilibrium
        input_field.update(disabled=is_either_equilibrium)
        checkbox.update(disabled=is_equilibrium)

    def getFunctionNames(self, filestem: str = None) -> List[str]:
        """
        Get name(s) of function(s) stored in window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve names from
        :param filestem: stem for file, acts as optional filter
        """

        if isinstance(filestem, str):
            filtered_names = [
                function_name
                for function_name in self.getFunctionNames()
                if self.getChosenFunctionStem(function_name) == filestem
            ]
            return filtered_names
        elif filestem is None:
            stem2name2func = self.stem2name2func
            function_names = [
                function_name
                for name2func in stem2name2func.values()
                for function_name in name2func.keys()
            ]
            return unique(function_names)
        else:
            raise TypeError("filestem must be of type str")

    def getFunctions(self, names: Union[str, Iterable[str]] = None) -> Union[Function, Iterable[Function]]:
        """
        Get function object from filestem and name.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve function from
        :param names: name(s) of function(s) to retrieve.
            Defaults to all loaded functions.
        """

        def get(name: str) -> str:
            filestem = self.getChosenFunctionStem(name)
            function = self.stem2name2func[filestem][name]
            return function

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getFunctionNames()
        }
        return recursiveMethod(**kwargs)

    def getChosenFunctionStem(self, name: str) -> str:
        """
        Get stem of file to load function from.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filename from
        :param name: name of function to retrieve filename for
        """
        combobox_key = self.getKey("function_filestem", name)
        filestem = self.getValue(combobox_key)
        return filestem

    def updateFunctionExpressions(self, names: Union[str, Iterable[str]]) -> None:
        """
        Update function expression from selected file.

        :param self: :class`~Layout.MainWindow.MainWindowRunner` to update expression in
        :param names: name(s) of function(s) to update expression for.
            Defaults to all functions.
        """

        def update(name: str) -> None:
            function_filestem = self.getChosenFunctionStem(name)
            image_folder = join("tex_eq", function_filestem)
            image_filename = '.'.join((name, "png"))
            image_filepath = join(image_folder, image_filename)
            image_data = open(image_filepath, 'rb').read()
            image_expression = self.getElements(self.getKey("function_expression", name))
            image_size = vars(image_expression)["Size"]
            image_expression.update(data=image_data, size=image_size)

        kwargs = {
            "base_method": update,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getFunctionNames()
        }
        return recursiveMethod(**kwargs)

    def getChosenParameterStem(self, name: str) -> str:
        """
        Get stem of file to load parameter from.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filestem from
        :param name: name of parameter to retrieve filestem for
        """
        combobox_key = self.getKey("parameter_filestem", name)
        filestem = self.getValue(combobox_key)
        return filestem

    def getParameterFromStem(self, name: str, filestem: str) -> Parameter:
        """
        Get parameter from file.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve quantity from
        :param name: name of parameter to retrieve quantity for
        :param filestem: stem for file to retrieve quantity from
        """
        return self.stem2name2param[filestem][name]

    def getChosenParameterType(self, name: str) -> str:
        """
        Get type to treat parameter as.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve type from
        :param name: name of parameter to retrieve type for
        """
        combobox_key = self.getKey("parameter_type", name)
        parameter_type = self.getValue(combobox_key)
        return parameter_type

    def getCustomParameterQuantities(
            self, names: Union[str, Iterable[str]] = None
    ) -> Union[Quantity, Dict[str, Quantity]]:
        """
        Get custom value for parameter.

        :param self: `~Layout.MainWindow.MainWindowRunner` to retrieve value from
        :param names: name(s) of parameter(s) to retrieve quantity(s) for
        """

        def get(name: str) -> str:
            return self.custom_param[name]

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": list(self.custom_param.keys())
        }
        return recursiveMethod(**kwargs)

    def isCustomParameter(self, name: str) -> bool:
        """
        Determine whether or not parameter should use custom value.

        :param self: `~Layout.MainWindow.MainWindowRunner` to retrieve parameter property from
        :param name: name of parameter
        :returns: True if parameter is custom.
            False if parameter is not custom.
        """
        custom_parameter_names = self.custom_param.keys()
        is_custom = name in custom_parameter_names
        return is_custom

    def getParameters(self, names: Union[str, Iterable[str]] = None) -> Union[Parameter, Iterable[Parameter]]:
        """
        Get parameter object from filestem and name.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve function from
        :param names: name(s) of parameter(s) to retrieve.
            Defaults to all loaded functions.
        """
        def get(name: str) -> Parameter:
            filestem = self.getChosenParameterStem(name)
            parameter = self.stem2name2param[filestem][name]
            return parameter

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def getParameterNames(self, parameter_types: Union[str, Iterable[str]] = None, custom_type: bool = None) -> List[str]:
        """
        Get name(s) of parameter(s) in model.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve names from
        :param parameter_types: type(s) of parameters to retrieve, acts as optional filter
        :param custom_type: set True to only retrieve custom parameters.
            Set False to only retrieve noncustom parameters.
            Defaults to all parameters, acts as optional filter.
        """
        if isinstance(parameter_types, str):
            parameter_names = self.getParameterNames()
            filtered_names = [
                parameter_name
                for parameter_name in parameter_names
                if self.getChosenParameterType(parameter_name) == parameter_types
            ]
            if custom_type is not None:
                filtered_names = [
                    parameter_name
                    for parameter_name in filtered_names
                    if self.isCustomParameter(parameter_name) == custom_type
                ]
            return filtered_names
        elif isinstance(parameter_types, list):
            filtered_names = []
            for parameter_type in parameter_types:
                filtered_names.extend(self.getParameterNames(parameter_types=parameter_type))
            return filtered_names
        elif parameter_types is None:
            stem2name2quant = self.stem2name2param
            parameter_names = [
                parameter_name
                for name2quant in stem2name2quant.values()
                for parameter_name in name2quant.keys()
            ]
            return unique(parameter_names)
        else:
            raise RecursiveTypeError(parameter_types)

    def getParameterQuantities(
            self, names: Union[str, Iterable[str]] = None, form: str = "quantity"
    ) -> Optional[Union[Quantity, Dict[str, Quantity]]]:
        """
        Get parameter(s), including value and unit

        __Recursion Base__
            return quantity, value, or unit of single parameter: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retrieve.
            Defaults to all loaded parameters.
        :param form: form of quantity to retrieve. Each desired option must be a substring of this argument.
            "quantity": retrieve value and unit
            "unit": retrieve only unit
            "value": retrieve only value
            "base" prefix or suffix: convert to base units (SI)

        :returns: Quantity for single parameter if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is str
            List of quantities if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is list.
            None if parameter is not included in window.
        """

        def get(name: str) -> Parameter:
            if self.isCustomParameter(name):
                quantity = self.getCustomParameterQuantities(names=name)
            else:
                filestem = self.getChosenParameterStem(name)
                quantity = self.getParameterFromStem(name, filestem).getQuantity()

            if "base" in form:
                quantity = quantity.to_base_units()
            if "quantity" in form:
                return quantity
            elif "value" in form:
                return quantity.magnitude
            elif "unit" in form:
                return quantity.units
            raise ValueError("Invalid entry for form input")

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": dict,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def setParameter(self, name: str, quantity: Quantity, custom: bool) -> None:
        """
        Set or overwrite custom parameter quantity stored in window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store quantity in
        :param name: name of parameter to set/overwrite
        :param quantity: quantity to set parameter as
        :param custom: set True to make parameter custom.
            Set False to make parameter noncustom.
        """
        old_quantity = self.getParameterQuantities(names=name)
        if isinstance(quantity, type(old_quantity)):
            self.custom_param[name] = quantity
            self.setParameterAsCustom(name, custom)
            self.updateParameterLabels(name)
        else:
            raise TypeError("new quantity must be same class as old quantity")

    def getInputParameterValue(self, names: Union[str, Iterable[str]] = None) -> Union[str, Iterable[str]]:
        """
        Get values for parameter-value input-fields.
        Uses present state of window.

        __Recursion Base__
            get field value for single parameter: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve values from
        :param names: name(s) of parameter(s) to retrieve values for.
            Defaults to all parameters.
        """
        def get(name: str) -> str:
            key = self.getKey("parameter_field", name)
            field_value = self.getValue(key)
            return field_value

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def setParameterAsCustom(self, name: str, custom: bool) -> None:
        """
        Visually display whether or not parameter is custom.

        :param self: `~Layout.MainWindow.MainWindowRunner` to display in
        :param name: name of parameter to display for
        :param custom: set True to display parameter as custom.
            Set False to display parameter as noncustom.
        """
        checkbox_key = self.getKey("custom_parameter", name)
        checkbox = self.getElements(checkbox_key)
        checkbox.update(value=custom)
        self.getWindow().write_event_value(checkbox_key, custom)

    def updateParametersFromStems(self, names: Union[str, Iterable[str]] = None):
        """
        Update parameter quantity(s) stored in window from chosen filestems.
        Uses present state of window.

        __Recursion Base__
            update quantity associated with single parameter: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to update parameters in
        :param names: name(s) of parameter(s) to update.
            Defaults to all loaded parameters.
        """

        def update(name: str) -> None:
            filestem = self.getChosenParameterStem(name)
            quantity = self.getParameterFromStem(name, filestem).getQuantity()
            self.setParameter(name=name, quantity=quantity, custom=False)

        kwargs = {
            "base_method": update,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def updateParametersFromFields(self, names: Union[str, Iterable[str]] = None) -> None:
        """
        Update parameter quantity(s) stored in window from input fields.
        Uses present state of window.

        __Recursion Base__
            update quantity associated with single parameter: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to update parameters in
        :param names: name(s) of parameter(s) to update.
            Defaults to all loaded parameters.
        """

        def update(name: str) -> None:
            field_value = self.getInputParameterValue(names=name)
            if field_value != '':
                old_unit, new_value = self.getParameterQuantities(names=name, form="unit"), float(field_value)
                new_quantity = new_value * old_unit
                self.setParameter(name=name, quantity=new_quantity, custom=True)

        kwargs = {
            "base_method": update,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def updateParameterLabels(self, names: Union[str, Iterable[str]] = None) -> None:
        """
        Update quantity (value and unit) label for parameter.

        __Recursion Base__
            update label for single parameter: names [str]

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to update label in
        :param names: name(s) of parameter(s) to update labels for.
            Defaults to all parameters.
        """

        def update(name: str) -> None:
            label_key = self.getKey("quantity_label", name)
            label_element = self.getElements(label_key)
            new_label = formatQuantity(self.getParameterQuantities(names=name))
            label_element.update(new_label)

        kwargs = {
            "base_method": update,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

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
            for name, quantity in readParameters(filename).items()
        }

        if choose_parameters:
            runner = ChooseParametersWindowRunner("Choose Parameters to Load", quantities=load_quantities)
            parameter_names = runner.getChosenParameters()
        else:
            parameter_names = self.getParameterNames()

        for name, quantity in load_quantities.items():
            if name in parameter_names:
                self.setParameter(name=name, quantity=quantity, custom=True)

    def getFreeParameterValues(self) -> Tuple[str, Dict[str, Tuple[float, float, int, Quantity]]]:
        """
        Open window allowing user to set minimum, maximum, and step count for each selected free parameter.
        Uses present state of window.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve free parameters from
        """
        free_parameter_names = self.getParameterNames(parameter_types="Free")
        free_parameter_quantities = self.getParameterQuantities(names=free_parameter_names)

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
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner`
            to call :class:`~Layout.SimulationWindow.SimulationWindowRunner`
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
