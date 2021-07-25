from __future__ import annotations

from functools import partial
from os.path import dirname, join
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
import yaml
from colour import Color
from igraph import Graph, plot
from numpy import ndarray
from pint import Quantity

from CustomErrors import RecursiveTypeError
from Function import Derivative, Function, Independent, Model, generateParameter, readFunctionsFromFiles, readParametersFromFiles
from Function import Parameter
from Layout.ChooseGraphLayoutWindow import ChooseGraphLayoutWindowRunner
from Layout.ChooseParametersWindow import ChooseParametersWindowRunner
from Layout.Layout import Element, Layout, Row, Tab, TabGroup, TabRow, TabbedWindow, WindowRunner, generateCollapsableSection, getKeys, getNameFromElementKey, storeElement
from Layout.SetFreeParametersWindow import SetFreeParametersWindowRunner
from Layout.SimulationWindow import SimulationWindowRunner
from YML import getDimensions, getStates, readLayout, readStates
from macros import StoredObject, expression2png, formatQuantity, getTexImage, recursiveMethod, unique

tet_types = ("Temporal", "Equilibrium", "Constant", "Function")
p_types = ("Constant", "Free")

psl_pre = "PARAMETER SECTION LABEL"
psc_pre = "PARAMETER SECTION COLLAPSABLE"
tet_pre = "TIME_EVOLUTION TYPE"
ice_pre = "INITIAL CONDITION EQUILIBRIUM"
ff_pre = "FUNCTION FILESTEM"
pf_pre = "PARAMETER FILESTEM"


class TimeEvolutionRow(TabRow, StoredObject):
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
        Constructor for :class:`~Layout.MainWindow.TimeEvolutionRow`.

        :param name: name of variable
        :param tab: tab that row is stored in
        """
        TabRow.__init__(self, name, tab)
        StoredObject.__init__(self, name)

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
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve initial condition from
        """
        return self.getTab().getInitialConditions(self.getName())

    def getTimeEvolutionTypes(self, **kwargs) -> Union[str, Iterable[str]]:
        """
        Get possible time-evolution types for variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve time-evolution types from
        :param kwargs: additional arguments to pass into
            :meth:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes`
        """
        tab: TimeEvolutionTab = self.getTab()
        return tab.getTimeEvolutionTypes(self.getName(), **kwargs)

    def getRowLabel(self) -> Union[sg.Text, sg.Image]:
        """
        Get element to label time-evolution row by variable.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve label from
        """
        kwargs = {
            "name": self.getName(),
            "size": self.getDimensions(name="variable_label")
        }
        return getTexImage(**kwargs)

    @storeElement
    def getTimeEvolutionTypeElement(self) -> sg.InputCombo:
        """
        Get element allowing user to set time-evolution type.

        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve element from
        """
        kwargs = {
            "values": self.getTimeEvolutionTypes(),
            "default_value": self.getTimeEvolutionTypes(index=0),
            "enable_events": True,
            "size": self.getDimensions(name="evolution_type_combobox"),
            "key": f"-{tet_pre:s} {self.getName():s}-"
        }
        return sg.InputCombo(**kwargs)

    @storeElement
    def getInitialConditionElement(self) -> sg.InputText:
        """
        Get element allowing user to input initial value.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve element from
        """
        kwargs = {
            "default_text": self.getInitialCondition(),
            "size": self.getDimensions(name="initial_condition_input_field"),
            "key": f"-INITIAL CONDITION VALUE {self.getName():s}-"
        }
        return sg.InputText(**kwargs)

    @storeElement
    def getInitialEquilibriumElement(self) -> sg.Checkbox:
        """
        Get element allowing user to choose whether or not variable begins simulation in equilibrium.
        
        :param self: :class:`~Layout.MainWindow.TimeEvolutionRow` to retrieve element from
        """
        kwargs = {
            "text": "Equilibrium",
            "default": False,
            "enable_events": True,
            "disabled": True,
            "size": self.getDimensions(name="initial_equilibrium_checkbox"),
            "key": f"-{ice_pre:s} {self.getName():s}"
        }
        return sg.Checkbox(**kwargs)


class TimeEvolutionTab(Tab):
    """
    Tab to organize time-evolution rows for variables.
    
    This contains
        #. Header of labels to indicate purpose of each element in variable row
        #. :class:`~Layout.MainWindow.TimeEvolutionRow` for each variable in tab
    
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

        self.variable_rows = [
            TimeEvolutionRow(variable_name, self)
            for variable_name in variable_names
        ]

    def getVariableRows(self) -> List[TimeEvolutionRow]:
        """
        Get variable rows in tab.

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
            """Base method for :meth:`~Layout.MainWindow.TimeEvolutionTab.getInitialConditions`."""
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
        :returns: Time-evolution type at index
            if :paramref:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes.index` is int.
            All time-evolution types for variable
            if :paramref:`~Layout.MainWindow.TimeEvolutionTab.getTimeEvolutionTypes.index` is None.
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
        rows = list(map(TimeEvolutionRow.getRow, self.getVariableRows()))
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


class ParameterRow(TabRow, StoredObject):
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
            parameter_types: Tuple[str] = p_types
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
        TabRow.__init__(self, name, section.getTab())
        StoredObject.__init__(self, name)
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
        filestems = list(map(Parameter.getStem, self.getParameters()))
        return filestems

    def getQuantities(self) -> List[Quantity]:
        """
        Get filepaths for files containing parameter.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve filepaths from
        """
        quantities = list(map(Parameter.getQuantity, self.getParameters()))
        return quantities

    def getParameters(self) -> List[Parameter]:
        """
        Get parameters for row.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve parameters from
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
        :returns: Parameter type at index
            if :paramref:`~Layout.MainWindow.ParameterRow.getParameterTypes.index` is int.
            All parameter types for parameter
            if :paramref:`~Layout.MainWindow.ParameterRow.getParameterTypes.index` is None.
        """
        if isinstance(index, int):
            parameter_types = self.getParameterTypes()
            return parameter_types[index]
        elif index is None:
            return self.parameter_types
        else:
            raise TypeError("index must be int")

    @storeElement
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
            "key": f"-CUSTOM PARAMETER {name:s}-"
        }
        return sg.Checkbox(**kwargs)

    @storeElement
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
            "key": f"-{pf_pre:s} {self.getName():s}-"
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

    @storeElement
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
            "key": f"-PARAMETER VALUE LABEL {self.getName():s}-"
        }
        return sg.Text(**kwargs)

    @storeElement
    def getValueInputElement(self) -> sg.InputText:
        """
        Get field to allow user to set parameter value.

        :param self: :class:`~Layout.MainWindow.ParameterRow` to retrieve element from
        """
        kwargs = {
            "default_text": '',
            "tooltip": "If 'Constant': enter new parameter value, then click button to update. " +
                       "If 'Sweep': this input field will be ignored." +
                       "Displayed units are preserved",
            "size": self.getDimensions(name="parameter_value_input_field"),
            "key": f"-PARAMETER VALUE INPUT {self.getName():s}-"
        }
        return sg.InputText(**kwargs)

    @storeElement
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
            "key": f"-PARAMETER TYPE {self.getName():s}-"
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
            "key": f"-{psl_pre:s} {name:s}-"
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
            "layout": list(map(ParameterRow.getRow, self.getParameterRows())),
            "size": self.getDimensions(name="parameter_section"),
            "key": f"-{psc_pre:s} {self.getName():s}-"
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
        section_layouts = map(ParameterSection.getLayout, self.getSections())
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


class FunctionRow(TabRow, StoredObject):
    """
    Row to display function info.

    This contains
        #. Label for name of function
        #. Label for expression of function
        #. Combobox to choose filestem to load function from

    :ivar functions: :class:`~Function.Function`s displayed by row
    """

    image_folder = "tex_eq"

    def __init__(self, name: str, tab: FunctionTab, functions: List[Function]) -> None:
        """
        Constructor for :class:`~Layout.MainWindow.FunctionRow`.

        :param name: name of function
        :param tab: tab that row is stored in
        :param functions: dictionary from of functions.
            Key is filestem for file.
            Value is function object loaded from file.
        """
        TabRow.__init__(self, name, tab)
        StoredObject.__init__(self, name)

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

    @staticmethod
    def getImageFoldername(filestem: str) -> str:
        """
        Get folderpath to save or load image from.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve folderpath from
        :param filestem: stem of file to retrieve folderpath for
        """
        top_folder = FunctionRow.image_folder
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
        filestems = list(map(Function.getStem, self.getFunctions()))
        return filestems

    def getFunctions(self) -> List[Function]:
        """
        Get :class:`~Function.Function` in order of filestems.

        :param self: :class:`~Layout.MainWindow.FunctionRow` to retrieve filepaths from
        """
        return self.functions

    @storeElement
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
            "key": f"-{ff_pre:s} {self.getName():s}-"
        }
        return sg.Combo(**kwargs)

    @storeElement
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
            "key": f"-FUNCTION EXPRESSION {self.getName():s}-"
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
        rows = list(map(FunctionRow.getRow, self.getFunctionRows()))
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
        #. :class:`~Layout.MainWindow.TimeEvolutionTabGroup` to allow user to set time-evolution for each variable
        #. :class:`~Layout.MainWindow.ParameterTabGroup` to allow user to set properties for each parameter

    :ivar stem2name2param: 2-level dictionary.
        First key is filestems for parameter files.
        Second key is name of parameter.
        Value is :class:`~Function.Parameter`.
    :ivar stem2name2func: 2-level dictionary.
        First key is filestem for function files.
        Second key is name of function.
        Value is :class:`~Function.Function`.
    :ivar variable_names: names of variable included in window
    :ivar parameter_names: names of parameters included in window
    :ivar function_names: names of function icnluded in window
    :ivar blueprint: dictionary indiciated how to set up elements in window
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

        self.stem2name2param = stem2name2obj["parameters"]
        self.stem2name2func = stem2name2obj["functions"]

        self.variable_names = []
        self.parameter_names = []
        self.function_names = []
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
            """Base method for :meth:`~Layout.MainWindow.MainWindow.addVariableNames`."""
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
            """Base method for :meth:`~Layout.MainWindow.MainWindow.addFunctionNames`."""
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
            """Base method for :meth:`~Layout.MainWindow.MainWindow.addParameterNames`."""
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

    @storeElement
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
                    "Functions::import",
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
            "key": "-TOOLBAR MENU-"
        }
        return sg.Menu(**kwargs)

    @storeElement
    def getOpenSimulationButton(self) -> sg.Button:
        """
        Get button to open :class:`~Layout.SimulationWindow.SimulationWindowRunner`.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve button from
        """
        text = "Open Simulation"
        kwargs = {
            "button_text": text,
            "key": f"-{text.upper():s}-"
        }
        return sg.Button(**kwargs)

    @storeElement
    def getGenerateGraphButton(self) -> sg.Button:
        """
        Get button to generate function-to-argument directional graph.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve button from
        """
        text = "Generate Graph"
        kwargs = {
            "button_text": text,
            "key": f"-{text.upper():s}-"
        }
        return sg.Button(**kwargs)

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for window.

        :param self: :class:`~Layout.MainWindow.MainWindow` to retrieve layout from
        """
        menu = self.getMenu()
        open_simulation_button = self.getOpenSimulationButton()
        generate_graph_button = self.getGenerateGraphButton()
        prefix_layout = Layout(rows=Row(window=self, elements=menu))
        suffix_layout = Layout(rows=Row(window=self, elements=[open_simulation_button, generate_graph_button]))
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

    :ivar custom_parameters: dictionary of custom parameters.
        Key is name for parameter.
        Value is custom-set value and unit for parameter.
    :ivar stem2path_param: dictionary from filestems to filepaths for files with parameters.
    :ivar stem2path_func: dictionary from filestems to filepaths for files with functions.
    :ivar stem2name2param: 2-level dictionary.
        First key is filestems for parameter files.
        Second key is name of parameter.
        Value is :class:`~Function.Parameter`.
    :ivar stem2name2func: 2-level dictionary.
        First key is filestem for function files.
        Second key is name of function.
        Value is :class:`~Function.Function`.
    :ivar getVariableNames: pointer to :meth:`~Layout.MainWindow.MainWindow.getVariableNames`
    :ivar getFunctionNames: pointer to :meth:`~Layout.MainWindow.MainWindow.getFunctionNames`
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
        self.custom_parameters = {}
        self.stem2path_param = {}
        self.stem2name2param = {}
        for filepath in parameter_filepaths:
            filestem = Path(filepath).stem
            self.stem2path_param[filestem] = filepath
            self.stem2name2param[filestem] = readParametersFromFiles(filepath)

        self.stem2path_func = {}
        self.stem2name2func = {}
        for filepath in function_filepaths:
            filestem = Path(filepath).stem
            self.stem2path_func[filestem] = filepath
            self.stem2name2func[filestem] = readFunctionsFromFiles(filepath)

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

    def runWindow(self) -> None:
        # noinspection PyTypeChecker
        window_object: MainWindow = self.getWindowObject()
        window = window_object.getWindow()

        toolbar_menu_key = getKeys(window_object.getMenu())
        open_simulation_key = getKeys(window_object.getOpenSimulationButton())
        generate_graph_key = getKeys(window_object.getGenerateGraphButton())

        while True:
            event, self.values = window.read()
            print(event)
            if event in [sg.WIN_CLOSED, event == 'Exit']:
                break
            menu_value = self.getValue(toolbar_menu_key)

            if menu_value is not None:
                if event == "Parameters::import":
                    self.loadParametersFromFile()
                elif event == "Functions::import":
                    self.loadFunctionsFromFile()
                elif "::set_time_evolution_types_to" in event:
                    time_evolution_type = event.replace("::set_time_evolution_types_to", '')
                    comboboxes = map(
                        TimeEvolutionRow.getTimeEvolutionTypeElement,
                        TimeEvolutionRow.getInstances()
                    )
                    self.setElementsAsValue(comboboxes, time_evolution_type)
                elif "::set_parameter_types_to" in event:
                    parameter_type = event.replace("::set_parameter_types_to", '')
                    comboboxes = map(
                        ParameterRow.getParameterTypeElement,
                        ParameterRow.getInstances()
                    )
                    self.setElementsAsValue(comboboxes, parameter_type)
                elif "::set_parameter_filestems_to" in event:
                    parameter_filestem = event.replace("::set_parameter_filestems_to", '')
                    comboboxes = map(
                        ParameterRow.getChooseFileElement,
                        ParameterRow.getInstances()
                    )
                    self.setElementsAsValue(comboboxes, parameter_filestem)
                elif "::set_function_filestems_to" in event:
                    function_filestem = event.replace("::set_function_filestems_to", '')
                    comboboxes = map(
                        FunctionRow.getChooseFileElement,
                        FunctionRow.getInstances()
                    )
                    self.setElementsAsValue(comboboxes, function_filestem)
            elif psl_pre in event:
                key = event.replace(psl_pre, psc_pre)
                self.toggleVisibility(key)
            elif tet_pre in event or ice_pre in event:
                variable_name = getNameFromElementKey(event)
                self.changeTimeEvolution(variable_name)
            elif ff_pre in event:
                function_name = getNameFromElementKey(event)
                self.updateFunctionExpressions(names=function_name)
            elif pf_pre in event:
                parameter_name = getNameFromElementKey(event)
                self.updateParametersFromStems(names=parameter_name)
            elif event == "Update Parameters":
                self.updateParametersFromFields()
            elif event == open_simulation_key:
                self.openSimulationWindow()
            elif event == generate_graph_key:
                self.openFunction2ArgumentGraph()
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
            "Function": list(map(Function.getName, model.getFunctions(filter_type=Independent))),
            "Parameter": self.getParameterNames(parameter_types="Free")
        }
        return plot_choices

    def getModel(self) -> Model:
        """
        Get model for window.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve model from
        """
        # noinspection PyTypeChecker
        function_objects: List[Function] = self.getFunctions()
        # noinspection PyTypeChecker
        parameters: List[Parameter] = self.getParameters()
        full_model = Model(functions=function_objects, parameters=parameters)

        # noinspection PyTypeChecker
        core_function_objects: List[Function] = full_model.getDerivatives()
        pre_length = -1
        while pre_length != len(core_function_objects):
            pre_length = len(core_function_objects)
            children_function_objects = unique(
                [
                    child_function_object
                    for function_object in core_function_objects
                    for child_function_object in function_object.getChildren()
                    if child_function_object not in core_function_objects
                ]
            )
            core_function_objects.extend(children_function_objects)

        getParameters = partial(Function.getFreeSymbols, generations="all", species="Parameter", return_type=str)
        core_parameter_names = unique(
            [
                parameter_name
                for function_object in core_function_objects
                for parameter_name in getParameters(function_object)
            ]
        )
        core_parameter_objects = full_model.getParameters(names=core_parameter_names)

        core_model = Model(functions=core_function_objects, parameters=core_parameter_objects)
        for derivative in core_model.getDerivatives():
            variable_name = derivative.getVariable(return_type=str)
            derivative.setTimeEvolutionType(self.getTimeEvolutionTypes(names=variable_name))
            derivative.setInitialCondition(self.getInitialConditions(names=variable_name))
        return core_model

    def setElementsAsValue(self, elements: Union[sg.Element, Iterable[sg.Element]], value: str) -> None:
        """
        Set values for collection of elements.

        :param elements: element(s) to set value for
        :param value: value to set in element(s)
        """
        window = self.getWindow()

        def set(element: sg.Element) -> None:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.setElementAsValue`"""
            if value in vars(element)["Values"]:
                element.update(value)
                key = getKeys(element)
                window.write_event_value(key, value)

        kwargs = {
            "args": elements,
            "base_method": set,
            "valid_input_types": sg.Element,
        }
        return recursiveMethod(**kwargs)

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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getInitialConditions`."""
            time_evolution_type_row = TimeEvolutionRow.getInstances(names=name)

            time_evolution_type = self.getValue(
                getKeys(time_evolution_type_row.getTimeEvolutionTypeElement())
            )
            is_equilibrium = time_evolution_type == "Equilibrium"
            is_initial_equilibrium_checked = self.getValue(
                getKeys(time_evolution_type_row.getInitialEquilibriumElement())
            )
            is_initial_equilibrium = is_initial_equilibrium_checked or is_equilibrium

            if is_initial_equilibrium:
                return "Equilibrium"
            else:
                input_field_key = getKeys(time_evolution_type_row.getInitialConditionElement())
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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getTimeEvolutionTypes`."""
            key = getKeys(TimeEvolutionRow.getInstances(names=name).getTimeEvolutionTypeElement())
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

    def changeTimeEvolution(self, name: str) -> None:
        """
        Disable input field for initial conditions
            if time-evolution type is set to "Equilibrium" or if initial condition is set to equilibrium.
        Disable checkbox if time-evolution type is set to "Equilibrium".
        Enable elements otherwise.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve elements from
        :param name: name of variable associated with elements
        """
        time_evolution_row = TimeEvolutionRow.getInstances(names=name)

        time_evolution_type = self.getValue(getKeys(time_evolution_row.getTimeEvolutionTypeElement()))
        is_equilibrium = time_evolution_type == "Equilibrium"

        checkbox = time_evolution_row.getInitialEquilibriumElement()
        is_initial_equilibrium = self.getValue(getKeys(checkbox))

        is_either_equilibrium = is_equilibrium or is_initial_equilibrium
        time_evolution_row.getInitialConditionElement().update(disabled=is_either_equilibrium)
        checkbox.update(disabled=is_equilibrium)

    def getPathsFromFunctionStems(self, filestems: Union[str, Iterable[str]] = None) -> Union[str, List[str]]:
        """
        Get filepaths for original loaded functions.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve path from
        :param filestems: stem(s) of file(s) to retrieve path(s) for
        """

        def get(filestem: str) -> str:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getPathsFromFunctionStems`"""
            return self.stem2path_func[filestem]

        kwargs = {
            "base_method": get,
            "args": filestems,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getFunctionStems()
        }
        return recursiveMethod(**kwargs)

    def getFunctionStems(self) -> List[str]:
        """
        Get filestems for original loaded functions.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve stems from
        """
        return list(self.stem2path_func.keys())

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

        def get(name: str) -> Function:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getFunctions`."""
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
        combobox_key = getKeys(FunctionRow.getInstances(names=name).getChooseFileElement())
        filestem = self.getValue(combobox_key)
        return filestem

    def setFunctions(self, function_objects: Union[Function, Iterable[Function]]) -> None:
        """
        Set or overwrite parameter stored in window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store parameter in
        :param function_objects: parameter(s) to set/overwrite.
            Overwrites based on parameter name stored in parameter object.
        """

        def set(function_object: Function) -> None:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.setFunctions`"""
            filestem = function_object.getStem()
            if isinstance(filestem, str):
                self.setFunctionWithStem(function_object)
            else:
                self.setFunctionWithoutStem(function_object)

        kwargs = {
            "base_method": set,
            "args": function_objects,
            "valid_input_types": Function,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

    def setFunctionWithStem(self, function_object: Function) -> None:
        """
        Set function from filestem.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filestem info from
        :param function_object: function to set/overwrite
        """
        name = function_object.getName()
        new_filestem = function_object.getStem()

        combobox_filestem = FunctionRow.getInstances(names=name).getChooseFileElement()
        combobox_key = getKeys(combobox_filestem)
        old_filestem = self.getValue(combobox_key)
        combobox_stems = vars(combobox_filestem)["Values"]

        if new_filestem in combobox_stems:
            if old_filestem != new_filestem:
                combobox_filestem.update(new_filestem)
                self.getWindow().write_event_value(combobox_key, new_filestem)
            self.updateFunctionExpressions(name)
        else:
            sg.PopupError(f"Filestem {new_filestem:s} not found for function {name:s}")

    def updateFunctionExpressions(self, names: Union[str, Iterable[str]] = None) -> None:
        """
        Update function expression from selected file.

        :param self: :class`~Layout.MainWindow.MainWindowRunner` to update expression in
        :param names: name(s) of function(s) to update expression for.
            Defaults to all functions.
        """

        def update(name: str) -> None:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.updateFunctionExpressions`."""
            function_filestem = self.getChosenFunctionStem(name)
            image_folder = join("tex_eq", function_filestem)
            image_filename = '.'.join((name, "png"))
            image_filepath = join(image_folder, image_filename)
            image_data = open(image_filepath, 'rb').read()
            image_expression = FunctionRow.getInstances(names=name).getExpressionLabel()
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

    def getPathsFromParameterStems(self, filestems: Union[str, Iterable[str]] = None) -> Union[str, List[str]]:
        """
        Get filepaths for original loaded parameters.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve path from
        :param filestems: stem(s) of file(s) to retrieve path(s) for
        """

        def get(filestem: str) -> str:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getPathsFromParameterStems`"""
            return self.stem2path_param[filestem]

        kwargs = {
            "base_method": get,
            "args": filestems,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterStems()
        }
        return recursiveMethod(**kwargs)

    def getParameterStems(self) -> List[str]:
        """
        Get filestems for original loaded parameters.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve stems from
        """
        return list(self.stem2path_param.keys())

    def getChosenParameterStem(self, name: str) -> str:
        """
        Get stem of file to load parameter from.
        Uses present state of window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filestem from
        :param name: name of parameter to retrieve filestem for
        """
        combobox_key = getKeys(ParameterRow.getInstances(names=name).getChooseFileElement())
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
        combobox_key = getKeys(ParameterRow.getInstances(names=name).getParameterTypeElement())
        parameter_type = self.getValue(combobox_key)
        return parameter_type

    def getCustomParameterNames(self) -> List[str]:
        """
        Get names of custom parameters stored in window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve names from
        """
        return list(self.custom_parameters.keys())

    def getCustomParameterQuantities(
            self, names: Union[str, Iterable[str]] = None
    ) -> Union[Quantity, Dict[str, Quantity]]:
        """
        Get custom value for parameter.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve value from
        :param names: name(s) of parameter(s) to retrieve quantity(s) for
        """

        def get(name: str) -> str:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getCustomParameterQuantities`."""
            return self.custom_parameters[name].getQuantity()

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": dict,
            "default_args": list(self.custom_parameters.keys())
        }
        return recursiveMethod(**kwargs)

    def isCustomParameter(self, name: str) -> bool:
        """
        Determine whether or not parameter should use custom value.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve parameter property from
        :param name: name of parameter
        :returns: True if parameter is custom.
            False if parameter is not custom.
        """
        custom_parameter_names = self.getCustomParameterNames()
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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getParameters`."""
            if self.isCustomParameter(name):
                quantity = self.getParameterQuantities(names=name)
                parameter = Parameter(name, quantity)
            else:
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

    def getParameterNames(
            self, parameter_types: Union[str, Iterable[str]] = None, custom_type: bool = None
    ) -> List[str]:
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
        :returns: Quantity for single parameter
            if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is str
            List of quantities
            if :paramref:`~Layout.MainWindow.MainWindowRunner.getQuantities.names` is list.
            None
            if parameter is not included in window.
        """

        def get(name: str) -> Parameter:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getParameterQuantities`."""
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

    def setParameters(self, parameters: Union[Parameter, Iterable[Parameter]]) -> None:
        """
        Set or overwrite parameter stored in window.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store parameter in
        :param parameters: parameter(s) to set/overwrite.
            Overwrites based on parameter name stored in parameter object.
        """

        def set(parameter: Parameter) -> None:
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.setParameters`"""
            filestem = parameter.getStem()
            if isinstance(filestem, str):
                self.setParameterWithStem(parameter)
            else:
                self.setParameterWithoutStem(parameter)

        kwargs = {
            "base_method": set,
            "args": parameters,
            "valid_input_types": Parameter,
            "output_type": list
        }
        return recursiveMethod(**kwargs)

    def setParameterAsCustom(self, name: str, custom: bool) -> None:
        """
        Set whether or not parameter should be treat as a custom parameter.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store parameter in
        :param name: name of parameter
        :param custom: set True to treat parameter as custom.
            Set False otherwise.
        """
        checkbox = ParameterRow.getInstances(names=name).getCustomCheckbox()
        checkbox_key = getKeys(checkbox)
        checkbox.update(custom)
        self.getWindow().write_event_value(checkbox_key, custom)

    def setParameterWithoutStem(self, parameter: Parameter) -> None:
        """
        Set parameter without associated filestem.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store parameter in
        :param parameter: parameter to set/overwrite
        """
        name = parameter.getName()
        self.custom_parameters[name] = parameter
        self.setParameterAsCustom(name, True)
        self.updateParameterLabels(name)

    def setParameterWithStem(self, parameter: Parameter) -> None:
        """
        Set parameter from filestem.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve filestem info from
        :param parameter: parameter to set/overwrite
        """
        name = parameter.getName()

        if self.isCustomParameter(name):
            del self.custom_parameters[name]
            self.setParameterAsCustom(name, False)

        new_filestem = parameter.getStem()

        combobox_filestem = ParameterRow.getInstances(names=name).getChooseFileElement()
        combobox_key = getKeys(combobox_filestem)
        old_filestem = self.getValue(combobox_key)
        combobox_stems = vars(combobox_filestem)["Values"]
        if new_filestem in combobox_stems:
            if old_filestem != new_filestem:
                combobox_filestem.update(new_filestem)
                self.getWindow().write_event_value(combobox_key, new_filestem)
            self.updateParameterLabels(name)
        else:
            sg.PopupError(f"Filestem {new_filestem:s} not found for parameter {name:s}")

    def getInputParameterValues(self, names: Union[str, Iterable[str]] = None) -> Union[str, Iterable[str]]:
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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.getInputParameterValues`."""
            input_field_key = getKeys(ParameterRow.getInstances(names=name).getValueInputElement())
            field_value = self.getValue(input_field_key)
            return field_value

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.updateParameterFromStems`."""
            filestem = self.getChosenParameterStem(name)
            parameter = self.getParameterFromStem(name, filestem)
            self.setParameterWithStem(parameter)

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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.updateParametersFromFields`."""
            field_value = self.getInputParameterValues(names=name)
            if field_value != '':
                old_unit, new_value = self.getParameterQuantities(names=name, form="unit"), float(field_value)
                new_quantity = new_value * old_unit
                new_parameter = Parameter(name, new_quantity)
                self.setParameters(new_parameter)

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
            """Base method for :meth:`~Layout.MainWindow.MainWindowRunner.updateParameterLabels`."""
            label_element = ParameterRow.getInstances(names=name).getQuantityLabel()
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

    def loadParametersFromFile(self, filepath: str = None, choose_parameters: bool = True) -> Optional[List[Parameter]]:
        """
        Load and store parameter quantities from file.
        
        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store quantities in
        :param filepath: path of file to load parameters from.
            Defaults to letting user choose file.
        :param choose_parameters: set True to allow user to choose which parameters to actually load.
            Set False to automatically load all parameters from file.
        """
        if filepath is None:
            file_types = (("YML", "*.yml"), ("YAML", "*.yaml"), ("Plain Text", "*.txt"), ("ALL Files", "*.*"),)
            kwargs = {
                "message": "Enter Filename to Load",
                "title": "Load Parameters",
                "file_types": file_types,
                "multiple_files": False
            }
            filepath = sg.PopupGetFile(**kwargs)
            if filepath is None:
                return None

        info = yaml.load(open(filepath, 'r'), Loader=yaml.Loader)
        filestems = self.getParameterStems()
        loaded_parameters = []
        for key, value in info.items():
            if key in filestems:
                path_from_stem = self.getPathsFromParameterStems(key)
                parameters_from_file = readParametersFromFiles(path_from_stem, names=value).values()
                loaded_parameters.extend(parameters_from_file)
            else:
                loaded_parameters.append(generateParameter(key, value))

        if choose_parameters:
            runner = ChooseParametersWindowRunner("Choose Parameters to Load", parameters=loaded_parameters)
            parameter_names = runner.getChosenParameters()
        else:
            parameter_names = self.getParameterNames()

        chosen_parameters = [
            parameter
            for parameter in loaded_parameters
            if parameter.getName() in parameter_names
        ]

        for parameter in chosen_parameters:
            self.setParameters(parameter)

    def loadFunctionsFromFile(self, filepath: str = None, choose_functions: bool = False) -> Optional[List[Function]]:
        """
        Load and store function objects from file.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to store objects in
        :param filepath: path of file to load functions from.
            Defaults to letting user choose file.
        :param choose_functions: set True to allow user to choose which functions to actually load.
            Set False to automatically load all functions from file.
        """
        if filepath is None:
            file_types = (("YML", "*.yml"), ("YAML", "*.yaml"), ("Plain Text", "*.txt"), ("ALL Files", "*.*"),)
            kwargs = {
                "message": "Enter Filename to Load",
                "title": "Load Function",
                "file_types": file_types,
                "multiple_files": False
            }
            filepath = sg.PopupGetFile(**kwargs)
            if filepath is None:
                return None

        info = yaml.load(open(filepath, 'r'), Loader=yaml.Loader)
        filestems = self.getFunctionStems()
        loaded_functions = []
        for key, value in info.items():
            if key in filestems:
                path_from_stem = self.getPathsFromFunctionStems(key)
                functions_from_file = readFunctionsFromFiles(path_from_stem, names=value).values()
                loaded_functions.extend(functions_from_file)
            else:
                loaded_functions.append(generateParameter(key, value))

        if choose_functions:
            runner = ChooseFunctionsWindowRunner("Choose Functions to Load", function_objects=loaded_functions)
            function_names = runner.getChosenFunctions()
        else:
            function_names = self.getFunctionNames()

        chosen_functions = [
            function_object
            for function_object in loaded_functions
            if function_object.getName() in function_names
        ]

        for function_object in chosen_functions:
            self.setFunctions(function_object)

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

    def generateFunction2ArgumentGraph(self, color1: str = "red", color2: str = "blue") -> Graph:
        """
        Generate directional graph from (1) derivative variable to (2) variables in derivative.

        :param self: :class:`~Layout.MainWindow.MainWindowRunner` to retrieve derivatives from
        :param color1: color of first node (and edges) in graph
        :param color2: color of last node (and edges) in graph
        :returns: Generated graph
        """
        derivatives: List[Union[Derivative, Function]] = self.getModel().getDerivatives()
        der2vars = {
            der.getVariable(return_type=str):
                der.getFreeSymbols(species="Variable", generations="all", return_type=str)
            for der in derivatives
        }
        variable_names = sorted(der2vars.keys(), key=lambda k: len(der2vars[k]))

        graph = Graph(
            n=len(variable_names),
            directed=True
        )
        colors = Color(color1).range_to(Color(color2), len(der2vars.keys()))
        vertex2color = [color.rgb for color in colors]

        for ider in range(len(variable_names)):
            derivative_name = variable_names[ider]
            vertex_color = vertex2color[ider]
            graph.vs[ider]["name"] = derivative_name
            graph.vs[ider]["color"] = vertex_color
            for variable_name in der2vars[derivative_name]:
                ivar = variable_names.index(variable_name)
                graph.add_edges([(ider, ivar)])
                graph.es[-1]["color"] = vertex_color
        graph.vs["label"] = graph.vs["name"]

        return graph

    def openFunction2ArgumentGraph(self) -> None:
        """
        Open plot showing function-to-argument directional graph.

        :param self: :class:`~Layout.Layout.MainWindowRunner` to retrieve functions from
        """
        choose_graph_layout_window = ChooseGraphLayoutWindowRunner("Choose Graph Layout")
        event, layout_code = choose_graph_layout_window.getLayoutCode()
        if event == "Submit":
            graph = self.generateFunction2ArgumentGraph()
            layout = graph.layout(layout_code)
            plot(graph, layout=layout)
