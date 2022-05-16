"""
This module contains wrappers and containers for PySimpleGUI elements.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg

from KeyList import KeyList
from macros import recursiveMethod


def generateCollapsableSection(layout: List[List[sg.Element]], **kwargs) -> sg.pin:
    """
    Create collapsable section of elements.

    :param layout: layout to generate collapsable section of
    :param kwargs: arguments to pass into :class:`~PySimpleGUI.PySimpleGUI.Column`
    :returns: collapsable section of elements.
    """
    return sg.pin(sg.Column(layout, **kwargs))


def generateEmptySpace(layout: List[List[sg.Element]]) -> sg.pin:
    """
    Create empty space corresponding to layout size.

    :param layout: layout to generate empty space of
    :returns: collapsable section of elements.
    """
    return sg.pin(sg.Column(layout, visible=False, expand_x=True), expand_x=True)


def rowsToColumnsLayout(row_objs: Iterable[Row]):
    """
    Align rows by column.
    E.g. first items of each row are aligned in same column.

    :param row_objs: rows to align by column. Each row must have equal number of elements.
    """
    rows = list(map(Row.getElements, row_objs))
    row_count = len(rows)

    if row_count == 0:
        layout = [[]]
    elif row_count >= 1:
        row_length = len(rows[0])
        for row in rows:
            print("temp:", len(row), row_length, getKeys(row))
            assert len(row) == row_length

        columns = [[] for _ in range(row_length)]

        for row in rows:
            for index, element in enumerate(row):
                columns[index].append([element])

        layout = [
            list(map(sg.Column, columns))
        ]
    return layout


def storeElement(instantiateElement):
    def wrapper(self, *args, **kwargs):
        method_name = instantiateElement.__name__
        try:
            self.stored_elements
        except AttributeError:
            self.stored_elements = {}

        try:
            element = self.stored_elements[method_name][(*args,)]
        except KeyError:
            element = instantiateElement(self, *args, **kwargs)

            if method_name not in self.stored_elements.keys():
                self.stored_elements[method_name] = {}
            self.stored_elements[method_name][(*args,)] = element

        return element

    return wrapper


def getKeys(elements: Union[sg.Element, Iterable[sg.Element]]) -> Union[str, Tuple[str]]:
    """
    Get key associated from element.

    :param elements: element(s) to retrieve key(s) from
    """

    def get(element: sg.Element) -> str:
        """Base method for :meth:`~Layout.Layout.getKey`"""
        return vars(element)["Key"]

    return recursiveMethod(
        args=elements,
        base_method=get,
        valid_input_types=sg.Element,
        output_type=tuple
    )


def getNameFromElementKey(key: str) -> Optional[str]:
    """
    Get name of symbol associated with element.

    :param key: key for element
    """
    splits = key.split(' ')
    name = splits[-1].replace('-', '')
    return name


class Element:
    """
    Wrapper for PySimpleGUI Element.

    This connects element to
        #. :class:`~Layout.Layout.Window`
        #. :class:`~Layout.Layout.WindowRunner`

    :ivar window: :class:`~Layout.Layout.Window` that contains element
    :ivar getWindowRunner: pointer to :meth:`~Layout.Layout.Window.getWindowRunner`
    :ivar getDimensions: pointer to :meth:`~Layout.Layout.Window.getDimensions`
    :ivar getKey: pointer to :meth:`~Layout.Layout.Window.getKey`
    """

    def __init__(
        self,
        window: Window,
        name: str = None
    ) -> None:
        """
        Constructor for :class:`~Layout.Layout.Element`.

        :param window: :class:`~Layout.Layout.Window` that contains element
        """
        self.window = window

        if isinstance(name, str):
            self.name = name

        self.getWindowRunner = window.getWindowRunner
        self.getDimensions = window.getDimensions
        self.getKey = window.getKey

    def getName(self) -> str:
        """
        Get name of element.

        :param self: :class:`~Layout.Layout.Element` to retrieve name from
        """
        return self.name

    def getWindowObject(self) -> Window:
        """
        Get :class:`~Layout.Layout.Window` that contains element.

        :param self: :class:`~Layout.Layout.Element` to retrieve window from
        """
        return self.window

    def getElement(self) -> sg.Column:
        """
        Get object as PySimpleGUI element.

        :param self: :class:`~Layout.Layout.Element` to retrieve element of
        """
        return sg.Column(
            layout=self.getLayout()
        )


class Row:
    """
    Row of PySimpleGUI elements.

    :ivar name: optional name of row
    :ivar window: :class:`~Layout.Layout.Window` that contains element
    :ivar elements: elements contained in row
    :ivar getWindowRunner: pointer to :meth:`~Layout.Layout.Window.getWindowRunner`
    :ivar getDimensions: pointer to :meth:`~Layout.Layout.Window.getDimensions`
    :ivar getKey: pointer to :meth:`~Layout.Layout.Window.getKey`
    """

    def __init__(
        self,
        name: str = None,
        elements: Union[sg.Element, Iterable[sg.Element]] = None,
        window: Window = None
    ) -> None:
        """
        Constructor for :class:`~Layout.Layout.Row`.

        :param name: name of row
        :param elements: initial elements to contain in row
        :param window: :class:`~Layout.Layout.Window` that contains row
        """
        self.name = name
        self.window = window

        if isinstance(elements, sg.Element):
            self.elements = [elements]
        elif isinstance(elements, list):
            self.elements = elements
        elif elements is None:
            self.elements = []
        else:
            raise TypeError("elements must be sg.Element or list")

        if window is not None:
            self.getDimensions = window.getDimensions
            self.getWindowRunner = window.getWindowRunner
            self.getKey = window.getKey

    def getName(self) -> str:
        """
        Get name of row.

        :param self: :class:`~Layout.Layout.Row` to retrieve name from
        """
        return self.name

    def addElements(self, elements: Union[sg.Element, Iterable[sg.Element]]) -> None:
        """
        Add elements to row.

        :param self: :class:`~Layout.Layout.Row` to add elements to
        :param elements: element(s) to add to row
        """

        def add(element: sg.Element) -> None:
            """Base method for :meth:`~Layout.Layout.Row.addElements`"""
            self.elements.append(element)

        return recursiveMethod(
            base_method=add,
            args=elements,
            valid_input_types=sg.Element,
            output_type=list
        )

    def getElements(self) -> List[sg.Element]:
        """
        Get elements contained in row.

        :param self: :class:`~Layout.Layout.Row` to retrieve elements from
        """
        return self.elements

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for elements in row.

        :param self: :class:`~Layout.Layout.Row` to retrieve layout from
        """
        return [self.getElements()]

    def getWindowObject(self) -> Window:
        """
        Get :class:`~Layout.Layout.Window` that contains row.

        :param self: :class:`~Layout.Layout.Row` to retrieve window from
        """
        return self.window

    def getAsColumn(self) -> sg.Column:
        """
        Get elements in rows as single :class:`~PySimpleGUI.Column` object.

        :param self: :class:`~Layout.Row.Row` to retrieve object from
        """
        layout = self.getLayout()
        column = sg.Column(layout)
        return column


class Layout:
    """
    Container for :class:`~Layout.Layout.Row`.

    :ivar rows: rows contained in layout
    """

    def __init__(self, rows: Union[Row, Iterable[Row]] = None) -> None:
        """
        Constructor for :class:`~Layout.Layout.Layout`.

        :param rows: initial rows to contain in layout
        """
        if isinstance(rows, Row):
            self.rows = [rows]
        elif isinstance(rows, list):
            self.rows = rows
        elif rows is None:
            self.rows = []
        else:
            raise TypeError("rows must be Row or list")

    def addRows(self, rows: Union[Row, Iterable[Row]]) -> None:
        """
        Add rows to layout.

        :param self: :class:`~Layout.Layout.Layout` to add rows to
        :param rows: row(s) to add to layout
        """

        def add(row: Row) -> None:
            """Base method for :meth:`~Layout.Layout.Layout.addRows`"""
            self.rows.append(row)

        return recursiveMethod(
            base_method=add,
            args=rows,
            valid_input_types=Row,
            output_type=list
        )

    def getRows(self) -> List[Row]:
        """
        Get rows contained in layout.

        :param self: :class:`~Layout.Layout.Layout` to retrieve rows from
        """
        return self.rows

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get PySimpleGUI-friendly layout for rows in layout.

        :param self: :class:`~Layout.Layout.Layout` to retrieve layout from
        """
        try:
            assert False
            layout = rowsToColumnsLayout(self.getRows())
        except AssertionError:
            layout = [[]]
            for row_obj in self.getRows():
                layout += row_obj.getLayout()
        return layout


class TabRow(Row):
    """
    :class:`~Layout.Layout.Row` contained in :class:`~Layout.Layout.Tab`.

    :ivar tab: :class:`~Layout.Layout.Tab` that contains row
    :ivar instances: dictionary of all :class:`~Layout.Layout.TabRow` by subclass.
        Key is name of instance.
        Value is instance of class for name.
    """

    def __init__(self, name: str, tab: Tab, **kwargs) -> None:
        """
        Constructor for :class:`~Layout.Layout.TabRow`.

        :param name: name of row
        :param tab: :class:`~Layout.Layout.Tab` that contains row
        :param kwargs: additional arguments to pass into :meth:`~Layout.Layout.Row`
        """
        self.tab = tab
        super().__init__(name, window=tab.getWindowObject(), **kwargs)

    def getTab(self):
        """
        Get :class:`~Layout.Layout.Tab` that contains row.

        :param self: :class:`~Layout.Layout.TabRow` to retrieve tab from
        """
        return self.tab


class Tab(Element):
    """
    Wrapper for PySimpleGUI Tab.

    :ivar name: name of tab
    :ivar window: :class:`~Layout.Layout.Window` that contains tab
    :ivar getWindowRunner: pointer to :meth:`~Layout.Layout.Window.getWindowRunner`
    :ivar getDimensions: pointer to :meth:`~Layout.Layout.Window.getDimensions`
    :ivar getKey: pointer to :meth:`~Layout.Layout.Window.getKey`
    """

    def __init__(self, name: str, window: Window) -> None:
        """
        Constructor for :class:`~Layout.Layout.Tab`.

        :param name: name of tab
        :param window: :class:`~Layout.Layout.Window` that contains tab
        """
        super().__init__(window, name=name)

    def getTab(self) -> sg.Tab:
        """
        Get PySimpleGUI Tab for tab.

        :param self: :class:`~Layout.Layout.Tab` to retrieve tab from
        """
        return sg.Tab(
            title=self.getName(),
            layout=self.getLayout(),
            pad=(0, 0)
        )


class Frame(Element):
    def __init__(self, name: str, window: Window) -> None:
        """
        Constructor for :class:`~Layout.Layout.Frame`.

        :param name: name of tab
        :param window: :class:`~Layout.Layout.Window` that contains frame
        """
        super().__init__(window, name=name)

    def getFrame(self) -> sg.Frame:
        """
        Get PySimpleGUI Frame for frame.

        :param self: :class:`~Layout.Layout.Frame` to retrieve frame from
        """
        return sg.Frame(
            title=self.getName(),
            layout=self.getLayout(),
            pad=(0, 0)
        )


class TabGroup(Element):
    """
    Wrapper for PySimpleGUI TabGroup and container for :class:`~Layout.Layout.Tab`.

    :ivar name: name of tabgroup
    :ivar suffix_layout: layout shared by all tabs in tabgroup, appended after tabs
    :ivar tabs: :class:`~Layout.Layout.Tab` and/or PySimpleGUI contained in tabgroup
    """

    def __init__(
        self,
        tabs: Union[Union[sg.Tab, Tab], Iterable[Union[sg.Tab, Tab]]],
        name: str = None,
        suffix_layout: Layout = None
    ) -> None:
        """
        Constructor for :class:`~Layout.Layout.Tabgroup`.

        :param name: name of tabgroup
        :param suffix_layout: layout shared by all tabs in tabgroup, appended after tabs
        :param tabs: tabs contained in tabgroup
        """
        self.name = name
        self.suffix_layout = Layout() if suffix_layout is None else suffix_layout

        if isinstance(tabs, (Tab, sg.Tab)):
            self.tabs = [tabs]
        elif isinstance(tabs, list):
            self.tabs = tabs
        elif tabs is None:
            self.tabs = []
        else:
            raise TypeError("tabs must be Tab or list")

    def getName(self) -> str:
        """
        Get name of tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to retrieve name from
        """
        return self.name

    def getTabs(self) -> List[Tab]:
        """
        Get tabs contained in tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to retrieve tabs from
        """
        return self.tabs

    def getTabGroup(self) -> sg.TabGroup:
        """
        Get PySimpleGUI TabGroup for tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to retrieve tabgroup from
        """
        tabs = []
        for tab in self.getTabs():
            if isinstance(tab, Tab):
                tabs.append(tab.getTab())
            elif isinstance(tab, sg.Tab):
                tabs.append(tab)
            else:
                raise TypeError("tab must be Tab or sg.Tab")
        return sg.TabGroup([tabs])

    def getSuffixLayout(self) -> List[List[sg.Element]]:
        """
        Get layout appended to and mutual to all tabs in tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to retrieve layout from
        """
        return self.suffix_layout.getLayout()

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get PySimpleGUI-friendly layout for tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to retrieve layout from
        """
        # noinspection PyTypeChecker
        return [[self.getTabGroup()]] + self.getSuffixLayout()

    def getAsTab(self) -> sg.Tab:
        """
        Get PySimpleGUI Tab for tabgroup.

        :param self: :class:`~Layout.Layout.TabGroup` to generate tab from
        """
        return sg.Tab(
            layout=self.getLayout(),
            title=self.getName()
        )


class RadioGroup(Row):
    """
    Wrapper for :class:`~PySimpleGUI.Radio`.
    """

    def __init__(
        self,
        radios: Union[sg.Radio, List[sg.Radio]],
        group_id: str,
        window: Window,
        name: str = None
    ):
        """
        Constructor for :class:`~Layout.Layout.RadioGroup`.

        :param radios: collection of radio elements
        :param group_id: id of radio group
        :param window: :class:`~Layout.Layout.Window` that contains group
        :param name: name of radio group qua row
        """
        super().__init__(
            name=name,
            elements=radios,
            window=window
        )

        self.group_id = group_id

    def getGroupId(self) -> str:
        """
        Get id of radio group.

        :param self: :class:`~Layout.Layout.RadioGroup` to retrieve id from
        """
        return self.group_id

    def getRadios(self) -> List[sg.Radio]:
        """
        Get radio elements that constitute group.

        :param self: :class:`~Layout.Layout.RadioGroup` to retrieve radios from
        """
        return self.getElements()

    def getChosenRadio(self) -> sg.Radio:
        """
        Get radio in group that is set to true.

        :param self: :class:`~Layout.Layout.RadioGroup` to retrieve radio from
        """
        radios = self.getRadios()
        window_obj = self.getWindowObject()
        window_runner = window_obj.getWindowRunner()

        for radio in radios:
            radio_key = getKeys(radio)
            radio_value = window_runner.getValue(radio_key)
            if radio_value:
                chosen_radio = radio
                break

        assert isinstance(chosen_radio, sg.Radio)
        return chosen_radio


class CheckboxGroup(Row):
    def __init__(
        self,
        checkboxes: Union[sg.Checkbox, List[sg.Checkbox]],
        window: Window,
        name: str = None
    ):
        """
        Constructor for :class:`~Layout.Layout.CheckboxGroup`.

        :param radios: collection of radio elements
        :param window: :class:`~Layout.Layout.Window` that contains group
        :param name: name of checkbox group
        """
        super().__init__(
            name=name,
            elements=checkboxes,
            window=window
        )
        
    def getCheckboxes(self) -> List[sg.Checkbox]:
        """
        Get checkboxes constituting group.

        :param self: :class:`~Layout.Layout.CheckboxGroup` to retrieve checkboxes from
        """
        checkboxes = self.getElements()
        return checkboxes

    def getCheckedCheckboxes(self) -> List[sg.Checkbox]:
        """
        Get checkboxes in group that are set to true.

        :param self: :class:`~Layout.Layout.CheckboxGroup` to retrieve checkboxes from
        """
        checkboxes = self.getCheckboxes()
        window_obj = self.getWindowObject()
        window_runner = window_obj.getWindowRunner()

        checked_checkboxes = []
        for checkbox in checkboxes:
            checkbox_key = getKeys(checkbox)
            checkbox_value = window_runner.getValue(checkbox_key)
            if checkbox_value:
                checked_checkboxes.append(checkbox)

        return checked_checkboxes

class Window:
    """
    Wrapper for PySimpleGUI Window.

    :ivar name: name of window
    :ivar runner: :class:`~Layout.Layout.WindowRunner` that runs window
    :ivar dimensions: dictionary of dimensions for elements contained in window
    :ivar key_list: :class:`~KeyList.KeyList` for element keys in window
    :ivar getPrefix: pointer to :meth:`~KeyList.KeyList.getPrefix`
    :ivar getKey: pointer to :meth:`~KeyList.KeyList.getKey`
    :ivar getKeyList: pointer to :meth:`~KeyList.KeyList.getKeyList`
    """

    def __init__(
        self,
        name: str,
        runner: WindowRunner,
        dimensions: Dict[str, Tuple[Optional[float], Optional[float]]] = None
    ) -> None:
        """
        Constructor for :class:`~Layout.Layout.Window`.

        :param name: name of window
        :param runner: :class:`~Layout.Layout.WindowRunner` that runs window
        :param dimensions: dictionary of dimensions for elements contained in window
        """
        self.name = name
        self.runner = runner
        self.dimensions = dimensions

        key_list = KeyList()
        self.key_list = key_list

        self.getPrefix = key_list.getPrefix
        self.getKey = key_list.getKey
        self.getKeyList = key_list.getKeyList

    def getName(self) -> str:
        """
        Get name of window.

        :param self: :class:`~Layout.Layout.Window` to retrieve name from
        """
        return self.name

    def getWindowRunner(self) -> WindowRunner:
        """
        Get :class:`~Layout.Layout.WindowRunner` that runs window.

        :param self: :class:`~Layout.Layout.Window` to retrieve runner from
        """
        return self.runner

    def getDimensions(
        self,
        name: str = None
    ) -> Union[dict, Tuple[Optional[float], Optional[float]]]:
        """
        Get dimensions for elements in window.

        :param self: :class:`~Layout.Layout.Window` to retrieve dimensions from
        :param name: name of dimension to retrieve
        :returns: Full dictionary of dimensions
            if :paramref:`~Layout.Layout.Window.getDimensions.name` is None.
            Single dimension
            if :paramref:`~Layout.Layout.Window.getDimensions.name` is str.
        """
        if isinstance(name, str):
            dimensions = self.getDimensions()
            return dimensions[name]
        elif name is None:
            return self.dimensions
        else:
            raise TypeError("column must be str")

    @storeElement
    def getWindow(self) -> sg.Window:
        """
        Get PySimpleGUI Window for window.

        :param self: :class:`~Layout.Layout.Window` to retrieve window from
        """
        return sg.Window(
            title=self.getName(),
            layout=self.getLayout(),
            grab_anywhere=False,
            size=self.getDimensions("window"),
            finalize=True
        )


class ChooseChecksWindow(Window):
    """
    This class contains the layout for the choose-by-checks window.
        #. Menu: This allows user to (un)check all selections.
        #. Header to indicate purpose of each column.
        #. Footer with submit and cancel buttons.
        #. Row for each selection, allowing user to choose using checkbox.
    """

    def __init__(
        self,
        name: str,
        runner: WindowRunner,
        get_rows: Callable[[], List[Row]],
        header_text: str = ''
    ):
        """
        Constructor for :class:`~Layout.Layout.ChooseChecksWindow`.

        :param name: name of window
        :param runner: :class:`~Layout.Layout.WindowRunner` that window is stored in
        :param get_rows: callable to retrieve selection rows from for window
        """
        dimensions = {
            "window": (None, None)  # dim
        }
        super().__init__(name, runner, dimensions=dimensions)

        self.header_text = header_text
        self.getRows = get_rows

    def getMenu(self) -> sg.Menu:
        """
        Get toolbar menu for window.

        :param self: :class:`~Layout.Layout.ChooseChecksWindow` to retrieve menu from
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
        return sg.Menu(
            menu_definition=menu_definition,
            key=self.getKey("toolbar_menu")
        )

    def getHeaderRow(self) -> Row:
        """
        Get header for window.
        This includes labels, which indicate the purpose of each column in the window.

        :param self: :class:`~Layout.Layout.ChooseChecksWindow` to retrieve header from
        """
        text = self.header_text
        text_element = sg.Text(text=text)
        row = Row(elements=text_element)
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

    def getRowsLayout(self) -> Layout:
        """
        Get layout for scrollable section containing row for each selection.

        :param self: :class:`~Layout.Layout.ChooseChecksWindow` to retrieve section from
        """
        column = sg.Column(
            layout=Layout(rows=self.getRows()).getLayout(),
            size=(None, 350),  # dim
            scrollable=True,
            vertical_scroll_only=True
        )
        layout = Layout(rows=Row(elements=column))
        return layout

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for window.

        :param self: :class:`~Layout.Layout.ChooseChecksWindow` to retrieve layout from
        """
        menu = Layout(rows=Row(elements=self.getMenu()))
        header = self.getHeaderRow()
        footer = self.getFooterRow()
        rows = self.getRowsLayout()
        return menu.getLayout() + header.getLayout() + rows.getLayout() + footer.getLayout()


class TabbedWindow(Window):
    """
    Wrapper for :class:`~Layout.Layout.Window` with tabs.

    :ivar tabs: :class:`~Layout.Layout.Tab` and/or PySimpleGUI contained in window
    """

    def __init__(
        self,
        name: str,
        runner: WindowRunner,
        dimensions: Dict[str, Tuple[Optional[int], Optional[int]]] = None
    ) -> None:
        """
        Constructor for :class:`~Layout.Layout.TabbedWindow`.

        :param name: name of window
        :param runner: :class:`~Layout.Layout.WindowRunner` that runs window
        :param dimensions: dictionary of dimensions for elements contained in window
        """
        super().__init__(name, runner, dimensions=dimensions)
        self.tabs = []

    def addTabs(self, tabs: Union[Union[sg.Tab, Tab], Iterable[Union[sg.Tab, Tab]]]) -> None:
        """
        Add tabs to window

        :param self: :class:`~Layout.Layout.TabbedWindow` to add tabs into
        :param tabs: tabs to contain in window
        """

        def add(tab: Union[sg.Tab, Tab]) -> None:
            """Base method for Layout.Layout.TabbedWindow.addTabs"""
            self.tabs.append(tab)

        return recursiveMethod(
            base_method=add,
            args=tabs,
            valid_input_types=(sg.Tab, Tab),
            output_type=list
        )

    def getTabs(self, names: Union[str, Iterable[str]] = None) -> List[sg.Tab]:
        """
        Get tabs contained in window as PySimpleGUI Tab.

        :param self: :class:`~Layout.Layout.TabbedWindow` to retrieve tabs from
        :param names: name(s) of tab(s) to retrieve.
            Defaults to all tabs.
        """

        if names is None:
            return self.tabs

        def get(name: str):
            """Base method for :meth:`~Layout.Layout.TabbedWindow.getTabs`"""
            for tab in self.tabs:
                if isinstance(tab, Tab):
                    if tab.getName() == name:
                        return tab
                elif isinstance(tab, sg.Tab):
                    if vars(tab)["Title"] == name:
                        return tab
            raise ValueError(f"tab {name:s} not found in {self.__class__.__name__:s}")

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=list
        )


class WindowRunner:
    """
    Object to run handle events in :class:`~Layout.Layout.Window`.

    :ivar window_obj: :class:`~Layout.Layout.Window` to run
    :ivar window: PySimpleGUI Window for :class:`~Layout.Layout.Window`
    :ivar values: most recent values read in PySimpleGUI Window
    :ivar getName: pointer to :meth:`~Layout.Layout.Window.getName`
    :ivar getPrefix: pointer to :meth:`~Layout.Layout.Window.getPrefix`
    :ivar getKey: pointer to :meth:`~Layout.Layout.Window.getKey`
    :ivar getKeyList: pointer to :meth:`~Layout.Layout.Window.getKeyList`
    """

    def __init__(self, window_obj: Window):
        """
        Constructor for :class:`~Layout.Layout.WindowRunner`.

        :param window_obj: :class:`~Layout.Layout.Window` to run
        """
        self.window_obj = window_obj
        self.window = None
        self.values = None

        self.getName = window_obj.getName
        self.getPrefix = window_obj.getPrefix
        self.getKey = window_obj.getKey
        self.getKeyList = window_obj.getKeyList

    def getWindow(self) -> sg.Window:
        """
        Get PySimpleGUI Window for runner.

        :param self: :class:`~Layout.Layout.WindowRunner` to retrieve window from
        """
        return self.getWindowObject().getWindow()

    def getWindowObject(self) -> Window:
        """
        Get :class:`~Layout.Layout.Window` for runner.

        :param self: :class:`~Layout.Layout.WindowRunner` to retrieve window from
        """
        return self.window_obj

    def getValue(self, key: str, combo_error: bool = True) -> Union[str, float]:
        """
        Get value of element at most recent event.

        :param self: :class:`~Layout.Layout.WindowRunner` to retrieve value from
        :param key: key for element
        :param combo_error: display popup error message if combobox value is not in default values.
            Set True to display message.
            Set False otherwise.
            Only called if element is sg.InputCombo.
        """
        element = self.getElements(key)
        value = self.values[key]
        if isinstance(element, sg.InputCombo):
            if combo_error and value not in element.Values:
                sg.PopupError(f"{value:s} not found in combobox ({key:s})")
        return value

    def getElements(self, keys: Union[str, Iterable[str]]) -> Union[sg.Element, Iterable[Element]]:
        """
        Get element(s) in window runner from key

        :param self: :class:`~Layout.Layout.WindowRunner` to retrieve element(s) from
        :param keys: key(s) of element(s) to retrieve
        """

        def get(key: str) -> sg.Element:
            """Base method for :meth:`~Layout.Layout.WindowRunner.getElements`"""
            return self.getWindow()[key]

        return recursiveMethod(
            base_method=get,
            args=keys,
            valid_input_types=str,
            output_type=list
        )

    def toggleVisibility(self, key: str) -> None:
        """
        Toggle whether GUI section is collapsed or not

        :param self: :class:`~Layout.Layout.WindowRunner` to toggle visibility in
        :param key: key of PySimpleGUI pin to toggle visibility for
        """
        element = self.getElements(key)
        element.update(visible=not element.visible)
