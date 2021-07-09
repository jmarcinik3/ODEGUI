from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg

from KeyList import KeyList


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


class Element:
    def __init__(self, window: Window) -> None:
        self.window = window

    def getWindowObject(self) -> Window: return self.window

    def getWindowRunner(self) -> WindowRunner: return self.getWindowObject().getWindowRunner()

    def getDimensions(self, **kwargs):
        return self.getWindowObject().getDimensions(**kwargs)

    def getElement(self) -> sg.Column:
        kwargs = {
            "layout": self.getLayout()
        }
        return sg.Column(**kwargs)

    def getKey(self, prefix: str, tag: str = None) -> str: return self.getWindowObject().getKey(prefix, tag)


class Row:
    def __init__(
            self,
            name: str = None,
            elements: Union[sg.Element, List[sg.Element]] = None,
            window: Window = None,
            dimensions: Dict[Tuple[Optional[float], Optional[float]]] = None
    ) -> None:
        self.name = name
        self.window = window
        self.dimensions = dimensions

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
        return self.name

    def addElements(self, elements: Union[sg.Element, List[sg.Element]]) -> None:
        if isinstance(elements, sg.Element):
            self.elements.append(elements)
        elif isinstance(elements, list):
            for element in elements:
                self.addElements(element)
        else:
            raise TypeError("elements must be sg.Element or list")

    def getRow(self) -> List[sg.Element]:
        return self.elements

    def getLayout(self) -> List[List[sg.Element]]:
        return [self.elements]

    def getWindowObject(self) -> Window:
        return self.window


class Layout:
    def __init__(self, rows: Union[Row, List[Row]] = None) -> None:
        if isinstance(rows, Row):
            self.rows = [rows]
        elif isinstance(rows, list):
            self.rows = rows
        elif rows is None:
            self.rows = []
        else:
            raise TypeError("rows must be sg.Element or list")

    def addRows(self, rows: Union[Row, List[Row]]) -> None:
        if isinstance(rows, Row):
            self.rows.append(rows)
        elif isinstance(rows, list):
            for row in rows:
                self.addRows(row)
        else:
            raise TypeError("rows must be Row or list")

    def getRows(self) -> List[Row]:
        return self.rows

    def getLayout(self) -> List[List[sg.Element]]:
        layout = [[]]
        for row in self.getRows():
            layout += row.getLayout()
        return layout


class TabRow(Row):
    def __init__(self, name: str, tab: Tab, **kwargs) -> None:
        self.tab = tab
        super().__init__(name, window=tab.getWindowObject(), **kwargs)

    def getTab(self): return self.tab


class Tab:
    def __init__(self, name: str, window: Window) -> None:
        self.name = name
        self.window = window

        self.getDimensions = window.getDimensions
        self.getWindowRunner = window.getWindowRunner
        self.getKey = window.getKey

    def getName(self) -> str: return self.name

    def getWindowObject(self) -> Window: return self.window

    def getTab(self) -> sg.Tab:
        kwargs = {
            "title": self.getName(),
            "layout": self.getLayout(),
            "pad": (0, 0)
        }
        return sg.Tab(**kwargs)


class TabGroup:
    def __init__(self, tabs: Union[Tab, List[Tab]], name: str = None, suffix_layout: Layout = None) -> None:
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
        return self.name

    def getTabs(self) -> List[Tab]:
        return self.tabs

    def getTabGroup(self) -> sg.TabGroup:
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
        return self.suffix_layout.getLayout()

    def getLayout(self) -> List[List[Union[sg.Element]]]:
        # noinspection PyTypeChecker
        return [[self.getTabGroup()]] + self.getSuffixLayout()

    def getAsTab(self) -> sg.Tab:
        kwargs = {
            "layout": self.getLayout(),
            "title": self.getName()
        }
        return sg.Tab(**kwargs)


class Window:
    def __init__(
            self, name: str, runner: WindowRunner, dimensions: Dict[str, Tuple[Optional[float], Optional[float]]] = None
    ) -> None:
        self.name = name
        self.runner = runner
        self.dimensions = dimensions
        self.key_list = KeyList()

    def getName(self) -> str:
        return self.name

    def getWindowRunner(self) -> WindowRunner:
        return self.runner

    def getDimensions(
            self, name: str = None
    ) -> Union[Dict[str, Tuple[Optional[float], Optional[float]]], Tuple[Optional[float], Optional[float]]]:
        if isinstance(name, str):
            dimensions = self.getDimensions()
            return dimensions[name]
        elif name is None:
            return self.dimensions
        else:
            raise TypeError("column must be str")

    def getPrefix(self, *args, **kwargs) -> str:
        """
        Get prefix for set of keys.
        
        :param self: :class:`~Layout.Layout.Window` to retrieve prefix from
        :param args: required arguments to pass into :meth:`~KeyList.KeyList`
        :param kwargs: additional arguments to pass into :meth:`~KeyList.KeyList`
        """
        return self.key_list.getPrefix(*args, **kwargs)

    def getKey(self, prefix: str, tag: str = None) -> str:
        return self.key_list.getKey(prefix, tag)

    def getKeyList(self, prefixes: str = None) -> List[str]:
        return self.key_list.getKeyList(prefixes)

    def getWindow(self) -> sg.Window:
        kwargs = {
            "title": self.getName(),
            "layout": self.getLayout(),
            "grab_anywhere": False,
            "size": self.getDimensions("window"),
            "finalize": True
        }
        return sg.Window(**kwargs)


class TabbedWindow(Window):
    def __init__(
            self, name: str, runner: WindowRunner, dimensions: Dict[str, Tuple[Optional[int], Optional[int]]] = None
    ) -> None:
        super().__init__(name, runner, dimensions=dimensions)
        self.tabs = []

    def addTabs(self, tabs):
        if isinstance(tabs, (Tab, sg.Tab)):
            self.tabs.append(tabs)
        elif isinstance(tabs, list):
            for tab in tabs:
                self.addTabs(tab)
        else:
            raise TypeError("tabs must be Tab, sg.Tab, or list")

    def getTabs(self) -> List[sg.Tab]:
        tabs = []
        for tab in self.tabs:
            if isinstance(tab, Tab):
                tabs.append(tab.getTab())
            elif isinstance(tab, sg.Tab):
                tabs.append(tab)
            else:
                raise TypeError("tab must be Tab or sg.Tab")
        return tabs


class WindowRunner:
    def __init__(self, window_object: Window):
        self.window_object = window_object
        self.window = None
        self.values = None

        self.getName = window_object.getName
        self.getPrefix = window_object.getPrefix
        self.getKey = window_object.getKey
        self.getKeyList = window_object.getKeyList

    def getWindow(self) -> sg.Window:
        if self.window is None:
            self.window = self.getWindowObject().getWindow()
        return self.window

    def getWindowObject(self) -> Window:
        return self.window_object

    def getValue(self, key: str) -> Union[str, float]:
        """
        __Purpose__
            Get value of element at most recent event
        __Inputs__
            key [str]: key for element
        """
        return self.values[key]

    def getElements(self, keys: Union[str, List[str]]) -> Union[sg.Element, List[sg.Element]]:
        """
        Get element(s) in window runner from key
        
        :param self: :class:`~Layout.Layout.WindowRunner` to retrieve element(s) from
        :param keys: key(s) of element(s) to retrieve
        """
        if isinstance(keys, str):
            return self.getWindow()[keys]
        elif isinstance(keys, list):
            return [self.getElements(keys=key) for key in keys]
        else:
            raise TypeError("keys must be str or list")

    def toggleVisibility(self, key: str) -> None:
        """
        __Purpose__
            Toggle whether GUI section is collapsed or not
            Toggles is_visible attribute of GUI for corresponding element-section
        __Inputs__
            key [str]: key for element-section to toggle visible (collapseness)
        """
        element = self.getElements(key)
        element.update(visible=not element.visible)
