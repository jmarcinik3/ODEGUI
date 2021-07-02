from __future__ import annotations

from os.path import basename
from typing import Dict, List, Union

# noinspection PyPep8Naming
import PySimpleGUI as sg
from pint import Quantity

from Layout.Layout import Layout, Row, Window, WindowRunner
from macros import formatQuantity, getTexImage


class ChooseParameterRow(Row):
    def __init__(self, name: str, quantity: Quantity, window: ChooseParametersWindow):
        super().__init__(name, window=window)
        self.quantity = quantity

        elements = [
            self.getParameterLabel(),
            self.getQuantityLabel(),
            self.getCheckbox()
        ]
        self.addElements(elements)

    def getQuantity(self) -> Quantity:
        return self.quantity

    def getParameterLabel(self) -> Union[sg.Image, sg.Text]:
        kwargs = {
            "name": self.getName(),
            "size": (110, None)  # dim
        }
        return getTexImage(**kwargs)

    def getQuantityLabel(self) -> sg.Text:
        kwargs = {
            "text": formatQuantity(self.getQuantity()),
            "size": (10, None)  # dim
        }
        return sg.Text(**kwargs)

    def getCheckbox(self) -> sg.Checkbox:
        kwargs = {
            "text": "Overwrite?",
            "default": True,
            "key": self.getName()
        }
        return sg.Checkbox(**kwargs)


class ChooseParametersWindow(Window):
    def __init__(
            self,
            name: str,
            runner: ChooseParametersWindowRunner,
            quantities: Dict[str, Quantity] = None,
            filename: str = ''
    ):
        dimensions = {
            "window": (None, None)  # dim
        }
        super().__init__(name, runner, dimensions=dimensions)
        self.parameter_rows = []
        self.quantities = quantities
        self.filename = filename

    def getFilename(self) -> str:
        return self.filename

    def addQuantity(self, name: str, quantity: Quantity) -> None:
        self.quantities[name] = quantity

    def addQuantities(self, quantities: Dict[str, Quantity]) -> None:
        for name, quantity in quantities.items():
            self.addQuantity(name, quantity)

    def getQuantities(self) -> Dict[str, Quantity]:
        return self.quantities

    def getParameterRows(self) -> List[ChooseParameterRow]:
        rows = [
            ChooseParameterRow(name, quantity, self)
            for name, quantity in self.getQuantities().items()
        ]
        return rows

    def getHeaderRow(self) -> Row:
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
        submit_button = sg.Submit()
        cancel_button = sg.Cancel()
        row = Row(elements=[submit_button, cancel_button])
        return row

    def getParameterRowsLayout(self) -> Layout:
        kwargs = {
            "layout": Layout(rows=self.getParameterRows()).getLayout(),
            "size": (None, 350),  # dim
            "scrollable": True,
            "vertical_scroll_only": True
        }
        layout = Layout(rows=Row(elements=sg.Column(**kwargs)))
        return layout

    def getLayout(self) -> List[List[sg.Element]]:
        header = self.getHeaderRow()
        footer = self.getFooterRow()
        parameter_rows = self.getParameterRowsLayout()
        return header.getLayout() + parameter_rows.getLayout() + footer.getLayout()


class ChooseParametersWindowRunner(WindowRunner):
    def __init__(self, name: str, **kwargs):
        window = ChooseParametersWindow(name, self, **kwargs)
        super().__init__(name, window)

    def getChosenParameters(self) -> List[str]:
        event, parameters = self.getWindow().read(close=True)
        if event in [sg.WIN_CLOSED, "Cancel"]:
            return []
        elif event == "Submit":
            return [name for name in parameters if parameters[name]]
