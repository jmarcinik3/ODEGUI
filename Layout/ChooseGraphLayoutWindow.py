from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

import PySimpleGUI as sg

from CustomErrors import RecursiveTypeError
from Layout.Layout import Layout, Row, Window, WindowRunner


class ChooseGraphLayoutWindow(Window):
    def __init__(self, name: str, runner: ChooseGraphLayoutWindowRunner) -> None:
        dimensions = {
            "window": (None, None)  # dim
        }
        super().__init__(name, runner, dimensions=dimensions)
        self.choice2lay = {
            "Circle": "circle",
            "Distributed Recursive": "drl",
            "Fruchterman-Reingold Force-Directed": "fr",
            "Kamada-Kawai Force-Directed": "kk",
            "Large Graph": "lgl"
        }

    def getLayoutChoices(self) -> List[str]:
        """
        Get proper choices for graph layout.

        :param self: `~Layout.Layout.ChooseGraphLayoutWindow` to retrieve choices from
        """
        return list(self.choice2lay.keys())

    def getLayoutCodes(self, choices: Union[str, List[str]] = None) -> Union[str, Dict[str, str]]:
        """
        Get layout codes for igraph from full layout name.

        :param self: `~Layout.Layout.ChooseGraphLayoutWindow` to retrieve code(s) from
        :param choices: name(s) of layout(s) to retrieve code(s) for.
            Defaults to all layouts.
        """
        if isinstance(choices, str):
            return self.choice2lay[choices]
        elif isinstance(choices, Iterable):
            return {choice: self.getLayoutCodes(choices=choice) for choice in choices}
        elif choices is None:
            return self.getLayoutCodes(choices=self.getLayoutChoices())
        else:
            raise RecursiveTypeError(choices)

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

    def getChooseLayoutElements(self) -> List[sg.Radio]:
        """
        Get elements to choose graph layout name.

        :param self: `~Layout.Layout.ChooseGraphLayoutWindow` to retrieve elements from
        """
        radios = []
        for choice in self.getLayoutChoices():
            kwargs = {
                "text": choice,
                "default": False,
                "group_id": 0,
                "key": self.getKey("layout_choice", choice)
            }
            radios.append(sg.Radio(**kwargs))
        return radios

    def getLayout(self) -> List[List[sg.Element]]:
        """
        Get layout for window.

        :param self: `~Layout.Layout.ChooseGraphLayoutWindow` to retrieve layout from
        """
        radios = self.getChooseLayoutElements()
        radio_layout = Layout()
        for radio in radios:
            radio_layout.addRows(rows=Row(window=self, elements=radio))
        footer_row = self.getFooterRow()
        return radio_layout.getLayout() + footer_row.getLayout()


class ChooseGraphLayoutWindowRunner(WindowRunner):
    def __init__(self, name: str) -> None:
        window_object = ChooseGraphLayoutWindow(name, self)
        super().__init__(window_object)

        self.getLayoutChoices = window_object.getLayoutChoices
        self.getLayoutCodes = window_object.getLayoutCodes

    def getLayoutCode(self) -> Tuple[str, str]:
        """
        Get chosen layout from window.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseGraphLayoutWindow.ChooseGraphLayoutWindowRunner` to retrieve choice from
        """
        window = self.getWindow()
        event = ''
        while event not in [sg.WIN_CLOSED, "Cancel"]:
            event, self.values = window.read()

            if event == "Submit":
                layout_code = self.getChosenLayoutCode()
                print(layout_code)
                if layout_code is not None:
                    window.close()
                    return event, layout_code
                else:
                    sg.PopupError("Select layout name or hit cancel button")
        window.close()
        return event, None

    def getChosenLayoutCode(self) -> str:
        """
        Get graph layout code chosen by user.
        Uses present state of window.

        :param self: :class:`~Layout.ChooseGraphLayoutWindow.ChooseGraphLayoutWindowRunner` to retrieve choice from
        """
        for layout_choice in self.getLayoutChoices():
            radio_key = self.getKey("layout_choice", layout_choice)
            radio_bool = self.getValue(radio_key)
            if radio_bool:
                layout_code = self.getLayoutCodes(choices=layout_choice)
                return layout_code
