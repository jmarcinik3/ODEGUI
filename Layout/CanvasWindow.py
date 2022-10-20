import tkinter as tk

import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Layout.Layout import storeElement


class CanvasWindow:
    def __init__(self) -> None:
        self.figure_canvas = None

    @storeElement
    def getCanvas(self) -> sg.Canvas:
        """
        Get canvas where figures will be plotted.

        :param self: :class:`~Layout.CanvasWindow.CanvasWindow` to retrieve canvas from
        """
        return sg.Canvas(key="-CANVAS-")

    def getFigureCanvas(self) -> FigureCanvasTkAgg:
        """
        Get figure-canvas in present state.

        :param self: :class:`~Layout.CanvasWindow.CanvasWindow` to retrieve figure-canvas from
        :returns: Figure-canvas object if figure has been drawn on canvas previously. None otherwise.
        """
        return self.figure_canvas

    @staticmethod
    def drawFigure(canvas: tk.Canvas, figure: Figure) -> FigureCanvasTkAgg:
        """
        Draw figure on canvas.

        :param canvas: canvas to draw figure on
        :param figure: figure containing data to draw on canvas
        """
        figure_canvas = FigureCanvasTkAgg(figure, canvas)
        figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        figure_canvas.draw()
        return figure_canvas

    def clearFigure(self, figure_canvas: FigureCanvasTkAgg = None) -> None:
        """
        Clear figure on figure-canvas aggregate.

        :param figure_canvas: figure-canvas aggregate to clear canvas on
        """
        if figure_canvas is None:
            figure_canvas = self.getFigureCanvas()

        if isinstance(figure_canvas, FigureCanvasTkAgg):
            figure_canvas.get_tk_widget().forget()

        plt.close("all")

    def updateFigureCanvas(
        self,
        figure: Figure,
    ) -> FigureCanvasTkAgg:
        """
        Update figure-canvas aggregate in simulation window.
        This plots the most up-to-date data and aesthetics on the plot.

        :param self: :class:`~Layout.CanvasWindow.CanvasWindow` to update figure-canvas in
        :param figure: new figure (containing most up-to-date info) to plot
        :returns: New figure-canvas stored in window.
        """
        assert isinstance(figure, Figure)

        figure_canvas = self.getFigureCanvas()
        if isinstance(figure_canvas, FigureCanvasTkAgg):
            self.clearFigure(figure_canvas)

        canvas = self.getCanvas()
        self.figure_canvas = self.drawFigure(canvas.TKCanvas, figure)
        self.getWindow().Refresh()
        return self.figure_canvas
