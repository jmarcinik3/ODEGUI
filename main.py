"""
This file contains functions necessary to open the main GUI window.
This includes the main function that starts the main window.
"""

# noinspection PyPep8Naming
from os import listdir
from os.path import isfile, join

# noinspection PyPep8Naming
import PySimpleGUI as sg

from Layout.MainWindow import MainWindowRunner
from macros import tex2pngFromFile


def main():
    """
    Main function to run GUI
    """
    tex2pngFromFile("tex", "var2tex.yml")

    sg.ChangeLookAndFeel("DarkGrey13")
    sg.SetOptions(
        element_padding=(1, 1),
        suppress_error_popups=True,
        suppress_raise_key_errors=False
    )

    param_dir = "parameters"
    parameter_filepaths = [
        join(param_dir, filepath)
        for filepath in listdir(param_dir)
        if isfile(join(param_dir, filepath))
    ]
    eq_dir = "equations"
    equation_filepaths = [
        join(eq_dir, filepath)
        for filepath in listdir(eq_dir)
        if isfile(join(eq_dir, filepath))
    ]
    
    time_evolution_layout = "tet_lay.json"
    parameter_input_layout = "param_lay.json"
    function_layout = "func_lay.json"
    kwargs = {
        "name": "Hair Bundle/Soma Model",
        "parameter_filepaths": parameter_filepaths,
        "function_filepaths": equation_filepaths,
        "time_evolution_layout": time_evolution_layout,
        "parameter_layout": parameter_input_layout,
        "function_layout": function_layout
    }
    gui = MainWindowRunner(**kwargs)
    gui.runWindow()


if __name__ == "__main__":
    main()
