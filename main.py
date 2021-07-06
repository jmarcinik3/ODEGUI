"""
This file contains functions necessary to open the main GUI window.
This includes the main function that starts the main window.
"""
import os

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
    sg.SetOptions(element_padding=(1, 1), suppress_error_popups=True, suppress_raise_key_errors=False)

    parameter_filenames = [
        os.path.join("parameters", filestem + ".yml")
        for filestem in ["parameters", "Martin2003", "Roongthumskul2011-7B", "Barral2018-5Aorange", "func_params"]
    ]
    equation_filenames = [
        os.path.join("equations", filestem + ".yml")
        for filestem in ["Ohms_law", "Martin2003", "Roongthumskul2011", "Barral2018", "soma_eqs", "var_funcs"]
    ]
    time_evolution_layout = "tet_lay.yml"
    parameter_input_layout = "param_lay.yml"
    function_layout = "func_lay.yml"
    kwargs = {
        "name": "Hair Bundle/Soma Model",
        "parameter_filenames": parameter_filenames,
        "function_filenames": equation_filenames,
        "time_evolution_layout": time_evolution_layout,
        "parameter_layout": parameter_input_layout,
        "function_layout": function_layout
    }
    gui = MainWindowRunner(**kwargs)
    gui.runWindow()


if __name__ == "__main__":
    main()
