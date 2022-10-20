"""
This file contains functions necessary to open the main GUI window.
This includes the main function that starts the main window.
"""

from os import listdir
from os.path import isfile, join

import PySimpleGUI as sg

from Layout.MainWindow import MainWindowRunner
from macros import tex2pngFromFile
from Config import specific_model_folderpath, var2tex_filepath

def main():
    """
    Main function to run GUI
    """
    tex2pngFromFile("tex", var2tex_filepath)

    sg.ChangeLookAndFeel("DarkGrey11")
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
    
    time_evolution_layout = join(specific_model_folderpath, "variable_layout.json")
    parameter_input_layout = join(specific_model_folderpath, "parameter_layout.json")
    function_layout = join(specific_model_folderpath, "function_layout.json")
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
