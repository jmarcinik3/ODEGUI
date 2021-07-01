"""
This file contains functions necessary to open the main GUI window.
This includes the main function that starts the main window.
"""
import os
from functools import partial
from os import mkdir
from os.path import isdir, isfile

# noinspection PyPep8Naming
import PySimpleGUI as sg
from sympy.printing.preview import preview

import YML
from Layout.MainWindow import MainWindowRunner

def main():
    """
    Main function to run GUI
    """
    # tex2png()
    
    sg.ChangeLookAndFeel("DarkGrey13")
    sg.SetOptions(element_padding=(1, 1), suppress_error_popups = True, suppress_raise_key_errors=False)
    
    parameter_filenames = [
        os.path.join("parameters", filestem + ".yml")
        for filestem in ["parameters", "Martin2003_parameters", "Roongthumskull2011-7B_parameters", "func_params"]
    ]
    equation_filenames = [
        os.path.join("equations", filestem + ".yml")
        for filestem in ["hb-soma_eqs", "hb_eqs", "soma_eqs", "var_funcs"]
    ]
    time_evolution_layout = "tet_lay.yml"
    parameter_input_layout = "param_lay.yml"
    function_layout = "func_lay.yml"
    kwargs = {
        "name": "Hair Bundle/Soma Model",
        "parameter_filenames": parameter_filenames,
        "function_filenames": equation_filenames,
        "time_evolution_filename": time_evolution_layout,
        "parameter_input_filename": parameter_input_layout,
        "function_filename": function_layout
    }
    gui = MainWindowRunner(**kwargs)
    gui.runWindow()
    
def tex2png(output_folder: str = "tex", tex_filename: str = "var2tex.yml", overwrite: bool = False) -> None:
    """
    Create PNG for quantity(s) in TeX form.
    File containing quantity name(s) to TeX math format must be made before call.
    
    :param output_folder: name of folder to save images in
    :param tex_filename: name of file containing name-to-TeX conversions.
        Keys in file are name of quantity.
        Values are corresponding TeX format.
    :param overwrite: set True to overwrite existing quantity if name already exists.
        Set False to skip quantities previously saved as TeX image.
    """
    tex_yml = YML.readVar2Tex(tex_filename)
    
    kwargs = {
        "packages": ("amsmath", "amsfonts", "amssymb", "mathtools"),
        "viewer": "file",
        "euler": False
    }
    create_png = partial(preview, **kwargs)
    
    if not isdir(output_folder): mkdir(output_folder)
    for key in YML.readVar2Tex(tex_filename):
        filepath = f"{output_folder:s}/{key:s}.png"
        if not isfile(filepath) or overwrite: create_png(tex_yml[key], filename=filepath)

if __name__ == "__main__": main()