from __future__ import annotations

import subprocess
import warnings
from collections.abc import KeysView
from functools import partial
from os.path import dirname
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Type, Union

import numpy as np
from igraph import Graph
from metpy.units import units
from numpy import ndarray
from pint import Quantity
from sympy import Expr
from sympy import Piecewise as spPiecewise
from sympy import Symbol, cosh, exp, latex, ln, pi, sin, solve, symbols, var
from sympy.core import function
from sympy.utilities.lambdify import lambdify

from CustomErrors import RecursiveTypeError
from macros import formatUnit, formatValue, recursiveMethod, unique
from YML import config_file_extensions, loadConfig, saveConfig

var2tex = loadConfig("var2tex.yml")


class PaperQuantity:
    """
    Stores information about objects generated from files, for models.

    :ivar name: name of object
    :ivar filestem: stem of file where object was loaded from
    :ivar model: model that contains object
    """

    def __init__(
        self,
        name: str,
        model: Model = None,
        filestem: str = None
    ) -> None:
        """
        Constructor for :class:`~Function.PaperQuantity`.

        :param name: name of object
        :param model: model that contains object
        :param filestem: stem of file where object was loaded from
        """
        assert isinstance(name, str)
        assert isinstance(filestem, str) or filestem is None
        assert isinstance(model, Model) or model is None

        self.name = name
        self.filestem = filestem
        self.model = model

    def getSymbol(self) -> Symbol:
        """
        Get symbol of object.

        :param self: :class:`~Function.PaperQuantity` to retrieve symbol from
        """
        name = self.getName()
        symbol = SymbolicVariables.getVariables(names=name)
        return symbol

    def getName(self) -> str:
        """
        Get name of object.

        :param self: :class:`~Function.PaperQuantity` to retrieve name from
        """
        return self.name

    def getStem(self) -> str:
        """
        Get filestem for file that generated object.

        :param self: :class:`~Function.PaperQuantity` to retrieve filestem from
        """
        return self.filestem

    def getModel(self) -> Model:
        """
        Get :class:`~Function.Model` that contains object.

        :param self: :class:`~Function.PaperQuantity` to retrieve model from
        """
        return self.model

    def setModel(self, model: Model) -> None:
        """
        Set :class:`~Function.Model` that contains object.

        :param self: :class:`~Function.PaperQuantity` to set model for
        :param model: model to set for object
        """
        assert isinstance(model, Model)

        self.model = model


class Parameter(PaperQuantity):
    """
    Store info pertinent to generate/simulate parameter.

    :ivar name: name of parameter
    :ivar quantity: quantity containing value and unit for parameter
    :ivar model: model that parameter is stored in
    """

    def __init__(self, name: str, quantity: Quantity, **kwargs) -> None:
        """
        Constructor for :class:`~Function.Parameter`.

        :param name: name of parameter
        :param kwargs: additional arguments to pass into :class:`~Function.PaperQuantity`
        """
        assert isinstance(quantity, Quantity)
        super().__init__(name, **kwargs)
        self.quantity = quantity

    def getQuantity(self) -> Quantity:
        """
        Get quantity (value and unit) for parameter.

        :param self: :class:`~Function.Parameter` to retrieve quantity from
        """
        return self.quantity

    def getFunctions(self, **kwargs) -> List[Function]:
        """
        Get functions that rely on parameter.

        :param self: :class:`~Function.Parameter` that :class:`~Function.Function` rely on
        :param kwargs: additional arguments to pass into :meth:`~Function.Function.getFreeSymbols`
        """
        name = self.getName()
        symbol = self.getSymbol()
        model = self.getModel()
        model_functions = model.getFunctions()
        functions = [
            function_object
            for function_object in model_functions
            if symbol in function_object.getParameters(**kwargs)
        ]
        return functions

    def getSaveContents(self) -> dict:
        """
        Get sufficient info from parameter for future recreation.

        :param self: :class:`~Function.Parameter` to retrieve info from
        """
        contents = {}
        quantity = self.getQuantity()
        contents["value"] = quantity.magnitude
        contents["unit"] = str(quantity.units)
        return contents


class Variable(PaperQuantity):
    """
    Store info pertinent to generate/simulate variable.

    :ivar name: name of variable
    :ivar time_evolution_type: time-evolution type of variable, i.e. how variable evolves over time
    :ivar model: model that variable is stored in
    """

    def __init__(
        self,
        name: str,
        time_evolution_type: str = "Temporal",
        initial_condition: float = 0.,
        **kwargs
    ) -> None:
        """
        Constructor for :class:`~Function.Variable`.

        :param name: name of variable
        :param time_evolution_type: time-evolution type of variable
        :param kwargs: additional arguments to pass into :class:`~Function.PaperQuantity`
        """
        assert isinstance(time_evolution_type, str)
        assert isinstance(initial_condition, (int, float))

        super().__init__(name, **kwargs)
        self.time_evolution_type = time_evolution_type

        self.initial_condition = None
        self.setInitialCondition(initial_condition)

    def getTimeEvolutionType(self) -> str:
        """
        Get time-evolution type for variable.

        :param self: :class:`~Function.Variable` to retrieve time evolution from
        """
        return self.time_evolution_type

    def getInitialCondition(self) -> float:
        """
        Get initial value for variable.

        :param self: :class:`~Function.Variable` to retrieve value from
        """
        return self.initial_condition

    def setInitialCondition(self, value: Union[str, float]) -> None:
        """
        Set initial condition or value for variable.

        :param self: :class:`~Function.Derivative` to set initial condition for
        :param value: initial condition to set for variable
        """
        self.initial_condition = value

    def getFunctions(self, **kwargs) -> List[Function]:
        """
        Get functions that rely on variable.

        :param self: :class:`~Function.variable` that :class:`~Function.Function` rely on
        :param kwargs: additional arguments to pass into :meth:`~Function.Function.getFreeSymbols`
        """
        model = self.getModel()
        model_functions = model.getFunctions()
        symbol = Symbol(self.getName())
        functions = [
            function_object
            for function_object in model_functions
            if symbol in function_object.getVariables(**kwargs)
        ]
        return functions

    def getSaveContents(self) -> dict:
        """
        Get sufficient info from variable for future recreation.

        :param self: :class:`~Function.Variable` to retrieve info from
        """
        time_evolution_type = self.getTimeEvolutionType()
        initial_condition = self.getInitialCondition()
        contents = {
            "time_evolution_type": time_evolution_type,
            "initial_condition": initial_condition
        }
        return contents


class Model:
    """
    Container for Function and Parameter objects.

    :ivar functions: dictionary of functions in model.
        Key is name of function.
        Value is Function object for function.
    :ivar parameters: dictionary of parameters in model.
        Key is name of parameter.
        Value if Parameter object for parameter.
    """

    def __init__(
            self,
            variables: Union[Variable, List[Variable]] = None,
            functions: Union[Function, List[Function]] = None,
            parameters: Union[Parameter, List[Parameter]] = None
    ) -> None:
        """
        Constructor for :class:`~Function.Model`.

        :param functions: functions to initially include in model
        :param parameters: parameters to initially include in model
        """
        if not isinstance(variables, Variable) and variables is not None:
            for variable in variables:
                assert isinstance(variable, Variable)
        if not isinstance(parameters, Parameter) and parameters is not None:
            for parameter in parameters:
                assert isinstance(parameter, Parameter)
        if not isinstance(functions, Function) and functions is not None:
            for function in functions:
                assert isinstance(function, Function)

        self.variables = {}
        self.functions = {}
        self.parameters = {}
        if parameters is not None:
            self.addPaperQuantities(parameters)
        if functions is not None:
            self.addPaperQuantities(functions)
        if variables is not None:
            self.addPaperQuantities(variables)

    def addPaperQuantities(self, quantity_objects: Union[PaperQuantity, List[PaperQuantity]]) -> None:
        """
        Add Function object(s) to model.
        Set self as model for Function object(s).

        :param self: :class:`~Function.Model` to add function to
        :param quantity_objects: function(s) to add to model
        """

        def add(quantity_object) -> None:
            """Base method for :meth:`~Function.Model.addPaperQuantities`"""
            name = quantity_object.getName()

            if isinstance(quantity_object, Function):
                if quantity_object not in self.getFunctions():
                    if name in self.getFunctionNames():
                        print(f"Overwriting {name:s}={quantity_object.getExpression():} into model")
                        del self.functions[name]
                    if quantity_object.isParameter():
                        print(f"Overwriting function {name:s}={quantity_object.getExpression():} as parameter")
                    elif name in self.getParameterNames():
                        print(f"Overwriting parameter {name:s} as function {name:s}={quantity_object.getExpression():}")
                        del self.parameters[name]
                    if not quantity_object.isParameter():
                        self.functions[name] = quantity_object
            elif isinstance(quantity_object, Parameter):
                quantity = quantity_object.getQuantity()
                if name in self.getFunctionNames():
                    print(f"Overwriting function {name:s} as parameter {name:s}={formatQuantity(quantity)}")
                    del self.functions[name]
                elif name in self.getParameterNames():
                    print(f"Overwriting parameter {name:s}={formatQuantity(quantity):s} into model")
                self.parameters[name] = quantity_object
            elif isinstance(quantity_object, Variable):
                if name in self.getVariableNames():
                    print(f"Overwriting variable {name:s} into model")
                    del self.variables[name]
                self.variables[name] = quantity_object

            if quantity_object.getModel() is not self:
                quantity_object.setModel(self)

        return recursiveMethod(
            base_method=add,
            args=quantity_objects,
            valid_input_types=(Variable, Function, Parameter),
            output_type=list
        )

    def getParameterNames(self):
        """
        Get names of parameters stored in model.

        :param self: :class:`~Function.Model` to retrieve names from
        """
        return list(self.parameters.keys())

    def getFunctionNames(self) -> Union[str, List[str]]:
        """
        Get names of stored functions in order added.

        :param self: :class:`~Function.Model` to retrieve names from
        """
        return list(self.functions.keys())

    def getVariableNames(self) -> Union[str, List[str]]:
        """
        Get names of stored variables in order added.

        :param self: :class:`~Function.Model` to retrieve names from
        """
        return list(self.variables.keys())

    def getParameters(self, names: Union[str, List[str]] = None) -> Union[Parameter, List[Parameter]]:
        """
        Get parameter object stored in model.

        :param self: :class:`~Function.Model` to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retrieve.
            Defaults to all parameters.
        """

        def get(name: str) -> Parameter:
            """Base method for :meth:`~Function.Model.getParameters`"""
            return self.parameters[name]

        return recursiveMethod(
            base_method=get,
            args=names,
            valid_input_types=str,
            output_type=list,
            default_args=self.getParameterNames()
        )

    def getParameterQuantities(self, names: Union[str, List[str]] = None) -> Union[Quantity, Dict[str, Quantity]]:
        """
        Get parameter quantities stored in model.

        __Recursion Base__
            return single parameter: names [str]

        :param self: :class:`~Function.Model` to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retrieve
        :returns: Quantity for parameter if :paramref:~Function.Model.getParameters.names` is str.
            List of quantities if :paramref:~Function.Model.getParameters.names` is list.
        """

        def get(name: str) -> Quantity:
            """Base method for :meth:`~Function.Model.getParameterQuantities`"""
            parameter = self.getParameters(names=str(name))
            quantity = parameter.getQuantity()
            return quantity

        return recursiveMethod(
            base_method=get,
            args=names,
            valid_input_types=str,
            output_type=dict,
            default_args=self.getParameterNames()
        )

    def getFunctions(
            self,
            names: Union[str, List[str]] = None,
            filter_type: Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]] = None
    ) -> Union[Function, List[Function]]:
        """
        Get functions stored in model.

        __Recursion Base__
            return single function if compatible with filter type: names [None]

        :param self: :class:`~Function.Model` to retrieve function(s) from
        :param names: name(s) of function(s) to retrieve
        :param filter_type: only retrieve function(s) of this class, acts as a filter
        """
        if isinstance(names, str):
            function_object = self.functions[names]
            if filter_type is None or isinstance(function_object, filter_type):
                return function_object
            else:
                raise TypeError(f"names input must correspond to {filter_type:s}")
        elif isinstance(names, list):
            functions = [self.getFunctions(names=name) for name in names]
            if filter_type is None:
                return functions
            else:
                return [function_object for function_object in functions if isinstance(function_object, filter_type)]
        elif names is None:
            return self.getFunctions(names=self.getFunctionNames(), filter_type=filter_type)
        else:
            raise RecursiveTypeError(names)

    def loadFunctionsFromFiles(self, filepaths: Union[str, List[str]]) -> None:
        """
        Add functions to model by parsing through YML file.

        :param self: :class:`~Function.Model` to add function(s) to
        :param filepaths: name(s) of config file(s) to retrieve function info from
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        for filepath in filepaths:
            self.addPaperQuantities(generateFunctionsFromFile(filepath, model=self))

    def loadParametersFromFiles(self, filepaths: Union[str, List[str]]) -> None:
        """
        Add parameters to model by parsing through YML file.

        :param self: :class:`~Function.Model` to add function(s) to
        :param filepaths: name(s) of config file(s) to retrieve parameter info from
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        for filepath in filepaths:
            self.addPaperQuantities(generateParametersFromFile(filepath, model=self))

    def saveQuantitiesToFile(self, filepath: str, specie: str) -> TextIO:
        """
        Save quantity object stored in model into YML file for future retrieval.

        :param self: :class:`~Function.Model` to retrieve parameters from
        :param filepath: path of file to save parameters into
        :param specie: specie of quantity object to save info for.
            Must be "Parameter" or "Function".
        :returns: new file
        """
        if specie == "Parameter":
            quantity_objects: Iterable[Parameter] = self.getParameters()
        elif specie == "Function":
            quantity_objects: Iterable[Function] = self.getFunctions()
        else:
            raise ValueError("invalid value for specie")

        save_contents = {}
        for quantity_object in quantity_objects:
            name = quantity_object.getName()
            filestem = quantity_object.getStem()
            if isinstance(filestem, str):
                if filestem not in save_contents.keys():
                    save_contents[filestem] = []
                save_contents[filestem].append(name)
            else:
                save_contents[name] = quantity_object.getSaveContents()

        file = saveConfig(save_contents, filepath)
        return file

    def saveFunctionsToFile(self, filepath: str) -> TextIO:
        """
        Save functions stored in model into file.
        Accepted formats are various markup, *.tex, *.pdf (with *.tex).

        :param self: :class:`~Function.Model` to retrieve functions from
        :returns: new file
        """
        function_objs: List[Function] = self.getFunctions()

        file_extension = Path(filepath).suffix
        if file_extension in config_file_extensions:
            save_contents = {}
            for function_obj in function_objs:
                name = function_obj.getName()
                filestem = function_obj.getStem()
                if isinstance(filestem, str):
                    try:
                        save_contents[filestem].append(name)
                    except KeyError:
                        save_contents[filestem] = [name]
                else:
                    save_contents[name] = function_obj.getSaveContents()

            file = saveConfig(save_contents, filepath)
            return file
        elif file_extension == ".tex":
            pre_mode = SymbolicVariables.mode
            SymbolicVariables.switchMode("tex")

            save_lines = [
                r"\documentclass{article}",
                r"\usepackage{amsmath, amssymb}",
                r"\begin{document}",
            ]

            for function_obj in function_objs:
                symbol = function_obj.getSymbol()
                symbol_tex = latex(symbol)
                expression = function_obj.getTexExpression(expanded=False)
                expression_tex = latex(expression)
                equation_tex = r"\begin{equation}" \
                    + f"{symbol_tex:s} = {expression_tex:s}" \
                    + r"\end{equation}"
                filestem = function_obj.getStem()
                save_lines.append(equation_tex)
            save_lines.append(r"\end{document}")

            SymbolicVariables.switchMode(pre_mode)

            with open(filepath, 'w') as file:
                for line in save_lines:
                    file.writelines(line)
                    file.write('\n')

            return file
        elif file_extension == ".pdf":
            tex_filepath = filepath.replace(".pdf", ".tex")
            save_directory = dirname(filepath)
            self.saveFunctionsToFile(tex_filepath)
            subprocess.run(["latexmk", "-pdf", f"-outdir={save_directory:s}", tex_filepath])

    def saveParametersToFile(self, filepath: str) -> TextIO:
        """
        Save parameters stored in model into file.
        Accepted formats are various markup, *.tex, *.pdf (with *.tex).

        :param self: :class:`~Function.Model` to retrieve parameters from
        :returns: new file
        """
        parameter_objs: List[Parameter] = self.getParameters()

        file_extension = Path(filepath).suffix
        if file_extension in config_file_extensions:
            save_contents = {}
            for parameter_obj in parameter_objs:
                name = parameter_obj.getName()
                filestem = parameter_obj.getStem()
                if isinstance(filestem, str):
                    try:
                        save_contents[filestem].append(name)
                    except KeyError:
                        save_contents[filestem] = [name]
                else:
                    save_contents[name] = parameter_obj.getSaveContents()

            file = saveConfig(save_contents, filepath)
            return file
        elif file_extension == ".tex":
            pre_mode = SymbolicVariables.mode
            SymbolicVariables.switchMode("tex")

            save_lines = [
                r"\documentclass{article}",
                r"\usepackage{amsmath, amssymb}",
                r"\setlength\parindent{0pt}",
                r"\begin{document}",
            ]

            for parameter_obj in parameter_objs:
                symbol = parameter_obj.getSymbol()
                symbol_tex = latex(symbol)
                quantity = parameter_obj.getQuantity()
                value_tex = formatValue(quantity)
                unit_tex = formatUnit(quantity, as_tex=True)

                parameter_tex = f"${symbol_tex:s}$"
                parameter_tex += f" = {value_tex:s}"
                if len(unit_tex) >= 1:
                    parameter_tex += f" ${unit_tex:s}$"
                parameter_tex += r' \\'

                filestem = parameter_obj.getStem()
                save_lines.append(parameter_tex)
            save_lines.append(r"\end{document}")

            SymbolicVariables.switchMode(pre_mode)

            with open(filepath, 'w') as file:
                for line in save_lines:
                    file.writelines(line)
                    file.write('\n')

            return file
        elif file_extension == ".pdf":
            tex_filepath = filepath.replace(".pdf", ".tex")
            save_directory = dirname(filepath)
            self.saveParametersToFile(tex_filepath)
            subprocess.run(["latexmk", "-pdf", f"-outdir={save_directory:s}", tex_filepath])

    def saveVariablesToFile(self, filepath: str) -> TextIO:
        """
        Save variables stored in model for future retrieval.

        :param self: :class:`~Function.Model` to retrieve variables from
        :param filepath: path of file to save variables into
        :returns: new file
        """
        variable_objs: List[Variable] = self.getVariables()

        file_extension = Path(filepath).suffix
        if file_extension in config_file_extensions:
            save_contents = {
                variable_obj.getName(): variable_obj.getSaveContents()
                for variable_obj in variable_objs
            }
            file = saveConfig(save_contents, filepath)
            return file
        elif file_extension == ".tex":
            pre_mode = SymbolicVariables.mode
            SymbolicVariables.switchMode("tex")

            save_lines = [
                r"\documentclass{article}",
                r"\usepackage{amsmath, amssymb}",
                r"\setlength\parindent{0pt}",
                r"\begin{document}",
            ]

            for variable_obj in variable_objs:
                symbol = variable_obj.getSymbol()
                symbol_tex = latex(symbol)

                time_evolution_type = variable_obj.getTimeEvolutionType()
                if time_evolution_type in ["Temporal", "Constant"]:
                    initial_condition = variable_obj.getInitialCondition()
                    initial_condition_tex = formatValue(initial_condition)
                    variable_tex = f"${symbol_tex:s}(0) = {initial_condition_tex:s}$"
                elif time_evolution_type == "Equilibrium":
                    variable_tex = f"${symbol_tex:s}$"
                elif time_evolution_type == "Function":
                    variable_name = variable_obj.getName()
                    model = variable_obj.getModel()
                    function_obj = model.getFunctions(names=variable_name)
                    expression = function_obj.getExpression(
                        expanded=False,
                        substitute_dependents=False
                    )
                    expression_tex = latex(expression)
                    variable_tex = f"${symbol_tex:s} = {expression_tex:}$"

                variable_tex += f" ({time_evolution_type:s})"
                variable_tex += r' \\'

                filestem = variable_obj.getStem()
                save_lines.append(variable_tex)
            save_lines.append(r"\end{document}")

            SymbolicVariables.switchMode(pre_mode)

            with open(filepath, 'w') as file:
                for line in save_lines:
                    file.writelines(line)
                    file.write('\n')

            return file
        elif file_extension == ".pdf":
            tex_filepath = filepath.replace(".pdf", ".tex")
            save_directory = dirname(filepath)
            self.saveVariablesToFile(tex_filepath)
            subprocess.run(["latexmk", "-pdf", f"-outdir={save_directory:s}", tex_filepath])

    def getDerivatives(self) -> List[Derivative]:
        """
        Get stored derivatives of given time-evolution type(s).

        __Recursion Base__
            get derivatives of single time-evolution type: time_evolution_types [str]
            get all derivatives: time_evolution_types [None]

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        """
        derivative_objs = self.getFunctions(filter_type=Derivative)
        return derivative_objs

    def getEquilibriumSolutions(
            self,
            names: Union[str, List[str]] = None,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_constants: bool = True,
            substitute_functions: bool = True
    ) -> Dict[Symbol, Expr]:
        """
        Get equilibria solutions from derivatives, to substitute into variables.

        :param self: :class:`~Function.Model` to solve for equilibrium solution(s) in
        :param names: name(s) of function(s) to solve for simulatenous equilibrium of
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getEquilibriumSolutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding expressions for function-type derivatives.
            Set False to leave them as symbolic variables.
        :returns: Dictionary of substitutions.
            Key is symbolic variable to replace.
            Value is equilibrium expression to substitute into variable.
        """
        if skip_parameters is None:
            skip_parameters = []

        if names is None:
            equilibrium_variables = self.getVariables(
                time_evolution_types="Equilibrium",
                return_type=Symbol
            )
        elif isinstance(names, (str, list)):
            equilibrium_variables = self.getVariables(
                names=names,
                return_type=Symbol
            )
        else:
            raise RecursiveTypeError(names)

        equilibrium_count = len(equilibrium_variables)
        if equilibrium_count == 0:
            return {}
        elif equilibrium_count >= 1:
            derivative_objs = self.getDerivativesFromVariables(equilibrium_variables)
            equilibrium_expressions = [
                derivative_obj.getExpression(expanded=True)
                for derivative_obj in derivative_objs
            ]

            bk_probs = list(symbols("pC0 pC1 pC2 pO2 pO3"))
            if set(bk_probs).issubset(set(equilibrium_variables)):
                equilibrium_expressions.append(sum(bk_probs) - 1)
            solutions = solve(equilibrium_expressions, equilibrium_variables)

            substitutions = {}
            if substitute_functions:
                function_substitutions = self.getFunctionSubstitutions(
                    substitute_parameters=substitute_parameters,
                    skip_parameters=skip_parameters,
                    substitute_constants=substitute_constants
                )
                substitutions.update(function_substitutions)
            if substitute_parameters:
                parameter_names = []
                for derivative_obj in derivative_objs:
                    new_parameter_names = derivative_obj.getParameters(
                        expanded=True,
                        return_type=str
                    )
                    for parameter_name in new_parameter_names:
                        if (
                            parameter_name not in skip_parameters and
                            parameter_name not in parameter_names
                        ):
                            parameter_names.append(parameter_name)

                parameter_substitutions = self.getParameterSubstitutions(parameter_names)
                substitutions.update(parameter_substitutions)
            if substitute_constants:
                constant_substitutions = self.getConstantSubstitutions()
                substitutions.update(constant_substitutions)

            solutions = {
                variable: solution.subs(substitutions)
                for variable, solution in solutions.items()
            }
            return solutions

    def getEquilibriumFunction(
            self,
            name: str,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_constants: bool = True,
            substitute_functions: bool = True
    ) -> Expr:
        """
        Get equilibrium function corresponding to derivative, to substitute into variable.

        :param self: :class:`~Function.Model` to solve for equilibrium solution in
        :param name: name of variable to retrieve equilibrium expression of
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getEquilibriumSolutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding expressions for function-type derivatives.
            Set False to leave them as symbolic variables.
        """
        equilibria = self.getEquilibriumSolutions(
            substitute_parameters=substitute_parameters,
            skip_parameters=skip_parameters,
            substitute_constants=substitute_constants,
            substitute_functions=substitute_functions
        )
        function_object = self.getDerivativesFromVariables(name)
        equilibrium = equilibria[function_object.getVariable()]
        return equilibrium

    def getFunctionSubstitutions(
            self,
            names: Union[str, List[str]] = None,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_constants: bool = True
    ) -> Dict[Symbol, Expr]:
        """
        Get functions to substitute into variables.

        :param self: :class:`~Function.Model` to retrieve functions from
        :param names: name(s) of variable(s) to retrieve function for
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set False to leave them as symbolic variables.
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getFunctionSubstitutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        """
        if skip_parameters is None:
            skip_parameters = []
        if names is None:
            names = self.getVariables(time_evolution_types="Function", return_type=str)

        variable_count = len(names)
        if variable_count == 0:
            return {}
        elif variable_count >= 1:
            functions = self.getFunctions(names=names)
            getExpression = partial(Function.getExpression, expanded=True)

            # noinspection PyTypeChecker
            expressions: List[Expr] = list(map(getExpression, functions))
            variables = list(map(Symbol, names))

            substitutions = {}
            if substitute_parameters:
                parameter_names = unique(
                    [
                        parameter_name
                        for function_object in functions
                        for parameter_name in
                        function_object.getParameters(expanded=True, return_type=str)
                        if parameter_name not in skip_parameters
                    ]
                )
                substitutions.update(self.getParameterSubstitutions(parameter_names))
            if substitute_constants:
                substitutions.update(self.getConstantSubstitutions())

            expressions = [expression.subs(substitutions) for expression in expressions]
            function_substitutions = {variables[i]: expressions[i] for i in range(variable_count)}
            return function_substitutions

    def getConstantSubstitutions(self, names: Union[str, List[str]] = None) -> Dict[Symbol, Expr]:
        """
        Get constants to substitute into variables.

        :param self: :class:`~Function.Model` to retrieve variable(s) from
        :param names: name(s) of variable(s) to substitute numerical constants in for
        """
        if names is None:
            names = self.getVariables(
                time_evolution_types="Constant",
                return_type=str
            )

        constant_count = len(names)
        if constant_count == 0:
            return {}
        elif constant_count >= 1:
            variables = self.getVariables(names=names)
            substitutions = {
                variable.getSymbol(): variable.getInitialCondition()
                for variable in variables
            }
            return substitutions

    def getParameterSubstitutions(
            self, names: Union[str, List[str]] = None, skip_parameters: Union[str, List[str]] = None
    ) -> Dict[Symbol, float]:
        """
        Get parameter values to substitute into parameters.

        :param names: name(s) of parameter to include in substitutions.
            Defaults to all parameters in model.
        :param skip_parameters: name(s) of parameter(s) to skip substitution for
        """
        if skip_parameters is None:
            skip_parameters = []
        quantities = self.getParameterQuantities(names=names)
        if names is None:
            names = quantities.keys()

        substitutions = {
            Symbol(parameter_name): quantities[parameter_name].to_base_units().magnitude
            for parameter_name in names
            if parameter_name not in skip_parameters
        }
        return substitutions

    def getDerivativeVector(
            self,
            names: List[str] = None,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_equilibria: bool = True,
            substitute_constants: bool = True,
            substitute_functions: bool = True,
            lambdified: bool = False
    ) -> Tuple[Union[List[Expr], function], Optional[Dict[Symbol, Expr]]]:
        """
        Get derivative vector corresponding to derivatives in :class:`~Function.Model`

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        :param names: name(s) of variable(s) ordered in same order derivatives will be returned
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getDerivativeVector.substitute_parameters` is set True.
        :param substitute_equilibria: set True to substitute equilibrium expressions in for variables in equilibrium.
            Set False to leave them as symbolic variables.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding expressions for function-type derivatives.
            Set False to leave them as symbolic variables.
        :param lambdified: set True to return derivative vector as lambda function handle.
            Set False to return symbolic derivative vector
        :returns: vector of derivatives.
            Uses derivatives in :class:`~Function.Model` to determine derivatives.
            Returned as list of symbolic expressions
            if :paramref:`~Function.Model.getDerivativeVector.lambdified` is set to True.
            Returned as lambda function handle
            if :paramref:`~Function.Model.getDerivativeVector.lambdified` is set to False.
        """
        if names is None:
            names = self.getVariableNames()
        if skip_parameters is None:
            skip_parameters = []

        use_memory = Function.usingMemory()
        Function.clearMemory()
        Function.setUseMemory(False)

        variable_substitutions = {}
        if substitute_equilibria:
            equilibrium_solutions = self.getEquilibriumSolutions(
                substitute_parameters=substitute_parameters,
                substitute_constants=substitute_constants,
                skip_parameters=skip_parameters
            )
            variable_substitutions.update(equilibrium_solutions)
        if substitute_constants:
            constant_substitutions = self.getConstantSubstitutions()
            variable_substitutions.update(constant_substitutions)
        if substitute_functions:
            function_substitutions = self.getFunctionSubstitutions(
                substitute_parameters=substitute_parameters,
                substitute_constants=substitute_constants,
                skip_parameters=skip_parameters
            )
            variable_substitutions.update(function_substitutions)

        parameter_substitutions = self.getParameterSubstitutions(skip_parameters=skip_parameters) if substitute_parameters else {}
        temporal_derivatives = self.getDerivativesFromVariables(names)

        derivative_vector = []
        for derivative in temporal_derivatives:
            derivative: Union[Derivative, Function]
            expression = derivative.getExpression(expanded=True)

            derivative_variables = derivative.getVariables(expanded=True)
            derivative_functions = derivative.getFunctions(expanded=True)

            variable_substitution = {
                variable: substitution
                for variable, substitution in variable_substitutions.items()
                if variable in derivative_variables + derivative_functions
            }
            derivative_parameters = derivative.getParameters(expanded=True)
            parameter_substitution = {
                parameter: value
                for parameter, value in parameter_substitutions.items()
                if parameter in derivative_parameters
            }
            expression = expression.subs({**variable_substitution, **parameter_substitution})

            derivative_vector.append(expression)

        Function.setUseMemory(use_memory)

        print("derivative vector:", derivative_vector)
        if lambdified:
            derivative_vector = lambdify((Symbol('t'), tuple(names)), derivative_vector, modules=["math"])
        if substitute_equilibria:
            return derivative_vector, equilibrium_solutions
        else:
            return derivative_vector

    def getInitialValues(
            self,
            names: Union[str, List[str]] = None,
            return_type: Type[Union[dict, list, ndarray]] = dict,
            initial_values: Dict[Symbol, float] = None
    ) -> Union[float, List[float], ndarray, dict]:
        """
        Get initial values for variables in model.

        :param self: :class:`~Function.Model` to retrieve derivatives from
        :param names: name(s) of variable(s) to retrieve values for
        :param return_type: class type for output.
            Must be dict, list, or ndarray.
            Only called if names is list.
        :param initial_values: dictionary of initial values if already known.
            Key is symbol for variable.
            Value is float for initial value.
        :returns: Initial values for desired variables.
            Dictionary of initial values if return_type is dict
                Key is symbol for variable
                Value is float for initial value.
            List of floats if return_type is list.
            ndarray of float if return_type is ndarray.
        """
        if initial_values is None:
            variable_objs = self.getVariables(names=names)
            if len(names) == 1:
                variable_objs = [variable_objs]

            initial_values = {
                variable_obj.getSymbol(): variable_obj.getInitialCondition()
                for variable_obj in variable_objs
            }

        if isinstance(names, str):
            symbol = SymbolicVariables.getVariables(names=names)
            return initial_values[symbol]
        elif isinstance(names, Symbol):
            return initial_values[names]
        elif isinstance(names, list):
            def initial_value(name): return self.getInitialValues(names=name, initial_values=initial_values)
            if return_type == list:
                return [initial_value(name) for name in names]
            elif return_type == ndarray:
                return np.array([initial_value(name) for name in names])
            elif return_type == dict:
                return {SymbolicVariables.getVariables(names=name): initial_value(name) for name in names}
            else:
                raise ValueError("return_type must be list, ndarray, or dict")
        else:
            raise RecursiveTypeError(names, [str, Symbol])

    def getVariables(
            self,
            names: Union[str, List[str]] = None,
            time_evolution_types: Union[str, List[str]] = None,
            return_type: Type[Union[Variable, Symbol, str]] = Variable
    ) -> Union[List[Variable], List[Symbol], List[str]]:
        """
        Get variables stored in model.

        __Recursion Base__
            get symbolic variable associated with single derivative: names [str]

        :param self: :class:`~Function.Model` to retrieve derivative variable(s) from
        :param names: name(s) of variable(s) to retrieve.
            Defaults to all variables.
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        :param return_type: class type to return elements in list output as
        """
        if names is not None and time_evolution_types is not None:
            raise ValueError("atleast one of names or time_evolution_types must be None")
        elif time_evolution_types is not None:
            if isinstance(time_evolution_types, str):
                variables = [
                    variable
                    for variable in self.getVariables()
                    if variable.getTimeEvolutionType() == time_evolution_types
                ]
            elif isinstance(time_evolution_types, list):
                variables = [
                    variable
                    for time_evolution_type in time_evolution_types
                    for variable in self.getVariables(time_evolution_types=time_evolution_type)
                ]
            else:
                raise RecursiveTypeError(time_evolution_types)
        else:
            variable_objs = self.variables

            def get(name: str) -> Variable:
                """Base method for :meth:`~Function.Model.getVariables`"""
                return variable_objs[name]

            variables = recursiveMethod(
                base_method=get,
                args=names,
                valid_input_types=str,
                output_type=list,
                default_args=self.getVariableNames()
            )

        if return_type == Symbol:
            variable_symbols = list(map(Variable.getSymbol, variables))
            return variable_symbols
        elif return_type == str:
            variable_names = list(map(Variable.getName, variables))
            return variable_names
        elif return_type == Variable:
            return variables
        else:
            raise ValueError("return_type must be of type Variable, Symbol, str")

    def getDerivativesFromVariables(
            self,
            variables: Union[str, Symbol, Variable, List[Union[str, Symbol, Variable]]]
    ) -> Union[Derivative, List[Derivative]]:
        """
        Get derivative corresponding to variable.

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        :param names: (name(s) of) variable(s)) associated with derivative(s)
        """
        derivative_objs = self.getDerivatives()

        def get(variable: Union[str, Symbol, Variable]) -> Derivative:
            """Base method for :meth:`~Function.Model.getDerivativesFromVariables`"""
            if isinstance(variable, str):
                name = variable
            elif isinstance(variable, Symbol):
                name = str(variable)
            elif isinstance(variable, Variable):
                name = variable.getName()
            else:
                raise TypeError(f"variable must be of type str, Symbol, or Variable")

            for derivative_obj in derivative_objs:
                derivative_variable_name = derivative_obj.getVariable(return_type=str)
                if derivative_variable_name == name:
                    return derivative_obj

        return recursiveMethod(
            base_method=get,
            args=variables,
            valid_input_types=(str, Symbol, Variable),
            output_type=list
        )

    def getFunction2ArgumentGraph(self) -> Graph:
        """
        Generate directional graph from (1) derivative variable to (2) variables in derivative.

        :param self: :class:`~Function.Model` to generate model, directional graph for
        :returns: Generated graph
        """        
        variable_objs = self.getVariables()
        var2vars = {}
        for variable_obj_from in variable_objs:
            time_evolution_type = variable_obj_from.getTimeEvolutionType()
            variable_name_from = variable_obj_from.getName()

            if time_evolution_type == "Temporal":
                derivative_obj_from = self.getDerivativesFromVariables(variable_name_from)
                variable_names_to = derivative_obj_from.getVariables(
                    expanded=True, 
                    return_type=str
                )
            elif time_evolution_type == "Equilibrium":
                derivative_obj_from = self.getDerivativesFromVariables(variable_name_from)
                variable_names_to = derivative_obj_from.getVariables(
                    expanded=True, 
                    return_type=str
                )
                variable_names_to.remove(variable_name_from)
            elif time_evolution_type == "Function":
                function_obj_from = self.getFunctions(names=variable_name_from)
                variable_names_to = function_obj_from.getVariables(
                    expanded=True, 
                    return_type=str
                )
            elif time_evolution_type == "Constant":
                variable_names_to = []

            variable_objs_to = self.getVariables(names=variable_names_to)
            var2vars[variable_obj_from] = variable_objs_to

        variable_objs_from = sorted(
            var2vars.keys(), 
            key=lambda k: len(var2vars[k])
        )
        variable_count = len(variable_objs_from)

        graph = Graph(
            n=variable_count,
            directed=True
        )
        # colors = Color(color1).range_to(Color(color2), len(var2vars.keys()))
        # vertex2color = [color.rgb for color in colors]
        evolution2color = {
            "Temporal": "red",
            "Equilibrium": "orange",
            "Function": "green",
            "Constant": "violet"
        }
        
        variable_names = list(map(Variable.getName, variable_objs_from))
        for variable_index_from in range(variable_count):
            variable_from = variable_objs_from[variable_index_from]
            variable_name_from = variable_from.getName()
            time_evolution_type_from = variable_from.getTimeEvolutionType()

            color_from = evolution2color[time_evolution_type_from]
            graph.vs[variable_index_from]["name"] = variable_name_from
            graph.vs[variable_index_from]["color"] = color_from

            variable_objs_to = var2vars[variable_from]
            for variable_obj_to in variable_objs_to:
                variable_name_to = variable_obj_to.getName()
                variable_index_to = variable_names.index(variable_name_to)

                graph.add_edges([(variable_index_from, variable_index_to)])
                graph.es[-1]["color"] = color_from
        
        graph.vs["label"] = graph.vs["name"]
        
        return graph

class Parent:
    """
    Stores properties for Function qua parent.

    :ivar instance_arguments: 2-level dictionary of arguments for children functions.
        Firsy key is name of child function.
        Second key is specie of instance arguments.
        Value is list of string arguments to substitute into child.
        Only used for :class:`~Function.Dependent` children.
    """

    def __init__(self, children: Dict[str, Dict[str, Union[str, List[str]]]]) -> None:
        """
        Constructor for :class:`~Function.Parent`.

        :param children: 2-level dictionary of info for children function.
            First key is name of child.
            Second key is specie of instance arguments for child.
            Value is strings for instance arguments.
        """
        assert isinstance(children, dict) or children is None

        self.instance_arguments = {}

        if children is not None:
            if isinstance(children, Child):
                children = [children]
            self.addChildren(children)

    def addChildren(
        self,
        children: Dict[str, Dict[str, Union[str, List[str]]]]
    ) -> Dict[str, Dict[str, List[Symbol]]]:
        """
        Add info to reference children functions of parent.

        :param self: :class:`~Function.Parent` to add info into
        :param children: dictionary of info to reference children.
            Key is name of child.
            Value is dictionary to pass into :paramref:`~Function.Parent.addChild.arguments`
        :returns: Dictionary of info to reference children.
            Key is name of child.
            Value is dictionary of info output from :meth:`~Function.Parent.addChild`
        """
        new_children = []
        for child_name, arguments in children.items():
            new_child = self.addChild(child_name, arguments)
            new_children.append(new_child)
        return new_children

    def addChild(
        self,
        name: str,
        arguments: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Add info to reference child function of parent.

        :param self: :class:`~Function.Parent` to add info into
        :param name: name of child to add into parent
        :param arguments: dictionary of info to reference child.
            Key is species of instance argument.
            Value is symbol(s) for this species of argument.
        :returns: Dictionary of info to reference child.
            Key is species of instance argument from parent.
            Value is symbol(s) for this species of argument.
        """
        assert isinstance(name, str)

        new_child = {}
        argument_species = arguments.keys()
        for argument_specie in argument_species:
            instance_argument = arguments[argument_specie]
            if isinstance(instance_argument, str):
                new_child[argument_specie] = [instance_argument]
            elif isinstance(instance_argument, list):
                for argument in instance_argument:
                    assert isinstance(argument, str)
                new_child[argument_specie] = instance_argument
            else:
                raise RecursiveTypeError(instance_argument)

        self.instance_arguments[name] = new_child
        return new_child

    def getInstanceArgumentSpecies(self, name: str) -> List[str]:
        """
        Get all possible species of instance arguments.

        :param self: parent to retrieve instance-argument species from
        :param name: name of child to retrieve instance-argument species for
        """
        return self.instance_arguments[name].keys()

    def getInstanceArguments(
        self,
        name: str = None,
        specie: str = None
    ) -> Union[str, List[str]]:
        """
        Get string instance arguments of given species for function.

        __Recursion Base__
            return all instance arguments of given specie: specie [None] and names [None]

        :param self: parent to retrieve instance arguments from
        :param specie: name of instance-argument species to retrieve from parent, acts as an optional filter
        :param name: name of child function to retrieve instance arguments for
        """
        if isinstance(name, str):
            if isinstance(specie, str):
                instance_argument_names = self.instance_arguments[name][specie]
                return instance_argument_names
            elif specie is None:
                return self.instance_arguments[name]
        elif name is None:
            if isinstance(specie, str):
                return {key: self.instance_arguments[key][specie] for key in self.instance_arguments.keys()}
            elif specie is None:
                return self.instance_arguments
        else:
            raise TypeError("specie must be str")


class Child:
    """
    Stores properties for Function qua child.
    """

    def __init__(self) -> None:
        """
        Constructor for :class:`~Function.Child`.
        """
        pass

    def getParents(self, names: Union[str, List[str]] = None) -> Union[Function, List[Function]]:
        """
        Get parent functions.

        __Recursion Base__
            return single parent: names [str]

        :param self: child to retrieve parent(s) from
        :param names: name(s) of parent(s) to retrieve from child.
            Defaults to all parents.
        """

        self: Function

        def get(name: str) -> Function:
            """Base method for :meth:`~Function.Model.getParents`"""
            parent = self.getModel().getFunctions(names=name)
            return parent

        return recursiveMethod(
            base_method=get,
            args=names,
            valid_input_types=str,
            output_type=list,
            default_args=self.getParentNames()
        )

    def getParentNames(self) -> Union[str, List[str]]:
        """
        Get names of parents.

        :param self: child to retrieve parent name(s) from
        """
        self: Function
        name = self.getName()
        parents = [
            function_object.getName()
            for function_object in self.getModel().getFunctions()
            if name in function_object.getFunctions(return_type=str)
        ]
        return parents


class Function(Child, Parent, PaperQuantity):
    """
    Stores pertinent properties relevant to all types of functions.

    :ivar name: name of function
    :ivar expression: symbolic expression stored in memory
    :ivar model: model that function is part of
    :ivar is_parameter: True is function is equal to parameter.
        False otherwise.
    """

    use_memory = False
    functions = []

    def __init__(
            self,
            name: str,
            properties: List[str],
            children: Dict[str, Dict[str, Union[Symbol, List[Symbol]]]] = None,
            model: Model = None,
            filestem: str = '',
            **kwargs
    ) -> None:
        """
        Constructor for :class:`~Function.Function`.

        :param properties: properties indicating types consistent with function
        :param children: argument to pass into :meth:`~Function.Parent`
        :param kwargs: extraneous arguments
        """
        Function.functions.append(self)

        Parent.__init__(self, children)
        Child.__init__(self)
        PaperQuantity.__init__(self, name, model=model, filestem=filestem)

        self.memory = {
            "expression": Expr
        }
        self.is_parameter = "Parameter" in properties

    @staticmethod
    def setUseMemory(use: bool) -> None:
        """
        Turn memory for all Functions on/off.

        :param use: set True to enable memore.
            Set False to disable memory.
        """
        Function.use_memory = use

    @staticmethod
    def usingMemory() -> bool:
        """
        Determine whether memory is enabled/disabled.

        :returns: True if memory is enabled.
            False if memory is disabled.
        """
        return Function.use_memory

    @staticmethod
    def clearMemory(functions: List[Function] = None) -> None:
        """
        Clear memory for Function objects.

        :param functions: functions to clear memory for.
            Defaults to all Function objects.
        """
        if functions is None:
            functions = Function.functions
        for function_object in functions:
            function_object.clearExpression()

    def clearExpression(self) -> None:
        """
        Clear memory of expression for Function object.

        :param self: :class:`~Function.Function` to clear memory in
        """
        self.memory["expression"] = None

    def isParameter(self):
        """
        Determine whether function is equal to parameter.

        :param self: :class:`~Function.Function` to determine for
        """
        return self.is_parameter

    def getParameters(self, **kwargs) -> Union[List[str], List[Symbol]]:
        """
        Get parameters in expression.

        :param kwargs: arguments to pass into :meth:`~Function.Function.getFreeSymbols`
        """
        return self.getFreeSymbols(species="Parameter", **kwargs)

    def getVariables(self, **kwargs) -> Union[List[str], List[Symbol]]:
        """
        Get variables in expression.

        :param kwargs: arguments to pass into :meth:`~Function.Function.getFreeSymbols`
        """
        return self.getFreeSymbols(species="Variable", **kwargs)

    def getFunctions(self, **kwargs):
        """
        Get functions in expression.

        :param kwargs: arguments to pass into :meth:`~Function.Function.getFreeSymbols`
        """
        return self.getFreeSymbols(species="Function", **kwargs)

    def getFreeSymbols(
            self,
            species: str = None,
            return_type: Type[Union[str, Symbol]] = Symbol,
            **kwargs
    ) -> Union[List[str], List[Symbol]]:
        """
        Get symbols in expression for function.

        :param self: :class:`~Function.Function` to retrieve symbols from
        :param species: species of free symbol to retrieve, acts as filter.
            Can be "Parameter", "Variable", "Function".
            Defaults to all free symbols.
        :param return_type: class type of output.
            Must be either sympy.Symbol or str.
        :param kwargs: additional arguments to pass into
            :meth:`~Function.Piecewise.getExpression` or into :meth:`~Function.NonPiecewise.getExpression`
        """
        expression = self.getExpression(**kwargs)
        free_symbols = list(expression.free_symbols)
        if species is None:
            free_symbol_names = list(map(str, free_symbols))
        else:
            unfiltered_names = list(map(str, free_symbols))

            model = self.getModel()
            if species == "Parameter":
                model_symbol_names = model.getParameterNames()
            elif species == "Variable":
                model_symbol_names = model.getVariables(return_type=str)
            elif species == "Function":
                model_variable_names = model.getVariables(return_type=str)
                model_symbol_names = [
                    function_name
                    for function_name in model.getFunctionNames()
                    if function_name not in model_variable_names
                ]
            else:
                raise ValueError(f"invalid species type {species:s}")

            free_symbol_names = [
                free_symbol_name
                for free_symbol_name in unfiltered_names
                if free_symbol_name in model_symbol_names
            ]

        if return_type == str:
            return free_symbol_names
        elif return_type == Symbol:
            return list(map(Symbol, free_symbol_names))
        else:
            raise ValueError("return_type must be Symbol or str")

    def getTexExpression(self, **kwargs):
        pre_mode = SymbolicVariables.mode
        SymbolicVariables.switchMode("tex")

        expression = self.getExpression(**kwargs)
        SymbolicVariables.switchMode(pre_mode)
        return expression

    def getExpression(self, **kwargs):
        """
        Get expression for function, depends on inherited classes.

        :param self: :class:`~Function.Function` to retrieve expression from
        :param kwargs: additional arguments to pass into
            :meth:`~Function.Piecewise.getExpression` or :meth:`~Function.NonPiecewise.getExpression`
        """
        expression_memory = self.memory["expression"]
        if not self.usingMemory() or expression_memory is None or isinstance(self, Dependent):
            if isinstance(self, Piecewise):
                expression = Piecewise.getExpression(self, **kwargs)
            elif isinstance(self, NonPiecewise):
                expression = NonPiecewise.getExpression(self, **kwargs)
            else:
                self: Function
                raise TypeError(f"function {self.getName():s} must inherit either Piecewise or NonPiecewise")
            self: Function
            self.memory["expression"] = expression
            return expression
        elif self.usingMemory() and isinstance(expression_memory, Expr):
            return expression_memory
        else:
            raise ValueError(f"improper expression in memory for {self.getName():s}")


class Derivative:
    """
    Stores properties for function qua derivative.

    :ivar variable: variable that derivative is derivative of
    :ivar time_evolution_type: time-evolution type of derivative (e.g. "Temporal", "Equilibrium", "Constant")
    :ivar initial_condition: initial condition value for associated variable
    """

    def __init__(
        self,
        variable_name: str,
        initial_condition: float = 0
    ) -> None:
        """
         Constructor for :class:`~Function.Derivative`.

        :param variable: variable that derivative is a derivative of
        :param initial_condition: initial value of associated variable
        """
        assert isinstance(variable_name, str)
        self.variable_name = variable_name

    def getVariable(self, return_type: Type[Union[Variable, Symbol, str]] = Variable) -> Union[Variable, Symbol, str]:
        """
        Get variable that derivative is derivative of.

        :param self: :class:`~Function.Derivative` to retreive variable from
        :param return_type: class type of output.
            Must be either sympy.Symbol or str.
        """
        variable_name = self.variable_name

        if return_type == Variable:
            model: Model = self.getModel()
            variable_obj = model.getVariables(names=variable_name)
            return variable_obj
        elif return_type == Symbol:
            variable_obj = self.getVariable(return_type=Variable)
            return variable_obj.getSymbol()
        elif return_type == str:
            return variable_name
        else:
            raise ValueError("return_type must be sp.Symbol or str")


class Dependent:
    """
    Stores properties for a function that requires input from another function.

    :ivar general_arguments: dictionary of arguments to be substituted from another function.
        Key is specie of arguments.
        Value is list of strings for arguments of specie.
    """

    def __init__(self, arguments: Union[str, List[str]]) -> None:
        """
        Constructor for :class:`~Function.Dependent`.

        :param arguments: general arguments to store in function
        """
        self.general_arguments = {}
        self.setGeneralArguments(arguments)

    def setGeneralArguments(
            self,
            arguments: Union[Union[str, List[str]], Dict[str, Union[Symbol, List[Symbol]]]],
            specie: str = None
    ) -> None:
        """
        Format stored general arguments.

        __Recursion Base__
            Set attribute for single species: arguments [sympy.Symbol, list of sympy.Symbol] and species [str]

        :param self: :class:`~Function.Dependent` to set general arguments for
        :param arguments: Dictionary of argument symbol(s)
            if :paramref:`~Function.Dependent.setGeneralArguments.arguments` is None.
            Key is specie of arguments.
            Value is symbol(s) for argument(s).
            Symbol(s) for argument(s)
            if :paramref:`~Function.Dependent.setGeneralArguments.arguments` is None.
        :param specie: name of specie for arguments
            if :paramref:`~Function.Dependent.setGeneralArguments.arguments` is Symbol or list.
            None
            if :paramref:`~Function.Dependent.setGeneralArguments.arguments` is dictionary.
        """
        if isinstance(specie, str):
            if isinstance(arguments, str):
                self.general_arguments[specie] = [arguments]
            elif isinstance(arguments, list):
                for argument in arguments:
                    assert isinstance(argument, str)
                self.general_arguments[specie] = arguments
            elif arguments is None:
                self.general_arguments[specie] = []
            else:
                raise RecursiveTypeError(arguments, str)
        elif specie is None:
            if isinstance(arguments, dict):
                for specie, argument in arguments.items():
                    self.setGeneralArguments(argument, specie)
            else:
                raise TypeError("arguments input must be dict when species input is None")
        else:
            raise TypeError("species input must be str")

    def getGeneralSpecies(self, nested: bool = True) -> List[str]:
        """
        Get species of general arguments for function.

        :param self: :class:`~Function.Dependent` to retrieve species from
        :param nested: set True to include species from children (implicit).
            Set False to only include species from self (explicit).
        """
        species = list(self.general_arguments.keys())
        if nested:
            model = self.getModel()
            for child_name in self.getFunctions(return_type=str):
                child = model.getFunctions(names=child_name)
                if isinstance(child, Dependent):
                    child_species = child.getGeneralSpecies()
                    if isinstance(child_species, str):
                        species.append(child_species)
                    elif isinstance(child_species, list):
                        species.extend(child_species)
                    elif isinstance(child_species, KeysView):
                        species.extend(list(child_species))
                    else:
                        raise TypeError(f"species for {child.getName():s} must be str, list, or dict_keys")
        return unique(species)

    def getGeneralArguments(
        self,
        species: Union[str, List[str]] = None,
        nested: bool = False
    ) -> Union[List[Symbol], Dict[str, List[Symbol]]]:
        """
        Get general arguments of function.

        __Recursion Base__
            return arguments of single species: species [str] and nested [False]

        :param self: :class:`~Function.Dependent` to retrieve arguments from
        :param species: specie(s) of arguments to retrieve, acts as an optional filter
        :param nested: set True to include arguments from children (implicit).
            Set False to only include arguments from self (explicit).

        :returns: dictionary of arguments if :paramref:`~Function.Dependent.getGeneralArguments.species` is list.
            Key is specie of argument.
            Value is symbols for argument of specie.
            Symbols for argument
            if :paramref:`~Function.Dependent.getGeneralArguments.species` is str.
        """
        if nested:
            model = self.getModel()
            general_arguments = self.getGeneralArguments(species=species, nested=False)
            self: Function
            for child_name in self.getFunctions(return_type=str):
                child = model.getFunctions(names=child_name)
                child_general_arguments = child.getGeneralArguments(species=species, nested=True)
                if isinstance(child_general_arguments, Symbol):
                    general_arguments.append(child_general_arguments)
                elif isinstance(child_general_arguments, list):
                    general_arguments.extend(child_general_arguments)
                else:
                    raise RecursiveTypeError(child_general_arguments)
            return general_arguments
        elif not nested:
            if isinstance(species, str):
                general_species = self.getGeneralSpecies(nested=False)
                if species in general_species:
                    specie_argument_names = self.general_arguments[species]
                    specie_argument_symbolics = SymbolicVariables.getVariables(names=specie_argument_names)
                    return specie_argument_symbolics
                elif species not in general_species:
                    return []
            elif isinstance(species, list):
                return {specie: self.getGeneralArguments(specie) for specie in species}
            elif species is None:
                return {specie: self.getGeneralArguments(specie) for specie in self.getGeneralSpecies()}
            else:
                raise RecursiveTypeError(species)

    def getInstanceArgumentExpressions(self, parent: Function, specie: str) -> List[Expr]:
        """
        Get expressions for instance arguments.

        :param self: :class:`~Function.Dependent` to retrieve general arguments from
        :param parent: :class:`~Function.Function` to retrieve instance arguments from
        :param specie: specie of arguments to substitute
        """
        self: Function
        self_name = self.getName()
        if specie == "functions":
            instance_names = parent.getInstanceArguments(specie=specie, name=self_name)
            parent_model = parent.getModel()
            instance_functions = []
            for instance_name in instance_names:
                sibling = parent_model.getFunctions(names=instance_name)
                if isinstance(sibling, Dependent):
                    instance_function = sibling.getExpression(parent=self)
                elif isinstance(sibling, Independent):
                    instance_function = sibling.getExpression(expanded=True)
                else:
                    raise TypeError(f"sibling for {self_name:s} must be Function")
                instance_functions.append(instance_function)
            return instance_functions
        else:
            return parent.getInstanceArguments(specie=specie, name=self_name)

    def getInstanceArgumentSubstitutions(self, parent: Function, specie: str) -> Dict[Symbol, Expr]:
        """
        Get instance-argument substitutions, to substitute into general arguments.

        :param self: :class:`~Function.Dependent` to retrieve general arguments from
        :param parent: :class:`~Function.Function` to retrieve instance arguments from
        :param specie: specie of arguments to substitute
        """
        general_arguments = self.getGeneralArguments(species=specie)
        instance_arguments = self.getInstanceArgumentExpressions(parent, specie)
        return dict(zip(general_arguments, instance_arguments))

    def getInstanceArgumentForm(self, parent: Function) -> Expr:
        """
        Get expression for dependent function, with instance arguments substituted into general arguments.

        :param self: :class:`~Function.Dependent` to retrieve general arguments from
        :param parent: :class:`~Function.Function` to retrieve instance arguments from
        """
        self: Function
        expression = Independent.getExpression(self, expanded=True)
        self: Dependent

        species = self.getGeneralSpecies()
        for specie in species:
            substitutions = self.getInstanceArgumentSubstitutions(parent, specie)
            expression = expression.subs(substitutions)

        return expression

    def getExpression(self, parent: Function = None):
        """
        Get symbolic expression for function.

        :param self: :class:`~Function.Dependent` to retrieve expression for
        :param parent: function to retrieve input arguments from
        """
        expression = eval(self.expression)
        if parent is None:
            return expression

        return self.getInstanceArgumentForm(parent)


class Independent:
    """
    Stores properties for a function that does not require input from another function.
    """

    def __init__(self) -> None:
        """
        Constructor for :class:`~Function.Independent`.
        """
        pass

    def getExpression(
        self,
        substitute_dependents: bool = None,
        expanded: bool = True
    ):
        """
        Get symbolic expression for function.

        :param self: :class:`~Function.Independent` to retrieve expression for
        :param substitute_dependents: set True to substitute all dependents into expression.
            Set False to substitute in accordance with :paramref:`~Function.Function.getExpression.expanded`.
            Defaults to :paramref:`~Function.Function.getExpression.expanded`.
        :param expanded: set True to substitute children into expression.
            Set False to retrieve original expression.
        """
        if substitute_dependents is None:
            substitute_dependents = expanded

        expression = eval(self.expression)
        self: Function
        model = self.getModel()
        if not expanded:
            has_model = isinstance(model, Model)
            if substitute_dependents:
                if has_model:
                    pass
                else:
                    warnings.warn(f"cannot substitute dependents without associated model for {self.getName():s}")
                    return expression
            else:
                return expression

        child_function_names = self.getFunctions(return_type=str)
        for child_name in child_function_names:
            child_symbol = SymbolicVariables.getVariables(names=child_name)
            child_obj = model.getFunctions(names=child_name)
            if isinstance(child_obj, Independent):
                if expanded:
                    child_expression = child_obj.getExpression(
                        expanded=True,
                        substitute_dependents=substitute_dependents
                    )
                    expression = expression.subs(child_symbol, child_expression)
            elif isinstance(child_obj, Dependent):
                if substitute_dependents:
                    child_expression = child_obj.getExpression(parent=self)
                    expression = expression.subs(child_symbol, child_expression)

        return expression


class Piecewise:
    """
    Stores info pertaining to piecewise function.

    :ivar pieces: function names constituting piecewise (pieces)
    :ivar conditions: conditions under which to use each piece
    """

    def __init__(self, pieces: List[str], conditions: List[str]) -> None:
        """
        Constructor for :class:`~Function.Piecewise`.

        :param pieces: symbols for function pieces
        :param conditions: conditions corresponding to function pieces
        """
        assert len(pieces) == len(conditions)
        for piece in pieces:
            assert isinstance(piece, str)

        self.pieces = pieces
        self.conditions = conditions

    def getConditions(self) -> List[bool]:
        """
        Get conditions, to determine which function piece to use.

        :param self: :class:`~Function.Piecewise` to retrieve conditions from
        """
        conditions = self.conditions
        symbolic_conditions = list(map(eval, conditions))
        return symbolic_conditions

    def getPieces(
            self, return_type: Type[Union[str, Symbol, Function]] = Symbol
    ) -> Union[List[str], List[Symbol], List[Function]]:
        """
        Get pieces constituting piecewise function.

        :param self: :class:`~Function.Piecewise` to retrieve constitutes from
        :param return_type: type of output to return list elements as.
        :returns: Names of pieces
            if :paramref:`~Function.Piecewise.getPieces.return_type` is str.
            Symbols for pieces
            if :paramref:`~Function.Piecewise.getPieces.return_type` is Symbol.
            Function objects from :class:`~Function.Model`.
            if :paramref:`~Function.Piecewise.getPieces.return_type` is Function.
        """
        pieces = self.pieces
        if return_type == str:
            piece_names = pieces
            return piece_names
        elif return_type == Symbol:
            piece_symbols = SymbolicVariables.getVariables(names=pieces)
            return piece_symbols
        elif return_type == Function:
            self: Function
            model = self.getModel()
            piece_names = self.getPieces(return_type=str)
            functions = model.getFunctions(names=piece_names)
            return functions
        else:
            raise ValueError("invalid return type")

    def getPieceCount(self) -> int:
        """
        Get number of pieces constituting piecewise function.

        :param self: :class:`~Function.Piecewise` to retrieve pieces from
        """
        return len(self.pieces)

    def getExpression(self, expanded: bool = False, **kwargs) -> spPiecewise:
        """
        Get symbolic piecewise expression.

        :param self: :class:`~Function.Piecewise` to retrieve expression for
        :param expanded: see :paramref:`~Function.Function.getExpression.expanded`
        :param kwargs: additional arguments to substitute into :meth:`~Function.Function.getExpression`
        """
        if expanded:
            function_objs: List[Function] = self.getPieces(return_type=Function)
            getExpression = partial(
                Function.getExpression,
                expanded=True,
                **kwargs,
            )
            pieces = list(map(getExpression, function_objs))
        else:
            pieces = self.getPieces(return_type=Symbol)

        conditions = self.getConditions()
        exprconds = [
            (pieces[i], conditions[i])
            for i in range(self.getPieceCount())
        ]
        return spPiecewise(*exprconds)


class NonPiecewise:
    """
    Stores info pertaining to nonpiecewise (standard) function.

    :ivar expression: symbolic expression for function
    """

    def __init__(self, expression: str) -> None:
        """
        Constructor for :class:`~Function.NonPiecewise`.

        :param expression: symbolic expression as str for function
        """
        assert isinstance(expression, str)
        self.expression = expression

    def getExpression(
            self,
            parent: Function = None,
            substitute_dependents: bool = None,
            expanded: Union[int, str] = False
    ) -> Expr:
        """
        Get symbol expression for function.

        :param self: :class:`~Function.Function` to retrieve expression for
        :param parent: function to retrieve input arguments from.
            Only called if self is Dependent function.
        :param substitute_dependents: set True to substitute all dependents into expression.
            Set False to substitute in accordance with :paramref:`~Function.Function.getExpression.expanded`.
            Defaults to :paramref:`~Function.Function.getExpression.expanded`.
        :param expanded: set True to substitute children into expression.
            Set False to retrieve original expression.
        """
        if isinstance(self, Independent):
            return Independent.getExpression(
                self,
                substitute_dependents=substitute_dependents,
                expanded=expanded
            )
        elif isinstance(self, Dependent):
            return Dependent.getExpression(self, parent=parent)


def FunctionMaster(
        name: str,
        expression: Union[Expr, List[Function]],
        inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
        **kwargs
) -> Function:
    """
    Generate function object with desired inheritance and properties.

    :param name: name of function
    :param expression: string expression/pieces for function
    :param inheritance: classes for Function object to inherit
    :param kwargs: arguments to pass into inheritance classes.
        Key is string name of class.
        Value is dictionary of arguments/parameters to pass into class.
    :returns: Generated function object
    """
    self = type("FunctionCore", (Function, *inheritance), {})(name, **kwargs)

    if Derivative in inheritance:
        Derivative.__init__(self, **kwargs["Derivative"])

    if Dependent in inheritance:
        Dependent.__init__(self, **kwargs["Dependent"])
    elif Independent in inheritance:
        Independent.__init__(self)
    else:
        raise ValueError("Function must inherit either Dependent or Independent")

    if Piecewise in inheritance:
        Piecewise.__init__(self, expression, **kwargs["Piecewise"])
    elif NonPiecewise in inheritance:
        NonPiecewise.__init__(self, expression)
    else:
        raise ValueError("Function must inherit either Piecewise of NonPiecewise")

    return self


class SymbolicVariables:
    valid_modes = ["tex", "symbol"]
    mode = "symbol"
    variable_names = []
    tex_variables = {}
    symbolic_variables = {}

    @classmethod
    def addVariables(cls, names: Union[str, List[str]]) -> Union[Symbol, List[Symbol]]:
        variable_names = cls.variable_names

        def add(name: str) -> None:
            assert isinstance(name, str)

            if name not in variable_names:
                try:
                    tex = var2tex[name].replace('$', '')
                except KeyError:
                    tex = name

                symbol_tex = Symbol(tex)
                symbol = Symbol(name)
                cls.variable_names.append(name)
                cls.tex_variables[name] = symbol_tex
                cls.symbolic_variables[name] = symbol

            globals()[name] = cls.getVariables(names=name)
            return globals()[name]

        return recursiveMethod(
            args=names,
            base_method=add,
            valid_input_types=str
        )

    @classmethod
    def getVariables(cls, names: Union[str, List[str]] = None) -> Symbol:
        mode = cls.mode

        if mode == "tex":
            variables = cls.tex_variables
        elif mode == "symbol":
            variables = cls.symbolic_variables

        def get(name: str):
            return variables[name]

        return recursiveMethod(
            args=names,
            base_method=get,
            valid_input_types=str,
            output_type=list,
            default_args=cls.variable_names
        )

    @classmethod
    def switchMode(cls, mode: str = None) -> str:
        if mode is None:
            present_mode = cls.mode
            if present_mode == "tex":
                cls.mode = "symbol"
            elif present_mode == "symbol":
                cls.mode = "tex"
        elif isinstance(mode, str):
            assert mode in SymbolicVariables.valid_modes
            cls.mode = mode

        for variable_name in cls.variable_names:
            globals()[variable_name] = cls.getVariables(names=variable_name)

        return cls.mode


def getFunctionInfo(contents: dict, model: Model = None) -> dict:
    """
    Get formatted dictionary of info to generate Function object.

    :param contents: 2/3-level dictionary of info directly from file.
        First key is name of function.
        Second key is name of property for function.
        Value is string or 1-level dictionary, which indicates property value.
    :param model: model to associated Function object with
    """

    # noinspection PyShadowingNames
    def getVariables(contents: dict) -> List[str]:
        """
        Get names of variables for function.

        :param contents: info for function
        """
        variable_names = contents["variables"]
        if isinstance(variable_names, str):
            variable_names = [variable_names]

        SymbolicVariables.addVariables(names=variable_names)
        return variable_names

    # noinspection PyShadowingNames
    def getParameters(contents: dict) -> List[str]:
        """
        Get names of parameters forfunction.

        :param contents: info for function
        """
        parameter_names = contents["parameters"]
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        SymbolicVariables.addVariables(names=parameter_names)
        return parameter_names

    # noinspection PyShadowingNames
    def getProperties(contents: dict) -> List[str]:
        """
        Get properties to give function (e.g. piecewise, dependent)

        :param contents: info for function
        """
        return contents["properties"]

    # noinspection PyShadowingNames
    def getArguments(contents: dict) -> Dict[str, List[str]]:
        """
        Get general arguments as string for function.

        :param contents: info for function
        """
        info_arguments = contents["arguments"]

        arguments = {}
        for specie, argument_names in info_arguments.items():
            SymbolicVariables.addVariables(names=argument_names)
            arguments[specie] = argument_names

        return arguments

    # noinspection PyShadowingNames
    def getVariableName(contents: dict) -> str:
        """
        Get name of associated variable for derivative.

        :param contents: info for function
        """
        variable_name = contents["variable"]
        SymbolicVariables.addVariables(names=variable_name)
        return variable_name

    # noinspection PyShadowingNames
    def getChildren(contents: dict) -> dict:
        """
        Get info to connect function with child.

        :param contents: info for function
        :returns: 2-level dictionary of instance arguments for function into child.
            First key is name of child function.
            Second key is specie of instance arguments.
            Value is symbols for instance arguments.
        """
        children_contents = contents["children"]
        children_names = list(children_contents.keys())
        SymbolicVariables.addVariables(names=children_names)

        children_dict = {}
        for child_name in children_names:
            if (child_contents := children_contents[child_name]) is not None:
                children_dict[child_name] = {}
                argument_species = child_contents.keys()
                for argument_specie in argument_species:
                    argument_variable_names = child_contents[argument_specie]
                    SymbolicVariables.addVariables(names=argument_variable_names)
                    children_dict[child_name][argument_specie] = argument_variable_names

        return children_dict

    kwargs = {}
    if model is not None:
        kwargs["model"] = model

    contents_keys = contents.keys()
    if "variables" in contents_keys:
        kwargs["variables"] = getVariables(contents)
    if "parameters" in contents_keys:
        kwargs["parameters"] = getParameters(contents)
    if "children" in contents_keys:
        kwargs["children"] = getChildren(contents)

    properties = getProperties(contents)
    kwargs["properties"] = properties
    if "Dependent" in properties:
        kwargs["Dependent"] = {
            "arguments": getArguments(contents)
        }
    if "Derivative" in properties:
        kwargs["Derivative"] = {
            "variable_name": getVariableName(contents)
        }
    if "Piecewise" in properties:
        kwargs["Piecewise"] = {
            "conditions": contents["conditions"]
        }
    return kwargs


def generateFunctionsFromFile(filepath: str, **kwargs) -> List[Function]:
    """
    Generate all functions from file.

    :param filepath: path of file to read functions from
    :param kwargs: additional arguments to pass into :meth:`~Function.generateFunction`
    :returns: Generated functions
    """
    assert isinstance(filepath, str)

    contents = loadConfig(filepath)
    filestem = Path(filepath).stem
    function_names = contents.keys()

    def generateFunctionPartial(name) -> Function:
        return generateFunction(
            name,
            contents[name],
            filestem=filestem,
            **kwargs
        )

    function_objs = list(map(generateFunctionPartial, function_names))
    return function_objs


def generateParametersFromFile(filepath: str, **kwargs) -> List[Parameter]:
    """
    Generate all parameters from file.

    :param filepath: path of file to read parameters from
    :param kwargs: additional arguments to pass into :meth:`~Function.generateParameter`
    :returns: Generated parameters
    """
    assert isinstance(filepath, str)

    contents = loadConfig(filepath)
    filestem = Path(filepath).stem
    parameter_names = contents.keys()

    def generateParameterPartial(name) -> Parameter:
        return generateParameter(
            name,
            contents[name],
            filestem=filestem,
            **kwargs
        )

    parameter_objs = list(map(generateParameterPartial, parameter_names))
    return parameter_objs


def generateVariablesFromFile(
    filepath: str,
    archive: str = None,
    **kwargs
) -> List[Parameter]:
    """
    Generate all variables from file.

    :param filepath: path of file to read variables from
    :param archive: (optional) archive to load variable file from
    :param kwargs: additional arguments to pass into :meth:`~Function.generateVariable`
    :returns: Generated variables
    """
    assert isinstance(filepath, str)

    contents = loadConfig(filepath, archive=archive)
    filestem = Path(filepath).stem
    variable_names = contents.keys()

    def generateVariablePartial(name) -> Variable:
        return generateVariable(
            name,
            contents[name],
            filestem=filestem,
            **kwargs
        )

    variable_objs = list(map(generateVariablePartial, variable_names))
    return variable_objs


def getInheritance(
        properties: List[str], name: str = ''
) -> List[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]]:
    """
    Get classes that function must inherit from properties.

    :param properties: properties to determine inheritance
    :param name: name of function, only called when error is thrown
    """
    inheritance = []

    if "Derivative" in properties:
        inheritance.append(Derivative)

    if "Dependent" in properties:
        inheritance.append(Dependent)
    elif "Independent" in properties:
        inheritance.append(Independent)
    else:
        raise ValueError(f"Function {name:s} must have either 'Dependent' or 'Independent' as property")

    if "Piecewise" in properties:
        inheritance.append(Piecewise)
    elif "Piecewise" not in properties:
        inheritance.append(NonPiecewise)
    else:
        raise ValueError(f"Function {name:s} must have either 'Piecewise' or 'NonPiecewise' as property")
    return inheritance


def generateFunction(
    name: str,
    contents: Dict[str, Union[str, dict]],
    model: Model = None,
    filestem: str = None
) -> Function:
    """
    Generate Function object.

    :param name: name of function to generate
    :param contents: dictionary of info needed to generate function
    :param model: :class:`~Function.Model` to add function into
    :param filestem: stem of filepath where function was loaded form, optional
    :returns: Generated function object
    """
    kwargs = getFunctionInfo(contents, model=model)
    kwargs["filestem"] = filestem

    contents_keys = contents.keys()
    if "form" in contents_keys:
        expression = contents["form"]
    elif "pieces" in contents_keys:
        expression = contents["pieces"]
    else:
        raise ValueError("info from functions_yml file must contain either form or pieces")

    inheritance = getInheritance(kwargs["properties"])
    SymbolicVariables.addVariables(names=name)

    function_obj = FunctionMaster(
        name,
        expression,
        inheritance=tuple(inheritance),
        **kwargs
    )
    return function_obj


def generateParameter(
    name: str,
    contents: Dict[str, Union[float, str]],
    **kwargs
) -> Parameter:
    """
    Generate Parameter object.

    :param name: name of parameter to generate
    :param contents: dictionary of info needed to generate parameter
    :param kwargs: additional arguments to pass into :class:`~Function.Parameter`
    :returns: Generated parameter object
    """
    value = float(contents["value"])
    unit = contents["unit"]
    quantity = value * units(unit)
    SymbolicVariables.addVariables(names=name)

    parameter_obj = Parameter(
        name,
        quantity,
        **kwargs
    )
    return parameter_obj


def generateVariable(
    name: str,
    contents: Dict[str, Any],
    **kwargs
) -> Variable:
    """
    Generate Variable object.

    :param name: name of variable to generate
    :param contents: dictionary of info needed to generate variable
    :param kwargs: additional arguments to pass into :class:`~Function.Variable`
    :returns: Generated variable object
    """
    time_evolution_type = contents["time_evolution_type"]
    initial_condition = contents["initial_condition"]
    SymbolicVariables.addVariables(names=name)

    variable_obj = Variable(
        name,
        time_evolution_type=time_evolution_type,
        initial_condition=initial_condition,
        **kwargs
    )
    return variable_obj


def createModel(function_ymls: Union[str, List[str]], parameter_ymls: Union[str, List[str]]) -> Model:
    """
    Create model from YML files.

    :param function_ymls: name(s) of YML file(s) containing info for function
    :param parameter_ymls: name(s) of YML file(s) containing info about parameter values/units
    """
    model = Model()
    model.loadParametersFromFiles(parameter_ymls)
    model.loadFunctionsFromFiles(function_ymls)
    return model


def readQuantitiesFromFiles(
    filepaths: Union[str, List[str]],
    specie: str,
    names: Iterable[str] = None
) -> Dict[str, PaperQuantity]:
    """
    Read file containing information about paper quantities.

    :param filepaths: name(s) of file(s) containing information
    :param names: only retrieve quantity objects in this collection of names, acts as filter.
        Defaults to a retrieve all quantity objects.
    :param specie: specie of paper quantity that file(s) contains info for.
        Must be "Parameter" or "Function".
    :returns: Dictionary of paper quantities.
        Key is name of paper quantity.
        Value is quantity object for paper quantity.
    """
    load_all = names is None
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    else:
        for filepath in filepaths:
            assert isinstance(filepath, str)

    if specie == "Parameter":
        generate = generateParameter
    elif specie == "Function":
        generate = generateFunction
    else:
        raise ValueError("invalid value for specie")

    quantity_objects = {}
    for filepath in filepaths:
        objs_contents = loadConfig(filepath)
        filestem = Path(filepath).stem
        for name, contents in objs_contents.items():
            if load_all or name in names:
                quantity_objects[name] = generate(
                    name,
                    contents,
                    filestem=filestem,
                )

    return quantity_objects


def readParametersFromFiles(*args, **kwargs) -> Dict[str, Parameter]:
    """
    Read file containing information about parameters.

    :param args: required arguments to pass into :meth:`~Function.readQuantitiesFromFiles`
    :param kwargs: additional arguments to pass into :meth:`~Function.readQuantitiesFromFiles`
    :returns: Dictionary of parameter objects.
        Key is name of parameter.
        Value is parameter object for parameter.
    """
    # noinspection PyTypeChecker
    return readQuantitiesFromFiles(*args, **kwargs, specie="Parameter")


def readFunctionsFromFiles(*args, **kwargs) -> Dict[str, Function]:
    """
    Read file containing information about parameters.

    :param args: required arguments to pass into :meth:`~Function.readQuantitiesFromFiles`
    :param kwargs: additional arguments to pass into :meth:`~Function.readQuantitiesFromFiles`
    :returns: Dictionary of function objects.
        Key is name of function.
        Value is function object for function.
    """
    # noinspection PyTypeChecker
    return readQuantitiesFromFiles(*args, **kwargs, specie="Function")
