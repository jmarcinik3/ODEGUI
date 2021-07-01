from __future__ import annotations

from collections.abc import KeysView
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import yaml
from numpy import ndarray
from pint import Quantity
from scipy import optimize
from sympy import Expr
from sympy import Piecewise as spPiecewise
# noinspection PyUnresolvedReferences
from sympy import Symbol, exp, pi, solve, symbols, var
from sympy.core import function
from sympy.utilities.lambdify import lambdify

import YML
from CustomErrors import RecursiveTypeError
from macros import getIndicies, unique


class Model:
    """
    __Purpose__
        Store Function objects for given model
    __Attributes__
        functions [list of Function]: stored Function objects
        parameters [dict of metpy.Quantity]: stored parameter values and units
            key [str] is parameter name
            value [metpy.Quantity] contains value and unit
        include_children [bool]: whether or not to include children of Function objects
            set True to include function and its children
            set False to include only given function, not children
    """

    def __init__(self, functions: List[Function] = None, parameters: Dict[str, Quantity] = None,
                 include_children: bool = True) -> None:
        self.functions = []
        self.parameters = {}
        self.include_children = include_children
        if parameters is not None: self.addParameters(parameters)
        if functions is not None: self.addFunctions(functions)

    def addParameters(self, quantities: Dict[str, Quantity]) -> None:
        """
        Set/add parameter(s) value and unit for model as Quantity.
        
        :param self: model to set parameter(s) for
        :param quantities: key is name of parameter.
            Value is Quantity storing value and unit.
        """
        for name, quantity in quantities.items(): self.parameters[name] = quantity

    def getParameters(self, names: Union[str, List[str]] = None) -> Union[Quantity, Dict[str, Quantity]]:
        """
        __Purpose__
            Get parameters stored in model
        __Return__
            metpy.Quantity: names [str, sympy.Symbol]
            dict of metpy.Quantity: names [list of str/sympy.Symbol]
                key [str] is name of parameter
                value [metpy.Quantity] is value and unit of parameter
        __Recursion Base__
            return single parameter: names [str]
            return all parameters: names [None]

        :param self: model to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retrieve
        """
        if isinstance(names, str):
            return self.parameters[names]
        elif isinstance(names, Symbol):
            return self.getParameters(names=str(names))
        elif isinstance(names, list):
            return {str(name): self.getParameters(names=name) for name in names}
        elif names is None:
            return self.parameters
        else:
            raise RecursiveTypeError(names, [str, Symbol])

    def getVariables(self, functions: Union[Function, List[Function]] = None) -> List[Symbol]:
        """
        __Purpose__
            Get variable(s) associated with function(s) in model
        __Recursion Base__
            return variable associated with single function: functions [Function]

        :param self: model to retrieve function objects from
        :param functions: function to retrieve variables from
        """
        if functions is None: functions = self.getFunctions()

        if isinstance(functions, Function):
            return functions.getVariables()
        elif isinstance(functions, list):
            variables = []
            for function in functions: variables.extend(self.getVariables(functions=function))
            return unique(variables)
        else:
            raise RecursiveTypeError(functions, Function)

    def addFunctions(self, functions: Union[Function, List[Function]]) -> None:
        """
        __Purpose__
            Add Function object(s) to self
            Set self for Function object(s)
        __Attributes__
            include_children: adds children of Function object(s) if set to True

        :param self: model to add function to
        :param functions: function(s) to add to model
        """
        if isinstance(functions, Function):
            if functions not in self.getFunctions():
                name = functions.getName()
                if name in self.getFunctionNames(): raise ValueError(
                    f"Function name {name:s} already used in Model instance")
                if functions.getModel() is not self: functions.setModel(self)
                self.functions.append(functions)
            if self.include_children: self.addFunctions(functions.getChildren())
        elif isinstance(functions, list):
            for function in functions: self.addFunctions(function)
        else:
            raise RecursiveTypeError(functions, Function)

    def loadFunctionsFromFiles(self, function_ymls: Union[str, List[str]]) -> None:
        """
        __Purpose__
            Add functions to model by parsing through YML file

        :param self: model to add function(s) to
        :param function_ymls: name of YML file to retrieve function info from
        """
        if isinstance(function_ymls, str): function_ymls = [function_ymls]

        for function_yml in function_ymls:
            file = yaml.load(open(function_yml), Loader=yaml.Loader)

            for name in file.keys():
                info = file[name]
                kwargs = getFunctionInfo(info, model=self)

                info_keys = info.keys()
                if "form" in info_keys:
                    function = eval(info["form"])
                elif "pieces" in info_keys:
                    function = [self.getFunctions(names=piece_name) for piece_name in info["pieces"]]
                else:
                    raise ValueError("info from functions_yml file must contain either form or pieces")

                createFunction(name, function, **kwargs)

    def loadParametersFromFile(self, parameters_yml: str) -> None:
        """
        __Purpose__
            Add parameters to model by parsing through YML file

        :param self: model to add function(s) to
        :param parameters_yml: name of YML file to retrieve parameter info from
        """
        parameters = YML.readParameters(parameters_yml)
        self.addParameters(parameters)

    def saveParametersToFile(self, filename: str) -> None:
        parameters = self.getParameters()
        parameters_dict = {
            name: {
                "value": quantity.magnitude,
                "unit": str(quantity.units)
            }
            for name, quantity in parameters.items()
        }
        file = open(filename, 'w')
        yaml.dump(parameters_dict, file)

    def saveTimeEvolutionTypesToFile(self, filename: str) -> None:
        derivatives = self.getDerivatives()
        time_evolution_types = {str(derivative.getVariable()): derivative.getTimeEvolutionType() for derivative in
                                derivatives}
        file = open(filename, 'w')
        yaml.dump(time_evolution_types, file)

    def getFunctionNames(self, functions: Union[Function, List[Function]] = None) -> Union[str, List[str]]:
        """
        Get name(s) of stored function(s) in order added
        __Recursion Base__
            retrieve name of single function: functions [Function]
            retrieve names of all functions: functions [None]

        :param self: model to retrieve function name(s) from
        :param functions: function(s) to retrieve name of
        """
        if isinstance(functions, Function):
            return functions.getName()
        elif isinstance(functions, list):
            return [self.getFunctionNames(functions=function) for function in functions]
        elif functions is None:
            return self.getFunctionNames(functions=self.functions)
        else:
            raise RecursiveTypeError(functions, Function)

    def getIndex(self, names: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Get index(es) of stored function(s) in order added.

        :param self: model to retrieve parameter index(es) from
        :param names: name(s) of parameter(s) to retrieve index(es) of
        """
        if isinstance(names, (str, list)):
            return getIndicies(names, self.getFunctionNames(), str)
        else:
            raise RecursiveTypeError(names)

    def getFunctions(self, names: Union[str, List[str]] = None, filter_type: Type[Function] = None) -> Union[
        Function, List[Function]]:
        """
        __Purpose__
            Get functions stored in model
        __Recursion Base__
            return single function if compatible with filter type: names [None]

        :param self: model to retrieve function(s) from
        :param names: name(s) of function(s) to retrieve
        :param filter_type: only retrieve function(s) of this class, acts as a filter
        """
        if isinstance(names, str):
            function = self.functions[self.getIndex(names)]
            if filter_type is None or isinstance(function, filter_type):
                return function
            else:
                raise TypeError(f"names input must correspond to {filter_type:s}")
        elif isinstance(names, list):
            functions = [self.getFunctions(names=name) for name in names]
            if filter_type is None:
                return functions
            else:
                return [function for function in functions if isinstance(function, filter_type)]
        elif names is None:
            return self.getFunctions(names=self.getFunctionNames(), filter_type=filter_type)
        else:
            raise RecursiveTypeError(names)

    def getDerivatives(self, time_evolution_types: Union[str, List[str]] = None) -> List[Derivative]:
        """
        __Purpose__
            Get stored derivatives of given time-evolution type(s)
        __Recursion Base__
            get derivatives of single time-evolution type: time_evolution_types [str]
            get all derivatives: time_evolution_types [None]

        :param self: model to retrieve derivative(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        """
        if isinstance(time_evolution_types, str):
            derivatives_all = self.getFunctions(filter_type=Derivative)
            return [derivative for derivative in derivatives_all if
                    derivative.getTimeEvolutionType() == time_evolution_types]
        elif isinstance(time_evolution_types, list):
            derivatives = []
            for time_evolution_type in time_evolution_types: derivatives.extend(
                self.getDerivatives(time_evolution_types=time_evolution_type))
            return derivatives
        elif time_evolution_types is None:
            return self.getFunctions(filter_type=Derivative)
        else:
            raise RecursiveTypeError(time_evolution_types)

    def getEquilibriumSolutions(self, names: Union[str, List[str]] = None,
                                substitute_parameters: bool = True,
                                skip_parameters: Union[str, List[str]] = None,
                                substitute_constants: bool = True,
                                substitute_functions: bool = True) -> Dict[Symbol, Expr]:
        """
        __Purpose__
            Get substitutions to substitute equilibrium variables into non-equilibrium derivatives
        __Return__
            dict:
                key [sympy.Symbol] is variable to substitute into
                value [sympy.Expr] is expression to substitute into variable

        :param self: model to solve for equilibrium solution(s) in
        :param names: name(s) of function(s) to solve for simulatenous equilibrium of
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getEquilibriumSolutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding functions in for all function-type derivatives.
            Set False to leave them as symbolic variables.
        """
        if skip_parameters is None: skip_parameters = []

        if isinstance(names, (str, list)):
            equilibria = self.getFunctions(names=names)
            if not all([isinstance(function, Derivative) for function in equilibria]): raise TypeError(
                "names must correspond to Derivative stored in Model")
        elif names is None:
            equilibria = self.getDerivatives(time_evolution_types="Equilibrium")
        else:
            raise RecursiveTypeError(names)

        equilibrium_count = len(equilibria)
        if equilibrium_count == 0:
            return {}
        elif equilibrium_count >= 1:
            equilibrium_variables = [equilibrium.getVariable() for equilibrium in equilibria]
            equilibrium_derivatives = [equilibrium.getForm(generations="all") for equilibrium in equilibria]

            bk_probs = list(symbols("pC0 pC1 pC2 pO2 pO3"))
            if set(bk_probs).issubset(set(equilibrium_variables)): equilibrium_derivatives.append(sum(bk_probs) - 1)
            solutions = solve(equilibrium_derivatives, equilibrium_variables)

            if substitute_functions:
                function_substitutions = self.getFunctionSubstitutions(
                    substitute_parameters=substitute_parameters,
                    skip_parameters=skip_parameters,
                    substitute_constants=substitute_constants
                )
                solutions = {
                    variable: solution.subs(function_substitutions)
                    for variable, solution in
                    solutions.items()
                }
            if substitute_parameters:
                parameter_names = []
                parameter_names_extend = parameter_names.extend
                for equilibrium in equilibria:
                    new_parameter_names = [
                        parameter
                        for parameter in equilibrium.getParameters(return_type=str)
                        if parameter not in skip_parameters
                    ]
                    parameter_names_extend(new_parameter_names)
                parameter_names = unique(parameter_names)
                parameter_substitutions = self.getParameterSubstitutions(parameter_names)

                solutions = {
                    variable: solution.subs(parameter_substitutions)
                    for variable, solution in solutions.items()
                }
            if substitute_constants:
                constant_substitutions = self.getConstantSubstitutions()
                solutions = {
                    variable: solution.subs(constant_substitutions)
                    for variable, solution in solutions.items()
                }
            return solutions

    def getEquilibriumFunction(self, name: str,
                               substitute_parameters: bool = True,
                               skip_parameters: Union[str, List[str]] = None,
                               substitute_constants: bool = True,
                               substitute_functions: bool = True) -> Expr:
        """
        __Purpose__
            Get equilibrium function corresponding to derivative

        :param self: model to solve for equilibrium solution in
        :param name: name of variable to retrieve equilibrium expression of
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getEquilibriumSolutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding functions in for all function-type derivatives.
            Set False to leave them as symbolic variables.
        """
        equilibria = self.getEquilibriumSolutions(
            substitute_parameters=substitute_parameters,
            skip_parameters=skip_parameters,
            substitute_constants=substitute_constants,
            substitute_functions=substitute_functions
        )
        function = self.getDerivativesFromVariableNames(names=name)
        equilibrium = equilibria[function.getVariable()]
        return equilibrium

    def getFunctionSubstitutions(self, names: Union[str, List[str]] = None,
                                 substitute_parameters: bool = True,
                                 skip_parameters: Union[str, List[str]] = None,
                                 substitute_constants: bool = True) -> Dict[Symbol, Expr]:
        """
        :param self: model to retrieve functions from
        :param names: name(s) of variable(s) to retrieve function for
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set False to leave them as symbolic variables.
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getFunctionSubstitutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        """
        if skip_parameters is None: skip_parameters = []
        if names is None: names = self.getDerivativeVariables(time_evolution_types="Function", return_type=str)

        variable_count = len(names)
        if variable_count == 0:
            return {}
        elif variable_count >= 1:
            functions = self.getFunctions(names=names)
            forms = [function.getForm(generations="all") for function in functions]
            variables = [Symbol(name) for name in names]

            if substitute_parameters:
                parameter_names = []
                parameter_names_extend = parameter_names.extend

                for function in functions:
                    new_parameter_names = [
                        parameter_name
                        for parameter_name in function.getParameters(return_type=str)
                        if parameter_name not in skip_parameters
                    ]
                    parameter_names_extend(new_parameter_names)
                parameter_names = unique(parameter_names)

                parameter_substitutions = self.getParameterSubstitutions(parameter_names)
                forms = [form.subs(parameter_substitutions) for form in forms]
            if substitute_constants:
                constant_substitutions = self.getConstantSubstitutions()
                forms = [form.subs(constant_substitutions) for form in forms]

            substitutions = {variables[i]: forms[i] for i in range(variable_count)}
            return substitutions

    def getConstantSubstitutions(self, names: Union[str, List[str]] = None) -> Dict[Symbol, Expr]:
        """
        __Purpose__
            Get substitutions to substitute constant intitial condition into given variable(s)
        __Return__
            dict:
                key [sympy.Symbol] is variable to substitute into
                value [sympy.Expr] is constant to substitute into variable

        :param self: model to retrieve constant derivative(s) from
        :param names: name(s) of constant derivative(s) to substitute numerical constants in for
        """
        if isinstance(names, (str, list)):
            constant_functions = self.getFunctions(names=names)
            for function in constant_functions:
                if not isinstance(function, Derivative): raise TypeError(
                    "names must correspond to Derivative stored in Model")
        elif names is None:
            constant_functions = self.getDerivatives(time_evolution_types="Constant")
        else:
            raise RecursiveTypeError(names)
        constant_count = len(constant_functions)
        if constant_count == 0:
            return {}
        elif constant_count >= 1:
            substitutions = {
                function.getVariable(): function.getInitialCondition()
                for function in constant_functions
            }
            return substitutions

    def getParameterSubstitutions(self, names: Union[str, List[str]] = None,
                                  skip_parameters: Union[str, List[str]] = None) -> Dict[Symbol, Expr]:
        """
        Substitute parameters into function from model.

        :param names: name(s) of parameter to include in substitutions.
            Defaults to all parameters in model.
        :param skip_parameters: name(s) of parameter(s) to skip substitution for
        """
        if skip_parameters is None: skip_parameters = []
        quantities = self.getParameters(names=names)
        if names is None: names = quantities.keys()

        substitutions = {
            Symbol(parameter_name):
                quantities[parameter_name].to_base_units().magnitude
            for parameter_name in names
            if parameter_name not in skip_parameters
        }
        return substitutions

    def getDerivativeVector(self, names: List[str] = None,
                            expanded: bool = True,
                            substitute_parameters: bool = True,
                            skip_parameters: Union[str, List[str]] = None,
                            substitute_equilibria: bool = True,
                            substitute_constants: bool = True,
                            substitute_functions: bool = True,
                            lambdified: bool = False) -> Tuple[
        Union[List[Expr], function], Optional[Dict[Symbol, Expr]]]:
        """
        Get derivative vector corresponding to derivatives in :class:`~Function.Model`

        :param self: model to retrieve derivative(s) from
        :param names: name(s) of variable(s) ordered in same order derivatives will be returned
        :param expanded: set True to get expanded expression for derivative, i.e. substitute all subexpressions into derivative expression.
            Set False to leave subexpressions as symbolic variables.
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getDerivativeVector.substitute_parameters` is set True.
        :param substitute_equilibria: set True to substitute equilibrium expressions in for variables in equilibrium.
            Set False to leave them as symbolic variables.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding functions in for all function-type derivatives.
            Set False to leave them as symbolic variables.
        :param lambdified: set True to return derivative vector as lambda function handle.
            Set False to return symbolic derivative vector
        :returns: vector of derivatives.
            Uses derivatives in :class:`~Function.Model` to determine derivatives.
            Returned as list of symbolic expressions if :paramref:`~Function.Model.getDerivativeVector.lambdified` is set to True.
            Returned as lambda function handle if :paramref:`~Function.Model.getDerivativeVector.lambdified` is set to False.
        """
        if skip_parameters is None: skip_parameters = []

        variable_substitutions = {}
        if substitute_equilibria:
            kwargs = {
                "substitute_parameters": substitute_parameters,
                "substitute_constants": substitute_constants,
                "skip_parameters": skip_parameters
            }
            equilibrium_solutions = self.getEquilibriumSolutions(**kwargs)
            variable_substitutions.update(equilibrium_solutions)
        if substitute_constants: variable_substitutions.update(self.getConstantSubstitutions())
        if substitute_functions:
            kwargs = {
                "substitute_parameters": substitute_parameters,
                "substitute_constants": substitute_constants,
                "skip_parameters": skip_parameters
            }
            variable_substitutions.update(self.getFunctionSubstitutions(**kwargs))

        parameter_substitutions = {}
        if substitute_parameters:
            kwargs = {
                "skip_parameters": skip_parameters
            }
            parameter_substitutions.update(self.getParameterSubstitutions(**kwargs))

        if names is None:
            temporal_derivatives = self.getDerivatives(time_evolution_types="Temporal")
            names = [str(derivative.getVariable()) for derivative in temporal_derivatives]
        else:
            temporal_derivatives = self.getDerivativesFromVariableNames(names=names)

        derivative_vector = []
        for derivative in temporal_derivatives:
            if expanded:
                form = derivative.getForm(generations="all")
            else:
                form = derivative.getForm()

            variable_substitution = {
                variable: substitution
                for variable, substitution in variable_substitutions.items()
                if variable in derivative.getVariables()
            }
            parameter_substitution = {
                parameter: value
                for parameter, value in parameter_substitutions.items()
                if parameter in derivative.getParameters()
            }
            form = form.subs({**variable_substitution, **parameter_substitution})

            derivative_vector.append(form)

        if lambdified: derivative_vector = lambdify((Symbol('t'), tuple(names)), derivative_vector)
        if substitute_equilibria:
            return derivative_vector, equilibrium_solutions
        else:
            return derivative_vector

    def getInitialValues(self,
                         names: Union[str, List[str]] = None,
                         return_type: Type[dict, list, ndarray] = dict,
                         initial_values: Dict[Symbol, float] = None) -> Union[float, List[float], ndarray, dict]:
        """
        Get initial values for variables in model.
        
        :param self: model to retrieve derivatives from
        :param names: name(s) of variables to retrieve values for
        :param return_type: class type for output.
            Must be dict, list, or ndarray.
            Only called if names is list.
        :param initial_values: dictionary of initial values if already known.
            Key is symbol for variable.
            Value is float for initial value.
        :returns: Initial values for desired variables.
            Dictionary of initial values if return_type is dict; key is symbol for variable, value if float for initial value.
            List of floats if return_type is list.
            ndarray of float if return_type is ndarray.
        """
        initial_values = {
            derivative.getVariable(): initial_value
            for derivative in self.getDerivatives()
            if isinstance((initial_value := derivative.getInitialCondition()), float)
        }

        if initial_values is None:
            derivatives = self.getDerivatives()

            initial_constant_substitutions = {
                derivative.getVariable(): initial_value
                for derivative in derivatives
                if isinstance((initial_value := derivative.getInitialCondition()), float)
            }

            substitutions = {
                **initial_constant_substitutions,
                **self.getConstantSubstitutions(),
                **self.getParameterSubstitutions()
            }

            equilibrium_equations, equilibrium_variables = [], []
            equations_append, variables_append = equilibrium_equations.append, equilibrium_variables.append
            for derivative in derivatives:
                if derivative.getInitialCondition() == "Equilibrium":
                    equations_append(derivative.getForm(generations="all").subs(substitutions))
                    variables_append(derivative.getVariable())

            variable_count = len(equilibrium_variables)
            equations_lambda = lambdify((tuple(equilibrium_variables),), equilibrium_equations)
            initial_guess = np.repeat(0.5, variable_count)
            roots = optimize.root(equations_lambda, initial_guess)
            solutions = {
                equilibrium_variables[i]: roots.x[i]
                for i in range(variable_count)
            }

            initial_values = {**initial_constant_substitutions, **solutions}
            print('1', initial_values)

        if isinstance(names, str):
            return initial_values[Symbol(names)]
        elif isinstance(names, Symbol):
            return initial_values[names]
        elif isinstance(names, list):
            initial_value = lambda name: self.getInitialValues(names=name, initial_values=initial_values)
            if return_type == list:
                return [initial_value(name) for name in names]
            elif return_type == ndarray:
                return np.array([initial_value(name) for name in names])
            elif return_type == dict:
                return {Symbol(name): initial_value(name) for name in names}
            else:
                raise ValueError("return_type must be list, ndarray, or dict")
        else:
            raise RecursiveTypeError(names, [str, Symbol])

    def getDerivativeVariables(self,
                               time_evolution_types: Union[str, List[str]] = None,
                               return_type: Type[Symbol, str] = Symbol) -> Union[List[Symbol], List[str]]:
        """
        Get variables stored in model in same order as derivatives
        
        __Recursion Base__
            get symbolic variable associated with single derivative: names [str]

        :param self: model to retrieve derivative variable(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        :param return_type: class type to return elements in list output as
        """
        if isinstance(time_evolution_types, str):
            derivatives = self.getDerivatives(time_evolution_types=time_evolution_types)
            variables = [derivative.getVariable() for derivative in derivatives]
        elif isinstance(time_evolution_types, list):
            variables = [
                self.getDerivativeVariables(time_evolution_types=time_evolution_type)
                for time_evolution_type in time_evolution_types
            ]
        elif time_evolution_types is None:
            derivatives = self.getDerivatives()
            variables = [derivative.getVariable() for derivative in derivatives]
        else:
            raise RecursiveTypeError(time_evolution_types)

        if return_type == Symbol:
            return variables
        elif return_type == str:
            return [str(variable) for variable in variables]
        else:
            raise ValueError("return_type must be Symbol or str")

    def getDerivativesFromVariableNames(self, names: Union[str, Symbol, List[Union[str, Symbol]]]) -> Union[
        Derivative, List[Derivative]]:
        """
        Get derivative corresponding to variable name.

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        :param names: name(s) of variable(s) associated with derivative(s)
        """
        if isinstance(names, (str, Symbol)):
            derivatives = self.getDerivatives()
            for derivative in derivatives:
                if derivative.getVariable(return_type=str) == str(names): return derivative
            raise ValueError("names input must correspond to some Derivative stored in Model")
        elif isinstance(names, list):
            return [self.getDerivativesFromVariableNames(names=name) for name in names]
        else:
            RecursiveTypeError(names, [str, Symbol])


class Parent:
    """
    __Purpose__
        Properties for Function related to being a parent function to another function
    __Attributes__
        children [list of Function]: collection of children associated with function
        instance_arguments [list of dict list of sympy.Symbol]: arguments which parent function inputs into child function
            first list index is index of stored child function
            dictionary key is argument species
            second list index is index of individual argument sent to child function
    """

    def __init__(self, children: Union[Dict[str, Union[Dict[Function], Dict[Symbol, List[Symbol]]]]]) -> None:
        self.children = []
        self.instance_arguments = {}

        if children is not None:
            if isinstance(children, Child): children = [children]
            for name, arguments in children.items(): self.addChild(name, arguments)
        self.setAsParent(self.getChildren())

    def addChild(self, name: str, arguments: Dict[str, Union[Function, List[Symbol]]]) -> None:
        self.children.append(arguments["function"])
        self.instance_arguments[name] = {}
        for specie in arguments.keys():
            if specie != "function":
                instance_argument = arguments[specie]
                if isinstance(instance_argument, Symbol): instance_argument = [instance_argument]
                self.instance_arguments[name][specie] = instance_argument

    def setAsParent(self, children: Union[Function, List[Function]]) -> None:
        """
        __Purpose__
            Set self as parent to children
            Set children as children to self

        :param self: parent to set as parent for child(s)
        :param children: children to set new parent for
        """
        if isinstance(children, Function):
            children.addParent(self)
        elif isinstance(children, list):
            for child in children: self.setAsParent(child)
        else:
            raise RecursiveTypeError(children, Function)

    def getChildIndex(self, names: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        __Purpose__
            Get indicies of children functions

        :param self: parent to retrieve child index(es) from
        :param names: name(s) of child(s) to retrieve from parent
        """
        if isinstance(names, (str, list)):
            return getIndicies(names, self.getChildrenNames(), str)
        else:
            raise RecursiveTypeError(names)

    def getChildren(self, names: Union[str, List[str]] = None) -> Union[Function, List[Function]]:
        """
        __Purpose__
            Get children functions in order added
        __Recursion Base__
            return all children: names [None]

        :param self: parent to retrieve child(s) from
        :param names: name(s) of child(s) to retrieve from parent
        """
        if isinstance(names, str):
            return self.children[self.getChildIndex(names)]
        elif isinstance(names, list):
            return [self.getChildren(names=name) for name in names]
        elif names is None:
            return self.getChildren(names=self.getChildrenNames())
        else:
            raise RecursiveTypeError(names)

    def getChildrenNames(self, functions: Union[Function, List[Function]] = None) -> Union[str, List[str]]:
        """
        __Purpose__
            Get names of stored parents in order added
        __Recursion Base__
            retrieve name of single parent: functions [Function]
            retrieve names of all parents: functions [None]

        :param self: parent to retrieve child name(s) from
        :param functions: function(s) to retrieve name(s) from parent
        """
        if isinstance(functions, Function):
            return functions.getName()
        elif isinstance(functions, list):
            return [self.getChildrenNames(functions=function) for function in functions]
        elif functions is None:
            return self.getChildrenNames(functions=self.children)
        else:
            raise RecursiveTypeError(functions, Function)

    def getInstanceArgumentSpecies(self, name: str) -> List[str]:
        """
        __Purpose__
            Get all possible species of instance arguments

        :param self: parent to retrieve instance-argument species from
        :param name: name of child to retrieve instance-argument species for
        """
        return self.instance_arguments[name].keys()

    def getInstanceArguments(self, name: str, specie: str = None) -> Union[Symbol, List[Symbol]]:
        """
        __Purpose__
            Get instance arguments of given species for each function associated with given names
        __Recursion Base__
            return all instance arguments of given specie: specie [None] and names [None]

        :param self: parent to retrieve instance arguments from
        :param specie: name of instance-argument species to retrieve from parent
        :param name: name of child function to retrieve instance arguments for
        """
        if isinstance(specie, str):
            return self.instance_arguments[name][specie]
        elif specie is None:
            return self.instance_arguments[name]
        else:
            raise TypeError("specie must be str")


class Child:
    """
    __Purpose__
        Properties for Function related to being a child function to another function
    __Attributes__
        parents [list of Function]: collection of parents associated with function
    """

    def __init__(self) -> None:
        self.parents = []

    def addParent(self, parents: Union[Function, List[Function]]) -> None:
        """
        __Purpose__
            Add function(s) as parent(s) to self
        __Recursion Base__
            Add single function as parent to self: parents [type(self)]

        :param self: child to add parent(s) to
        :param parents: parent(s) to add to child
        """
        if isinstance(parents, Function) and parents not in self.getParents():
            self.parents.append(parents)
        elif isinstance(parents, list):
            for parent in parents: self.addParent(parent)
        else:
            raise RecursiveTypeError(parents, Function)

    def getParentIndex(self, names: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        __Purpose__
            Get indicies of parents functions

        :param self: child to retrive parent index(es) from
        :param names: name(s) of parent(s) to retrieve index(es) from child
        """
        if isinstance(names, (str, list)):
            return getIndicies(names, self.getParentNames(), str)
        else:
            raise RecursiveTypeError(names)

    def getParents(self, names: Union[str, List[str]] = None) -> Union[Function, List[Function]]:
        """
        __Purpose__
            Get parent functions in order added
        __Recursion Base__
            return all parents: names [None]

        :param self: child to retrieve parent(s) from
        :param names: name(s) of parent(s) to retrieve from child
        """
        if isinstance(names, str):
            return self.parents[self.getParentIndex(names)]
        elif isinstance(names, list):
            return [self.getParents(names=name) for name in names]
        elif names is None:
            return self.getParents(names=self.getParentNames())
        else:
            raise RecursiveTypeError(names)

    def getParentNames(self, functions: Union[Function, List[Function]] = None) -> Union[str, List[str]]:
        """
        __Purpose__
            Get names of stored parents in order added
        __Recursion Base__
            retrieve name of single parent: functions [Function]
            retrieve names of all parents: functions [None]

        :param self: child to retrieve parent name(s) from
        :param functions: parent(s) to retrieve name(s) from child
        """
        if isinstance(functions, Function):
            return functions.getName()
        elif isinstance(functions, list):
            return [self.getParentNames(functions=function) for function in functions]
        elif functions is None:
            return self.getParentNames(functions=self.parents)
        else:
            raise RecursiveTypeError(functions, Function)


class Function(Child, Parent):
    """
    __Purpose__
        Object to store core information about a function
    __Attributes__
        name [str]: name of function
        variables [list of sympy.Symbol]: collection of variables explicitly included in function form
        parameters [list of sympy.Symbol]: collection of parameters explicitly included in function form
        model [Model]: model to associated with function
    """

    def __init__(self, name: str, variables: Union[Symbol, List[Symbol]] = None,
                 parameters: Union[Symbol, List[Symbol]] = None, children: Union[Child, List[Child]] = None,
                 model: Model = None, **kwargs) -> None:
        self.name = name

        self.variables = []
        self.addVariables(variables)
        self.parameters = []
        self.addParameters(parameters)

        Parent.__init__(self, children)
        Child.__init__(self)

        self.model = Model()
        self.setModel(model)

    def getName(self) -> str:
        """
        __Purpose__
            Get name of function

        :param self: function to retrieve name of
        """
        return self.name

    def getSymbol(self) -> Symbol:
        """
        __Purpose__
            Get symbolic variable of function

        :param self: function to retrieve symbolic variable of
        """
        return Symbol(self.getName())

    def addVariables(self, variables: Union[Symbol, List[Symbol]]) -> None:
        """
        __Purpose__
            Add variable to function

        :param self: function to add variable(s) to
        :param variables: variable(s) to add to function
        """
        if isinstance(variables, Symbol):
            self.variables.append(variables)
        elif isinstance(variables, list):
            for variable in variables:
                if not isinstance(variable, Symbol): raise TypeError("variable must be sympy.Symbol")
                self.addVariables(variable)
        elif variables is None:
            pass
        else:
            raise RecursiveTypeError(variables, Symbol)

    def addParameters(self, parameters: Union[Symbol, List[Symbol]]) -> None:
        """
        __Purpose__
            Set parameters attribute as list

        :param self: function to add parameter(s) to
        :param parameters: parameter(s) to add to function
        """
        if isinstance(parameters, Symbol):
            self.parameters.append(parameters)
        elif isinstance(parameters, list):
            for parameter in parameters:
                if not isinstance(parameter, Symbol): raise TypeError("parameter must be sympy.Symbol")
                self.addParameters(parameter)
        elif parameters is None:
            pass
        else:
            raise RecursiveTypeError(parameters, Symbol)

    def setModel(self, model: Model) -> None:
        """
        __Purpose__
            Set model attribute as Model
            Add function to model if not already in model

        :param self: function to set model with
        :param model: new model to associated with function
        """
        self.model = model
        if isinstance(model, Model):
            if self not in model.getFunctions(): model.addFunctions(self)
        else:
            raise TypeError("model input must be Model")

    def getModel(self) -> Model:
        """
        __Purpose__
            Get model associated with function

        :param self: function to retrieve model of
        """
        return self.model


class Derivative:
    """
    __Purpose__
        Properties relating to function being a time derivative
    __Attributes__
        variable [sympy.Symbol]: variable that derivative is derivative of
        time_evolution_type [str]: time-evolution type of derivative (e.g. "Temporal", "Equilibrium", "Constant")
        initial_condition [float]: initial condition value for associated variable
    """

    def __init__(self, variable: Symbol, time_evolution_type: str = "Temporal",
                 initial_condition: float = 0) -> None:
        """
         __Inputs__
            variable [sympy.Symbol, str]: variable that derivative is a derivative of
            time_evolution_type [str]: time-evolution type of derivative (e.g. "Temporal", "Equilibrium", "Constant")
            initial_condition [float]: initial value of variable
        """
        self.variable = None
        self.setVariable(variable)
        self.time_evolution_type = time_evolution_type
        self.setTimeEvolutionType(time_evolution_type)
        self.initial_condition = initial_condition
        self.setInitialCondition(initial_condition)

    def getVariable(self, return_type: Type[Symbol, str] = Symbol) -> Union[Symbol, str]:
        """
        Get variable that derivative is derivative of.

        :param self: :class:`~Function.Derivative` to retreive variable from
        :param return_type: class type of output.
            Must be either sympy.Symbol or str.
        """
        if return_type == Symbol:
            return self.variable
        elif return_type == str:
            return str(self.variable)
        else:
            raise ValueError("return_type must be sp.Symbol or str")

    def setVariable(self, variable: Union[str, Symbol]) -> None:
        """
        Set variable associated with derivative.
        
        :param self: :class:`~Function.Derivative` associated with variable
        :param variable: variable to associated with derivative
        """
        if isinstance(variable, str):
            self.setVariable(Symbol(variable))
        elif isinstance(variable, Symbol):
            self.variable = variable
        else:
            raise TypeError("variable input must be str of sympy.Symbol")

    def getTimeEvolutionType(self) -> str:
        """
        Get time-evolution type of variable associated with derivative.

        :param self: :class:`~Function.Derivative` to retrieve time-evolution type from
        """
        return self.time_evolution_type

    def setTimeEvolutionType(self, time_evolution_type: str) -> None:
        """
        Set time-evolution type for variable.

        :param self: :class:`~Function.Derivative` to set time-evolution type for
        :param time_evolution_type: time-evolution type to set for variable
        """
        self.time_evolution_type = time_evolution_type

    def getInitialCondition(self) -> Union[str, float]:
        """
        Get initial numerical condition for variable associated with derivative.

        :param self: :class:`~Function.Derivative` to retrieve initial condition from
        :returns: Initial value float if value is provided.
            "Equilibrium" if variable begins in equilibrium with respect to other variables.
        """
        if self.getTimeEvolutionType() == "Equilibrium":
            initial_condition = "Equilibrium"
        else:
            initial_condition = self.initial_condition
        return initial_condition

    def setInitialCondition(self, value: Union[str, float]) -> None:
        """
        Set initial condition or value for variable.

        :param self: :class:`~Function.Derivative` to set initial condition for
        :param value: initial condition to set for variable
        """
        self.initial_condition = value


class Dependent:
    """
    __Purpose__
        Store properties for a function that requires input from another function
    __Attributes__
        arguments [dict of list of sympy.Symbol]: arguments to be substituted into from another function
            keys [str] are species of argument (e.g. "variables", "parameters", "functions")
            values [sympy.Symbol] are symbolic general arguments
    """

    def __init__(self, arguments: Union[Symbol, List[Symbol]]) -> None:
        self.general_arguments = {}
        self.setGeneralArguments(arguments)

    def setGeneralArguments(self, arguments: Union[Symbol, List[Symbol]], species: str = None) -> None:
        """
        __Purpose__
            Set receiver_arguments attribute as list or dict
        __Inputs__
            arguments [sympy.Symbol, list of sympy.Symbol, dict of list of sympy.Symbol]: contains information on how to set attribute
                sympy.Symbol, list of sympy.Symbol:
                    sets species [str] as key for attribute, single symbol in arguments as value
                    species must be given
                dict of list of sympy.Symbol:
                    key [str] is species type for argument
                    values [sympy.Symbol, list of sympy.Symbol] are arguments associated with species
            species [str]: species to associated with given arguments
        __Recursion Base__
            Set attribute for single species: arguments [sympy.Symbol, list of sympy.Symbol] and species [str]
        """
        if isinstance(species, str):
            if isinstance(arguments, Symbol):
                self.general_arguments[species] = [arguments]
            elif isinstance(arguments, list):
                for argument in arguments:
                    if not isinstance(argument, Symbol): raise TypeError("argument must be sympy.Symbol")
                self.general_arguments[species] = arguments
            elif arguments is None:
                self.general_arguments[species] = []
            else:
                raise RecursiveTypeError(arguments, Symbol)
        elif species is None:
            if isinstance(arguments, dict):
                for specie, argument in arguments.items(): self.setGeneralArguments(argument, specie)
            else:
                raise TypeError("arguments input must be dict when species input is None")
        else:
            raise TypeError("species input must be str")

    def getGeneralSpecies(self, nested: bool = True) -> List[str]:
        """
        __Purpose__
            Get species of general arguments that the dependent functions requires
        __Inputs__
            nested [bool]: set True to include species from children (implicit); set False to only include self species (explicit)
        """
        species = list(self.general_arguments.keys())
        if nested:
            for child in self.getChildren():
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

    def getGeneralArguments(self, species: Union[str, List[str]] = None, nested: bool = False) -> Union[
        List[Symbol], Dict[str, List[Symbol]]]:
        """
        __Purpose__
            Get general arguments of Dependent Function object
        __Inputs__
            species [str, list of str]: species of argument to get
            nested [bool]:
                True: includes general arguments of children functions
                False: includes general arguments explicitly of self function
        __Return__
            dict of list of sympy.Symbol: species [None, list of str]
                key is argument species
                value is list of sympy.Symbol general arguments
            list of sympy.Symbol: species [str]
        __Recursion Base__
            return arguments of single species: species [str] and nested [False]
        """
        if nested:
            general_arguments = self.getGeneralArguments(species=species, nested=False)
            for child in self.getChildren():
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
                    return self.general_arguments[species]
                elif species not in general_species:
                    return []
            elif isinstance(species, list):
                return {specie: self.getGeneralArguments(specie) for specie in species}
            elif species is None:
                return {specie: self.getGeneralArguments(specie) for specie in
                        self.getGeneralSpecies()}
            else:
                raise RecursiveTypeError(species)

    def getInstanceArgumentSubstitutions(self, parent: Function, specie: str) -> List[Expr]:
        """
        __Purpose__
            Get which instance arguments to substitute into general arguments
        __Inputs__
            parent [Parent]: parent function to retrieve instance arguments from
            specie [str]: species of argument to retrieve
        """
        self_name = self.getName()
        if specie == "functions":
            instance_function_symbols = parent.getInstanceArguments(specie=specie, name=self_name)
            instance_names = [str(instance_name) for instance_name in instance_function_symbols]
            instance_functions = []
            for instance_name in instance_names:
                sibling = parent.getChildren(names=instance_name)
                if isinstance(sibling, Dependent):
                    instance_functions.append(sibling.getInstanceArgumentFunction(parent))
                elif isinstance(sibling, Function):
                    instance_functions.append(sibling.getForm(generations="all"))
                else:
                    raise TypeError(f"sibling for {self_name:s} must be Function")
            return instance_functions
        else:
            return parent.getInstanceArguments(specie=specie, name=self_name)

    def getArgumentSubstitutions(self, parent: Function, specie: str) -> Dict[str, Expr]:
        """
        __Purpose__
            Get substitutions for dependent, i.e. which instance argument to substitute into each general argument
        __Inputs__
            parent [Parent]: parent function to retrieve instance arguments from
            specie [str]: species of argument to retrieve
        """
        general_arguments = self.getGeneralArguments(species=specie)
        instance_arguments = self.getInstanceArgumentSubstitutions(parent, specie)
        return dict(zip(general_arguments, instance_arguments))

    def getInstanceArgumentFunction(self, parent: Function) -> Expr:
        """
        __Purpose__
            Get dependent function with instance arguments substituted into general arguments
        __Inputs__
            parent [Parent]: parent function to retrieve instance arguments from
        """
        species = self.getGeneralSpecies()
        function_sub = self.getForm(generations="all")
        for specie in species:
            substitutions = self.getArgumentSubstitutions(parent, specie)
            function_sub = function_sub.subs(substitutions)
        return function_sub


class Independent:
    """
    __Purpose__
        Store properties for a function that does not require input from another function
    """

    def __init__(self) -> None: pass


class Piecewise:
    """
    __Purpose__
        Store information pertaining to piecewise function
    __Attributes__
        functions [list of Function]: function pieces making up piecewise function
        conditions [bool]: conditions under which each function piece is used
    """

    def __init__(self, functions: List[Function], conditions: List[str]) -> None:
        if len(functions) != len(conditions): raise ValueError(
            "each function must have exactly one corresponding condition")
        self.functions = functions
        self.conditions = [eval(condition) for condition in conditions]  # eval() to convert str to bool

    def getConditions(self) -> List[bool]:
        """
        __Purpose__
            Get conditions that determine which function piece to use
        """
        return self.conditions

    def getPieces(self) -> List[Function]:
        """
        __Purpose__
            Get Function objects constituting Piecewise object
        """
        return self.functions

    def getPieceCount(self) -> int:
        """
        __Purpose__
            Get number of pieces constituting piecewise function
        """
        return len(self.functions)

    def getForm(self, generations: Union[int, str] = 1) -> spPiecewise:
        """
        __Purpose__
            Get symbolic piecewise expression for self
        """
        functions = self.getPieces()
        if generations == "all":
            pieces = [function.getForm(generations="all") for function in functions]
        elif generations >= 0:
            pieces = [function.getForm(generations=generations) for function in functions]
        else:
            raise ValueError("generations must be 'all' or some integer greater than or equal to 0")
        conditions = self.getConditions()
        exprconds = [(pieces[i], conditions[i]) for i in range(self.getPieceCount())]
        return spPiecewise(*exprconds)

    def getVariables(self, nested: bool = True) -> List[Symbol]:
        """
        __Purpose__
            Get unique list of variables in function, including pieces
        __Inputs__
            nested [bool]: set True to include variables from children (implicit), False to only include self variables (explicit)
        """
        variables = self.variables
        for function in self.getPieces(): variables.extend(function.getVariables(nested))
        return unique(variables)

    def getParameters(self, nested: bool = True, return_type: Type[Symbol, str] = Symbol) -> Union[
        List[Symbol], List[str]]:
        """
        Get unique list of parameters in function.
        
        :param self: :class:`~Function.NonPiecewise` to retrieve parameters from
        :param nested: set True to include parameters from children (implicit).
            Set False to only include self parameters (explicit).
        :param return_type: class type to return elements in list output as
        """
        parameters = self.parameters
        for function in self.getPieces(): parameters.extend(function.getParameters(nested=nested))
        unique_parameters = unique(parameters)

        if return_type == Symbol:
            return unique_parameters
        elif return_type == str:
            return [str(parameter) for parameter in unique_parameters]
        else:
            raise ValueError("return_type must be Symbol or str")


class NonPiecewise:
    """
    __Purpose__
        Store information pertaining to nonpiecewise (standard) function
    __Attributes__
        functions [list of Function]: function form of object
    """

    def __init__(self, expression: Expr) -> None:
        self.expression = expression

    def getForm(self, parent: Function = None, substitute_dependents: bool = True,
                generations: Union[int, str] = 0) -> Expr:
        """
        __Purpose__
            Get functional form of Function object
        """
        expression = self.expression
        if isinstance(self, Dependent) and isinstance(parent, Function): return self.getInstanceArgumentFunction(parent)
        for child in self.getChildren():
            child_symbol = child.getSymbol()
            if isinstance(child, Dependent):
                if substitute_dependents:
                    child_expression = child.getForm(parent=self)
                else:
                    child_expression = child_symbol
            elif isinstance(child, Independent):
                if generations == "all":
                    child_expression = child.getForm(substitute_dependents=substitute_dependents,
                                                     generations=generations)
                elif generations >= 1:
                    child_expression = child.getForm(substitute_dependents=substitute_dependents,
                                                     generations=generations - 1)
                elif generations == 0:
                    child_expression = child_symbol
                else:
                    raise ValueError("generations must be 'all' or some integer greater than or equal to 0")
            else:
                raise TypeError("child must be of type Dependent xor Independent")
            expression = expression.subs(child_symbol, child_expression)
        return expression

    def getVariables(self, nested: bool = True) -> List[Symbol]:
        """
        __Purpose__
            Get unique list of variables in function
        __Inputs__
            nested [bool]: set True to include variables from children (implicit), False to only include self variables (explicit)
        """
        variables = self.variables
        if nested:
            for child in self.getChildren():
                child_variables = child.getVariables(nested)
                if isinstance(child_variables, Symbol):
                    variables.append(child_variables)
                elif isinstance(child_variables, list):
                    variables.extend(child_variables)
                else:
                    raise RecursiveTypeError(child_variables, Symbol)
        return unique(variables)

    def getParameters(self, nested: bool = True, return_type: Type[Symbol, str] = Symbol) -> Union[
        List[Symbol], List[str]]:
        """
        Get unique list of parameters in function.
        
        :param self: :class:`~Function.NonPiecewise` to retrieve parameters from
        :param nested: set True to include parameters from children (implicit).
            Set False to only include self parameters (explicit).
        :param return_type: class type to return elements in list output as
        """
        parameters = self.parameters
        if nested:
            for child in self.getChildren():
                child_parameters = child.getParameters(nested=nested, return_type=return_type)
                if isinstance(child_parameters, Symbol):
                    parameters.append(child_parameters)
                elif isinstance(child_parameters, list):
                    parameters.extend(child_parameters)
                else:
                    raise RecursiveTypeError(child_parameters, Symbol)
        unique_parameters = unique(parameters)

        if return_type == Symbol:
            return unique_parameters
        elif return_type == str:
            return [str(parameter) for parameter in unique_parameters]
        else:
            raise ValueError("return_type must be Symbol or str")


def FunctionMaster(name: str, function: Union[Expr, List[Function]],
                   inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
                   **kwargs) -> Function:
    """
    __Purpose__
        Get Function object with desired properties
    __Inputs__
        name [str]: name associated with function
        function [sympy.Expr, list of Function]:
            sympy.Expr: form of function (e.g. x**2 in f(x)=x**2) for non-piecewise function
            list of Function: collection of function pieces for piecewise function
        inheritance [tuple of classes]: classes for function to inherit
            Dependent: inherit if function requires input (e.g. variable or parameter input)
            Derivative: inherit if function should be included in ODE of model
            Piecewise: inherit if function is piecewise
            NonPiecewise: inherit if function is non-piecewise
        **kwargs [dict]: arguments to be given to each inherited class's __init__
            e.g. kwargs["Dependent"] holds arguments for Dependent __init__ if function inherits Dependent class
    """

    class FunctionCore(Function, *inheritance):
        """
        __Purpose__
            Object to store various information about a function
            Allows Function object to dynamically inherit necessary classes

        .. seealso:: Function
        """

        def __init__(self, name: str, function: Union[Expr, List[Function]],
                     inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
                     **kwargs) -> None:
            """
            __Purpose__
                Instantiate Function object with necessary class inheritance
            __Inputs__
                cf. FunctionMaster
            """
            Function.__init__(self, name, **kwargs)

            if Derivative in inheritance: Derivative.__init__(self, **kwargs["Derivative"])

            if Dependent in inheritance:
                Dependent.__init__(self, **kwargs["Dependent"])
            elif Independent in inheritance:
                Independent.__init__(self)
            else:
                raise ValueError("Function must inherit either Dependent or Independent")

            if Piecewise in inheritance:
                Piecewise.__init__(self, function, **kwargs["Piecewise"])
            elif NonPiecewise in inheritance:
                NonPiecewise.__init__(self, function)
            else:
                raise ValueError("Function must inherit either Piecewise of NonPiecewise")

    return FunctionCore(name, function, inheritance, **kwargs)


def getFunctionInfo(info: Dict[str, Union[str, List[str], Dict[str, Union[str, List[str]]]]], model: Model = None) -> \
Dict[str, List[str]]:
    """
    __Purpose__
        Get properly formatted dictionary of information to create Function object
    __Inputs__
        info [dict]: collection of information to properly create Function object
            {"properties": [list of str],
            "variables": [str, list of str],
            "parameters": [str, list of str],
            "form": [str],
            "children":
                [{child_name:
                    {"variables": [str, list of str],
                    "parameters": [str, list of str]}
                }]
            "arguments":
                "variables": [str, list of str]
                "parameters": [str, list of str]
                "functions": [str, list of str]
            "variable": [str]
            }
        model [Model]: model to associated Function object with
    """

    def getVariables(info: dict) -> List[Symbol]:
        """
        __Purpose__
            Get variable for function
            Create symbolic variable of each parameter
        __Inputs__
            info [dict]: contains information and properties of function
        """
        return var(info["variables"])

    def getParameters(info: dict) -> List[Symbol]:
        """
        __Purpose__
            Get parameters for function
            Create symbolic variable of each parameter
        __Inputs__
            info [dict]: contains information and properties of function
        """
        return var(info["parameters"])

    def getProperties(info: dict) -> List[str]:
        """
        __Purpose__
            Get collection of properties to give function (e.g. piecewise, dependent)
        __Inputs__
            info [dict]: contains information and properties of function
        """
        return info["properties"]

    def getArguments(info: dict) -> Dict[str, List[Symbol]]:
        """
        __Purpose__
            Get input arguments into function
            Create symbolic variable of each argument
        __Inputs__
            info [dict]: contains information and properties of function
        """
        arguments = info["arguments"]
        return {key: var(arguments[key]) for key in arguments.keys()}

    def getVariable(info: dict) -> Symbol:
        """
        __Purpose__
            Get associated variable for derivative
        __Inputs__
            info [dict]: contains information and properties of function
        """
        return Symbol(info["variable"])

    def getChildren(model: Model, info: dict) -> Dict[str, Dict[str, Union[Function, Union[Symbol, List[Symbol]]]]]:
        """
        __Purpose__
            Get nested functions (a.k.a. children functions), composite functions (e.g. g(x) in f(g(x)))
            Get associated arguments to input into each child function, keeping track of argument type (e.g. parameter, variable)
            Create symbolic variable of each argument
        __Inputs__
            model [Model]: model to retrieve child Function object from
            children_info [dict]:
                keys [str] are name of child function
                values [dict, None] contain instance argument(s) into child function
                    keys [str] are species of argument (e.g. "variables","parameters","functions")
                    values [str, list of str] are argument names
        __Return__
            dict
                keys [str] are name of child function
                values contain information about child function in relation to parent
                    keys: values
                        "function": Function object of child function from model
                        argument_type [str]: arguments [sympy.Symbol, list of sympy.Symbol] of given argument_type to input to child function
        """
        children_info = info["children"]
        children_names = list(children_info.keys())
        var(children_names)

        children_dict = {}
        for child_name in children_names:
            child_info = children_info[child_name]
            if child_info is not None:
                children_dict[child_name] = {argument_type: var(child_info[argument_type]) for
                                             argument_type in child_info.keys()}
            elif child_info is None:
                children_dict[child_name] = {}
            children_dict[child_name]["function"] = model.getFunctions(child_name)
        return children_dict

    kwargs = {}
    if model is not None: kwargs["model"] = model

    info_keys = info.keys()
    if "variables" in info_keys: kwargs["variables"] = getVariables(info)
    if "parameters" in info_keys: kwargs["parameters"] = getParameters(info)
    if "children" in info_keys: kwargs["children"] = getChildren(model, info)

    properties = getProperties(info)
    kwargs["properties"] = properties
    if "Dependent" in properties: kwargs["Dependent"] = {"arguments": getArguments(info)}
    if "Derivative" in properties: kwargs["Derivative"] = {"variable": getVariable(info)}
    if "Piecewise" in properties: kwargs["Piecewise"] = {"conditions": info["conditions"]}
    return kwargs


def createFunction(name: str, function: Union[str, List[str]], properties: Tuple[str] = (), **kwargs) -> None:
    """
    __Purpose__
        Create Function object with desired...
            parent-child associations
            variable-parameter distinctions
            properties

    :param name: name of new function
    :param function: expression (non-piecewise) or collection of function-object pieces (piecewise) for new function object
    :param properties: collection of properties to give new function
    :param kwargs: see FunctionCore
    """
    inheritance = []

    if "Derivative" in properties: inheritance.append(Derivative)

    if "Dependent" in properties:
        inheritance.append(Dependent)
    elif "Independent" in properties:
        inheritance.append(Independent)
    else:
        raise ValueError("Function must have either 'Dependent' or 'Independent' as property")

    if "Piecewise" in properties:
        inheritance.append(Piecewise)
    elif "Piecewise" not in properties:
        inheritance.append(NonPiecewise)

    FunctionMaster(name, function, inheritance=tuple(inheritance), **kwargs)


def createModel(function_ymls: str, parameters_yml: str) -> Model:
    """
    __Purpose__
        Create model from given YML files
    __Inputs__
        functions_yml [str]: name of YML file containing function/equation info
        parameters_yml [str]: name of YML file containing parameters info

    :param function_ymls: name of YML file containing information for function
    :param parameters_yml: name of YML file containing information about parameter values/units
    """
    model = Model()
    model.loadParametersFromFile(parameters_yml)
    model.loadFunctionsFromFiles(function_ymls)
    return model
