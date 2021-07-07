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
from sympy import Symbol, cosh, exp, ln, pi, solve, symbols, var
from sympy.core import function
from sympy.utilities.lambdify import lambdify

import YML
from CustomErrors import RecursiveTypeError
from macros import formatQuantity, unique


class Parameter:
    def __init__(self, name: str, quantity: Quantity, model: Model) -> None:
        self.name = name
        self.quantity = quantity
        self.model = model

    def getName(self) -> str:
        """
        Get name of parameter.

        :param self: `~Function.Parameter` to retrieve name from
        """
        return self.name

    def getQuantity(self) -> Quantity:
        """
        Get quantity (value and unit) for parameter.

        :param self: `~Function.Parameter` to retrieve quantity from
        """
        return self.quantity

    def getModel(self) -> Model:
        """
        Get model that parameter is stored in.

        :param self: `~Function.Parameter` to retrieve model from
        """
        return self.model


class Model:
    """
    __Purpose__
        Store Function objects for given model
    __Attributes__
        functions [list of Function]: stored Function objects
        parameters [dict of metpy.Quantity]: stored parameter values and units
            key [str] is parameter name
            value [metpy.Quantity] contains value and unit
    """

    def __init__(
            self,
            functions: List[Function] = None,
            parameters: Dict[str, Quantity] = None,
    ) -> None:
        self.functions = {}
        self.parameters = {}
        if parameters is not None:
            self.addParameters(parameters)
        if functions is not None:
            self.addFunctions(functions)

    def addParameters(self, quantities: Dict[str, Quantity]) -> None:
        """
        Set/add parameter(s) value and unit for model as Quantity.
        
        :param self: `~Function.Model` to set parameter(s) for
        :param quantities:
            Key is name of parameter.
            Value is quantity containing value and unit for parameter.
        """
        for name, quantity in quantities.items():
            if name in self.getFunctionNames():
                print(f"Overwriting function {name:s} as parameter {name:s}={formatQuantity(quantity)}")
                del self.functions[name]
            elif name in self.getParameterNames():
                print(f"Overwriting parameter {name:s}={formatQuantity(quantity):s} into model")
            self.parameters[name] = Parameter(name, quantity, self)

    def addFunctions(self, functions: Union[Function, List[Function]]) -> None:
        """
        Add Function object(s) to model.
        Set self as model for Function object(s).

        :param self: `~Function.Model` to add function to
        :param functions: function(s) to add to model
        """
        if isinstance(functions, Function):
            if functions not in self.getFunctions():
                name = functions.getName()
                if name in self.getFunctionNames():
                    print(f"Overwriting {name:s}={functions.getForm():} into model")
                    del self.functions[name]
                if functions.isParameter():
                    print(f"Setting model for parameter/function {name:s}={functions.getForm():}")
                elif name in self.getParameterNames():
                    print(f"Overwriting parameter {name:s} as function {name:s}={functions.getForm():}")
                    del self.parameters[name]

                if functions.getModel() is not self:
                    functions.setModel(self)
                if not functions.isParameter():
                    self.functions[name] = functions
        elif isinstance(functions, list):
            for function in functions:
                self.addFunctions(function)
        else:
            raise RecursiveTypeError(functions, Function)

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

    def getParameters(self, names: Union[str, List[str]] = None) -> Union[Quantity, Dict[str, Quantity]]:
        """
        Get parameter quantities stored in model.
        
        __Recursion Base__
            return single parameter: names [str]

        :param self: `~Function.Model` to retrieve parameter(s) from
        :param names: name(s) of parameter(s) to retrieve
        :returns: Quantity for parameter if :paramref:~Function.Model.getParameters.names` is str.
            List of quantities if :paramref:~Function.Model.getParameters.names` is list.
        """
        if isinstance(names, str):
            parameter = self.parameters[names]
            quantity = parameter.getQuantity()
            return quantity
        elif isinstance(names, Symbol):
            return self.getParameters(names=str(names))
        elif isinstance(names, list):
            return {str(name): self.getParameters(names=name) for name in names}
        elif names is None:
            return self.getParameters(names=self.getParameterNames())
        else:
            raise RecursiveTypeError(names, [str, Symbol])

    def getFunctions(
            self, names: Union[str, List[str]] = None, filter_type: Type[Function] = None
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
            function = self.functions[names]
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

    def loadFunctionsFromFiles(self, ymls: Union[str, List[str]]) -> None:
        """
        Add functions to model by parsing through YML file.

        :param self: `~Function.Model` to add function(s) to
        :param ymls: name(s) of YML file(s) to retrieve function info from
        """
        if isinstance(ymls, str):
            ymls = [ymls]

        for yml in ymls:
            generateFunctionsFromFile(yml, model=self)

    def loadParametersFromFile(self, ymls: Union[str, List[str]]) -> None:
        """
        Add parameters to model by parsing through YML file.

        :param self: `~Function.Model` to add function(s) to
        :param ymls: name(s) of YML file(s) to retrieve parameter info from
        """
        if isinstance(ymls, str):
            ymls = [ymls]

        for yml in ymls:
            parameters = YML.readParameters(yml)
            self.addParameters(parameters)

    def saveParametersToFile(self, filename: str) -> None:
        """
        Save parameters stored in model into YML file for future retrieval.
        
        :param self: :class:`~Function.Model` to retrieve parameters from
        :param filename: name of file to save parameters into
        """
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
        time_evolution_types = {
            str(derivative.getVariable()): derivative.getTimeEvolutionType()
            for derivative in derivatives
        }
        file = open(filename, 'w')
        yaml.dump(time_evolution_types, file)

    def getDerivatives(self, time_evolution_types: Union[str, List[str]] = None) -> List[Derivative]:
        """
        __Purpose__
            Get stored derivatives of given time-evolution type(s)
        __Recursion Base__
            get derivatives of single time-evolution type: time_evolution_types [str]
            get all derivatives: time_evolution_types [None]

        :param self: `~Function.Model` to retrieve derivative(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        """
        if isinstance(time_evolution_types, str):
            filtered_derivatives = [
                derivative
                for derivative in self.getFunctions(filter_type=Derivative)
                if derivative.getTimeEvolutionType() == time_evolution_types
            ]
            return filtered_derivatives
        elif isinstance(time_evolution_types, list):
            derivatives = [
                derivative
                for time_evolution_type in time_evolution_types
                for derivative in self.getDerivatives(time_evolution_types=time_evolution_type)
            ]
            return derivatives
        elif time_evolution_types is None:
            # noinspection PyTypeChecker
            derivatives: List[Derivative] = self.getFunctions(filter_type=Derivative)
            return derivatives
        else:
            raise RecursiveTypeError(time_evolution_types)

    def getEquilibriumSolutions(
            self,
            names: Union[str, List[str]] = None,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_constants: bool = True,
            substitute_functions: bool = True
    ) -> Dict[Symbol, Expr]:
        """
       Get substitutions to substitute equilibrium variables into non-equilibrium derivatives.

        :param self: `~Function.Model` to solve for equilibrium solution(s) in
        :param names: name(s) of function(s) to solve for simulatenous equilibrium of
        :param substitute_parameters: set True to substitute numerical values in for all parameters.
            Set false otherwise
        :param skip_parameters: name(s) of parameter(s) to skip substitution for.
            Only called if :paramref:`~Function.Model.getEquilibriumSolutions.substitute_parameters` is set True.
        :param substitute_constants: set True to substitute numerical values in for all constant derivative.
            Set False to leave them as symbolic variables.
        :param substitute_functions: set True to substitute corresponding functions in for all function-type derivatives.
            Set False to leave them as symbolic variables.
        :returns: Dictionary of substitutions.
            Key is symbolic variable to replace.
            Value is equilibrium expression to substitute into variable.
        """
        if skip_parameters is None:
            skip_parameters = []

        if isinstance(names, (str, list)):
            equilibria = self.getFunctions(names=names)
            if not all([isinstance(function, Derivative) for function in equilibria]):
                raise TypeError("names must correspond to Derivative stored in Model")
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
            if set(bk_probs).issubset(set(equilibrium_variables)):
                equilibrium_derivatives.append(sum(bk_probs) - 1)
            solutions = solve(equilibrium_derivatives, equilibrium_variables)

            substitutions = {}
            if substitute_functions:
                kwargs = {
                    "substitute_parameters": substitute_parameters,
                    "skip_parameters": skip_parameters,
                    "substitute_constants": substitute_constants
                }
                substitutions.update(self.getFunctionSubstitutions(**kwargs))
            if substitute_parameters:
                parameter_names = unique(
                    [
                        parameter
                        for equilibrium in equilibria
                        for parameter in
                        equilibrium.getFreeSymbols(species="Parameter", generations="all", return_type=str)
                        if parameter not in skip_parameters
                    ]
                )
                substitutions.update(self.getParameterSubstitutions(parameter_names))
            if substitute_constants:
                substitutions.update(self.getConstantSubstitutions())

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
        __Purpose__
            Get equilibrium function corresponding to derivative

        :param self: `~Function.Model` to solve for equilibrium solution in
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

    def getFunctionSubstitutions(
            self,
            names: Union[str, List[str]] = None,
            substitute_parameters: bool = True,
            skip_parameters: Union[str, List[str]] = None,
            substitute_constants: bool = True
    ) -> Dict[Symbol, Expr]:
        """
        :param self: `~Function.Model` to retrieve functions from
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
            forms = [function.getForm(generations="all") for function in functions]
            variables = [Symbol(name) for name in names]

            substitutions = {}
            if substitute_parameters:
                parameter_names = unique(
                    [
                        parameter_name
                        for function in functions
                        for parameter_name in
                        function.getFreeSymbols(species="Parameter", generations="all", return_type=str)
                        if parameter_name not in skip_parameters
                    ]
                )
                substitutions.update(self.getParameterSubstitutions(parameter_names))
            if substitute_constants:
                substitutions.update(self.getConstantSubstitutions())

            forms = [form.subs(substitutions) for form in forms]
            function_substitutions = {variables[i]: forms[i] for i in range(variable_count)}
            return function_substitutions

    def getConstantSubstitutions(self, names: Union[str, List[str]] = None) -> Dict[Symbol, Expr]:
        """
        __Purpose__
            Get substitutions to substitute constant intitial condition into given variable(s)
        __Return__
            dict:
                key [sympy.Symbol] is variable to substitute into
                value [sympy.Expr] is constant to substitute into variable

        :param self: `~Function.Model` to retrieve constant derivative(s) from
        :param names: name(s) of constant derivative(s) to substitute numerical constants in for
        """
        if isinstance(names, (str, list)):
            constant_functions = self.getFunctions(names=names)
            for function in constant_functions:
                if not isinstance(function, Derivative):
                    raise TypeError("names must correspond to Derivative stored in Model")
        elif names is None:
            constant_functions = self.getDerivatives(time_evolution_types="Constant")
        else:
            raise RecursiveTypeError(names)
        constant_count = len(constant_functions)
        if constant_count == 0:
            return {}
        elif constant_count >= 1:
            substitutions = {function.getVariable(): function.getInitialCondition() for function in constant_functions}
            return substitutions

    def getParameterSubstitutions(
            self, names: Union[str, List[str]] = None, skip_parameters: Union[str, List[str]] = None
    ) -> Dict[Symbol, Expr]:
        """
        Substitute parameters into function from model.

        :param names: name(s) of parameter to include in substitutions.
            Defaults to all parameters in model.
        :param skip_parameters: name(s) of parameter(s) to skip substitution for
        """
        if skip_parameters is None:
            skip_parameters = []
        quantities = self.getParameters(names=names)
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

        :param self: `~Function.Model` to retrieve derivative(s) from
        :param names: name(s) of variable(s) ordered in same order derivatives will be returned
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
        if skip_parameters is None:
            skip_parameters = []

        use_memory = Function.usingMemory()
        Function.clearMemory()
        Function.setUseMemory(False)

        variable_substitutions = {}
        if substitute_equilibria:
            kwargs = {
                "substitute_parameters": substitute_parameters,
                "substitute_constants": substitute_constants,
                "skip_parameters": skip_parameters
            }
            equilibrium_solutions = self.getEquilibriumSolutions(**kwargs)
            variable_substitutions.update(equilibrium_solutions)
        if substitute_constants:
            variable_substitutions.update(self.getConstantSubstitutions())
        if substitute_functions:
            kwargs = {
                "substitute_parameters": substitute_parameters,
                "substitute_constants": substitute_constants,
                "skip_parameters": skip_parameters
            }
            variable_substitutions.update(self.getFunctionSubstitutions(**kwargs))

        kwargs = {
            "skip_parameters": skip_parameters
        }
        parameter_substitutions = self.getParameterSubstitutions(**kwargs) if substitute_parameters else {}

        if names is None:
            temporal_derivatives = self.getDerivatives(time_evolution_types="Temporal")
            names = [str(derivative.getVariable()) for derivative in temporal_derivatives]
        else:
            temporal_derivatives = self.getDerivativesFromVariableNames(names=names)

        derivative_vector = []
        for derivative in temporal_derivatives:
            form = derivative.getForm(generations="all")

            derivative_variables = derivative.getFreeSymbols(species="Variable", generations="all")
            variable_substitution = {
                variable: substitution
                for variable, substitution in variable_substitutions.items()
                if variable in derivative_variables
            }
            derivative_parameters = derivative.getFreeSymbols(species="Parameter", generations="all")
            parameter_substitution = {
                parameter: value
                for parameter, value in parameter_substitutions.items()
                if parameter in derivative_parameters
            }
            form = form.subs({**variable_substitution, **parameter_substitution})

            derivative_vector.append(form)

        Function.setUseMemory(use_memory)
        if lambdified:
            derivative_vector = lambdify((Symbol('t'), tuple(names)), derivative_vector)
        if substitute_equilibria:
            return derivative_vector, equilibrium_solutions
        else:
            return derivative_vector

    def getInitialValues(
            self,
            names: Union[str, List[str]] = None,
            return_type: Type[dict, list, ndarray] = dict,
            initial_values: Dict[Symbol, float] = None
    ) -> Union[float, List[float], ndarray, dict]:
        """
        Get initial values for variables in model.
        
        :param self: `~Function.Model` to retrieve derivatives from
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
                **initial_constant_substitutions, **self.getConstantSubstitutions(), **self.getParameterSubstitutions()
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
            solutions = {equilibrium_variables[i]: roots.x[i] for i in range(variable_count)}

            initial_values = {**initial_constant_substitutions, **solutions}

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

    def getVariables(
            self, time_evolution_types: Union[str, List[str]] = None, return_type: Type[Symbol, str] = Symbol
    ) -> Union[List[Symbol], List[str]]:
        """
        Get variables stored in model in same order as derivatives
        
        __Recursion Base__
            get symbolic variable associated with single derivative: names [str]

        :param self: `~Function.Model` to retrieve derivative variable(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        :param return_type: class type to return elements in list output as
        """
        if isinstance(time_evolution_types, str):
            derivatives = self.getDerivatives(time_evolution_types=time_evolution_types)
            variables = [derivative.getVariable() for derivative in derivatives]
        elif isinstance(time_evolution_types, list):
            variables = [
                self.getVariables(time_evolution_types=time_evolution_type)
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

    def getDerivativesFromVariableNames(
            self, names: Union[str, Symbol, List[Union[str, Symbol]]]
    ) -> Union[Derivative, List[Derivative]]:
        """
        Get derivative corresponding to variable name.

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        :param names: name(s) of variable(s) associated with derivative(s)
        """
        if isinstance(names, (str, Symbol)):
            derivatives = self.getDerivatives()
            for derivative in derivatives:
                if derivative.getVariable(return_type=str) == str(names):
                    return derivative
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

    def __init__(self, children: Dict[str, Dict[str, Union[Symbol, List[Symbol]]]]) -> None:
        self.instance_arguments = {}

        if children is not None:
            if isinstance(children, Child):
                children = [children]
            self.addChildren(children)

    def addChildren(self, children: Dict[str, Dict[str, Union[Symbol, List[Symbol]]]]) -> \
            Dict[str, Dict[str, List[Symbol]]]:
        """
        Add information to reference children functions of parent.

        :param self: :class:`~Function.Parent` to add info into
        :param children: dictionary of information to reference children.
            Key is name of child.
            Value is dictionary to pass into :paramref:`~Function.Parent.addChild.arguments`
        :returns: Dictionary of information to reference children.
            Key is name of child.
            Value is dictionary of information output from :meth:`~Function.Parent.addChild`
        """
        return {name: self.addChild(name, arguments) for name, arguments in children.items()}

    def addChild(self, name: str, arguments: Dict[str, List[Symbol]]) -> Dict[str, List[Symbol]]:
        """
        Add information to reference child function of parent.

        :param self: :class:`~Function.Parent` to add info into
        :param name: name of child to add into parent
        :param arguments: dictionary of information to reference child.
            Key is species of instance argument.
            Value is symbol(s) for this species of argument.
        :returns: Dictionary of information to reference child.
            Key is species of instance argument from parent.
            Value is symbol(s) for this species of argument.
        """
        new_child = {
            specie: [instance_argument]
            if isinstance((instance_argument := arguments[specie]), Symbol)
            else instance_argument
            for specie in arguments.keys()
        }
        self.instance_arguments[name] = new_child
        return new_child

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
            self: Function
            children = self.getModel().getFunctions(names=names)
            return children
        elif isinstance(names, list):
            return [self.getChildren(names=name) for name in names]
        elif names is None:
            return self.getChildren(names=self.getChildrenNames())
        else:
            raise RecursiveTypeError(names)

    def getChildrenNames(self) -> Union[str, List[str]]:
        """
        __Purpose__
            Get names of stored parents in order added
        __Recursion Base__
            retrieve name of single parent: functions [Function]
            retrieve names of all parents: functions [None]

        :param self: parent to retrieve child name(s) from
        :param functions: function(s) to retrieve name(s) from parent
        """
        self: Function
        free_symbols = self.getFreeSymbols(generations=0, substitute_dependents=False)
        free_symbol_names = [str(free_symbol) for free_symbol in free_symbols]
        dependent_children_names = list(self.instance_arguments.keys())

        children_names = dependent_children_names
        if isinstance(self, Independent):
            model_function_names = self.getModel().getFunctionNames()
            model_children_names = [
                name
                for name in free_symbol_names
                if name in model_function_names
            ]
            children_names.extend(model_children_names)
        return unique(children_names)

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
        pass

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
            parents = self.getModel().getFunctions(names=names)
            return parents
        elif isinstance(names, list):
            return [self.getParents(names=name) for name in names]
        elif names is None:
            return self.getParents(names=self.getParentNames())
        else:
            raise RecursiveTypeError(names)

    def getParentNames(self) -> Union[str, List[str]]:
        """
        __Purpose__
            Get names of stored parents in order added
        __Recursion Base__
            retrieve name of single parent: functions [Function]
            retrieve names of all parents: functions [None]

        :param self: child to retrieve parent name(s) from
        :param functions: parent(s) to retrieve name(s) from child
        """
        name = self.getName()
        parents = [
            function.getName()
            for function in self.getModel().getFunctions()
            if name in function.getChildrenNames()
        ]
        return parents


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

    use_memory = False
    functions = []

    def __init__(
            self,
            name: str,
            properties: List[str],
            children: Union[str, List[str]] = None,
            model: Model = None,
            **kwargs
    ) -> None:
        Function.functions.append(self)
        self.form = None
        self.name = name
        self.is_parameter = "Parameter" in properties

        self.model = None
        self.setModel(model)

        Parent.__init__(self, children)
        Child.__init__(self)

    @staticmethod
    def setUseMemory(use: bool) -> None:
        Function.use_memory = use

    @staticmethod
    def usingMemory() -> bool:
        return Function.use_memory

    @staticmethod
    def clearMemory(functions: List[Function] = None) -> None:
        if functions is None:
            functions = Function.functions
        for function in functions:
            function.clearForm()

    def clearForm(self) -> None:
        self.form = None

    def isParameter(self):
        """
        Determine whether function is equal to parameter.

        :param self: `~Function.Function` to determine for
        """
        return self.is_parameter

    def getName(self) -> str:
        """
        __Purpose__
            Get name of function

        :param self: `~Function.Function` to retrieve name of
        """
        return self.name

    def getSymbol(self) -> Symbol:
        """
        __Purpose__
            Get symbolic variable of function

        :param self: `~Function.Function` to retrieve symbolic variable of
        """
        return Symbol(self.getName())

    def getFreeSymbols(
            self, species: str = None, return_type: Type[Union[str, Symbol]] = Symbol, **kwargs
    ) -> Union[List[str], List[Symbol]]:
        """
        Get symbols in expression for function.
        
        :param self: :class:`~Function.Function` to retrieve symbols from
        :param species: species of free symbol to retrieve, acts as filter.
            Can be "Parameter", "Variable".
            Defaults to all free symbols.
        :param return_type: class type of output.
            Must be either sympy.Symbol or str.
        :param kwargs: additional arguments to pass into :meth:`~Function.Piecewise.getForm` or into :meth:`~Function.NonPiecewise.getForm`
        """
        free_symbols = list(self.getForm(**kwargs).free_symbols)
        if species is None:
            free_symbol_names = [str(free_symbol) for free_symbol in free_symbols]
        else:
            unfiltered_names = [str(free_symbol) for free_symbol in free_symbols]

            model = self.getModel()
            if species == "Parameter":
                model_symbol_names = model.getParameterNames()
            elif species == "Variable":
                model_symbol_names = model.getVariables(return_type=str)
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
            return [Symbol(free_symbol_name) for free_symbol_name in free_symbol_names]
        else:
            raise ValueError("return_type must be Symbol or str")

    def getForm(self, **kwargs):
        """
        Get expression for function, dependent on inherited classes.
        
        :param self: :class:`~Function.Function` to retrieve form from
        :param kwargs: additional arguments to pass into :meth:`~Function.Piecewise.getForm` or :meth:`~Function.NonPiecewise.getForm`
        """
        if not self.usingMemory() or self.form is None or isinstance(self, Dependent):
            if isinstance(self, Piecewise):
                form = Piecewise.getForm(self, **kwargs)
            elif isinstance(self, NonPiecewise):
                form = NonPiecewise.getForm(self, **kwargs)
            else:
                raise TypeError(f"function {self.getName():s} must inherit either Piecewise or NonPiecewise")
            self.form = form
            return form
        elif self.usingMemory() and isinstance(self.form, Expr):
            return self.form
        else:
            raise ValueError(f"improper form in memory for {self.getName():s}")

    def setModel(self, model: Model) -> None:
        """
        Set model for function.
        Add function to model if not already in done.

        :param self: :class:`~Function.Function` to set model with
        :param model: new model to associated with function
        """
        self.model = model
        if isinstance(model, Model):
            if self not in model.getFunctions():
                model.addFunctions(self)
        elif model is None:
            pass
        else:
            raise TypeError("model input must be Model")

    def getModel(self) -> Model:
        """
        Get model that function is stored within.

        :param self: :class:`~Function.Function` to retrieve model of
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

    def __init__(
            self, variable: Symbol, time_evolution_type: str = "Temporal", initial_condition: float = 0
    ) -> None:
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
                    if not isinstance(argument, Symbol):
                        raise TypeError("argument must be sympy.Symbol")
                self.general_arguments[species] = arguments
            elif arguments is None:
                self.general_arguments[species] = []
            else:
                raise RecursiveTypeError(arguments, Symbol)
        elif species is None:
            if isinstance(arguments, dict):
                for specie, argument in arguments.items():
                    self.setGeneralArguments(argument, specie)
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

    def getGeneralArguments(
            self, species: Union[str, List[str]] = None, nested: bool = False
    ) -> Union[List[Symbol], Dict[str, List[Symbol]]]:
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
                return {specie: self.getGeneralArguments(specie) for specie in self.getGeneralSpecies()}
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
        if len(functions) != len(conditions):
            raise ValueError(
                "each function must have exactly one corresponding condition"
            )
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

    def getForm(self, generations: Union[int, str] = 0) -> spPiecewise:
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


class NonPiecewise:
    """
    __Purpose__
        Store information pertaining to nonpiecewise (standard) function
    __Attributes__
        functions [list of Function]: function form of object
    """

    def __init__(self, expression: Expr) -> None:
        self.expression = expression

    def getForm(
            self, parent: Function = None, substitute_dependents: bool = True, generations: Union[int, str] = 0
    ) -> Expr:
        """
        __Purpose__
            Get functional form of Function object
        """
        if isinstance(self, Dependent) and isinstance(parent, Independent):
            return self.getInstanceArgumentFunction(parent)

        expression = self.expression
        if generations == 0:
            return expression

        self: Function
        for child in self.getChildren():
            child_symbol = child.getSymbol()
            if isinstance(child, Dependent):
                child: Function
                if substitute_dependents:
                    child_expression = child.getForm(parent=self)
                else:
                    child_expression = child_symbol
            elif isinstance(child, Independent):
                child: Function
                if generations == "all":
                    child_expression = child.getForm(
                        substitute_dependents=substitute_dependents,
                        generations=generations
                    )
                elif generations >= 1:
                    child_expression = child.getForm(
                        substitute_dependents=substitute_dependents,
                        generations=generations - 1
                    )
                elif generations == 0:
                    child_expression = child_symbol
                else:
                    raise ValueError("generations must be 'all' or some integer greater than or equal to 0")
            else:
                raise TypeError("child must be of type Dependent xor Independent")
            expression = expression.subs(child_symbol, child_expression)
        return expression


def FunctionMaster(
        name: str,
        function: Union[Expr, List[Function]],
        inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
        **kwargs
) -> Function:
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

        def __init__(
                self,
                name: str,
                function: Union[Expr, List[Function]],
                inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
                **kwargs
        ) -> None:
            """
            __Purpose__
                Instantiate Function object with necessary class inheritance
            __Inputs__
                cf. FunctionMaster
            """
            if Derivative in inheritance:
                Derivative.__init__(self, **kwargs["Derivative"])

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

            Function.__init__(self, name, **kwargs)

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
                        argument_type [str]: arguments [sympy.Symbol, list of sympy.Symbol] of given argument_type to input to child function
        """
        children_info = info["children"]
        children_names = list(children_info.keys())
        var(children_names)

        children_dict = {}
        for child_name in children_names:
            child_info = children_info[child_name]
            if child_info is not None:
                children_dict[child_name] = {
                    argument_type: var(child_info[argument_type])
                    for argument_type in child_info.keys()
                }
            elif child_info is None:
                children_dict[child_name] = {}
        return children_dict

    kwargs = {}
    if model is not None:
        kwargs["model"] = model

    info_keys = info.keys()
    if "variables" in info_keys:
        kwargs["variables"] = getVariables(info)
    if "parameters" in info_keys:
        kwargs["parameters"] = getParameters(info)
    if "children" in info_keys:
        kwargs["children"] = getChildren(model, info)

    properties = getProperties(info)
    kwargs["properties"] = properties
    if "Dependent" in properties:
        kwargs["Dependent"] = {
            "arguments": getArguments(info)
        }
    if "Derivative" in properties:
        kwargs["Derivative"] = {
            "variable": getVariable(info)
        }
    if "Piecewise" in properties:
        kwargs["Piecewise"] = {
            "conditions": info["conditions"]
        }
    return kwargs


def generateFunctionsFromFile(filename: str, **kwargs) -> List[Function]:
    file = yaml.load(open(filename), Loader=yaml.Loader)
    return [generateFunction(name, file[name], **kwargs) for name in file.keys()]


def getInheritance(properties: List[str], name: str = '') -> List[Type[Function]]:
    """
    Get classes that function must inherit from properties.

    :param properties: properties to determine inheritance
    :param name: name of function, used when error is thrown
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


def generateFunction(name: str, info: Dict[str, dict], model: Model = None) -> Function:
    """
    Generate Function object.
    
    This includes
        #. parent-child associations
        #. variable-parameter distinctions
        #. properties

    :param name: name of function to generate
    :param info: dictionary of information needed to generate function
    :param model: :class:`~Function.Model` to add function into
    :returns: Generated function
    """
    kwargs = getFunctionInfo(info, model=model)

    info_keys = info.keys()
    if "form" in info_keys:
        form = eval(info["form"])
    elif "pieces" in info_keys:
        form = [model.getFunctions(names=piece_name) for piece_name in info["pieces"]]
    else:
        raise ValueError("info from functions_yml file must contain either form or pieces")

    inheritance = getInheritance(kwargs["properties"])

    return FunctionMaster(name, form, inheritance=tuple(inheritance), **kwargs)


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
