from __future__ import annotations

import warnings
from collections.abc import KeysView
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import yaml
from metpy.units import units
from numpy import ndarray
from pint import Quantity
from scipy import optimize
from sympy import Expr
from sympy import Piecewise as spPiecewise
# noinspection PyUnresolvedReferences
from sympy import Symbol, cosh, exp, ln, pi, solve, symbols, var
from sympy.core import function
from sympy.utilities.lambdify import lambdify

from CustomErrors import RecursiveTypeError
from macros import formatQuantity, recursiveMethod, unique


class PaperQuantity:
    """
    Stores information about objects generated from files, for models.

    :ivar name: name of object
    :ivar filestem: stem of file where object was loaded from
    :ivar model: model that contains object
    """

    def __init__(self, name: str, model: Model, filestem: str = None) -> None:
        """
        Constructor for :class:`~Function.PaperQuantity`.

        :param name: name of object
        :param model: model that contains object
        :param filestem: stem of file where object was loaded from
        """
        self.name = name
        self.filestem = filestem
        self.model = model

    def getName(self, return_type: Type[str, Symbol] = str) -> str:
        """
        Get name of object.

        :param self: `~Function.PaperQuantity` to retrieve name from
        :param return_type: return type of output.
            Must be str or Symbol.
        """
        if return_type == str:
            return self.name
        elif return_type == Symbol:
            return Symbol(self.name)
        else:
            raise ValueError("invalid return type")

    def getStem(self) -> str:
        """
        Get filestem for file that generated object.

        :param self: `~Function.PaperQuantity` to retrieve filestem from
        """
        return self.filestem

    def getModel(self) -> Model:
        """
        Get :class:`~Function.Model` that contains object.

        :param self: `~Function.PaperQuantity` to retrieve model from
        """
        return self.model

    def setModel(self, model: Model) -> None:
        """
        Set :class:`~Function.Model` that contains object.

        :param self: `~Function.PaperQuantity` to set model for
        :param model: model to set for object
        """
        self.model = model


class Parameter(PaperQuantity):
    """
    Store info pertinent to generate/simulate parameter.

    :ivar name: name of parameter
    :ivar quantity: quantity containing value and unit for parameter
    :ivar model: model that parameter is stored in
    """

    def __init__(self, name: str, quantity: Quantity, model: Model = None, filestem: str = None) -> None:
        """
        Constructor for :class:`~Function.Parameter`.

        :param name: name of parameter
        :param quantity: quantity containing value and unit for parameter
        :param model: model that parameter is stored in
        """
        super().__init__(name, model=model, filestem=filestem)
        self.name = name
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
        model = self.getModel()
        model_functions = model.getFunctions()
        symbol = Symbol(self.getName())
        functions = [
            function_object
            for function_object in model_functions
            if symbol in function_object.getFreeSymbols(species="Parameter", **kwargs)
        ]
        return functions


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
            functions: List[Function] = None,
            parameters: List[Parameter] = None,
    ) -> None:
        """
        Constructor for :class:`~Function.Model`.
        
        :param functions: functions to initially include in model
        :param parameters: parameters to initially include in model
        """
        self.functions = {}
        self.parameters = {}
        if parameters is not None:
            self.addPaperQuantities(parameters)
        if functions is not None:
            self.addPaperQuantities(functions)

    def addPaperQuantities(self, quantity_objects: Union[PaperQuantity, List[PaperQuantity]]) -> None:
        """
        Add Function object(s) to model.
        Set self as model for Function object(s).

        :param self: :class:`~Function.Model` to add function to
        :param quantity_objects: function(s) to add to model
        """
        if isinstance(quantity_objects, Function):
            if quantity_objects not in self.getFunctions():
                name = quantity_objects.getName()
                if name in self.getFunctionNames():
                    print(f"Overwriting {name:s}={quantity_objects.getExpression():} into model")
                    del self.functions[name]
                if quantity_objects.isParameter():
                    print(f"Overwriting function {name:s}={quantity_objects.getExpression():} as parameter")
                elif name in self.getParameterNames():
                    print(f"Overwriting parameter {name:s} as function {name:s}={quantity_objects.getExpression():}")
                    del self.parameters[name]

                if quantity_objects.getModel() is not self:
                    quantity_objects.setModel(self)
                if not quantity_objects.isParameter():
                    self.functions[name] = quantity_objects
        elif isinstance(quantity_objects, Parameter):
            name = quantity_objects.getName()
            quantity = quantity_objects.getQuantity()
            if name in self.getFunctionNames():
                print(f"Overwriting function {name:s} as parameter {name:s}={formatQuantity(quantity)}")
                del self.functions[name]
            elif name in self.getParameterNames():
                print(f"Overwriting parameter {name:s}={formatQuantity(quantity):s} into model")
            self.parameters[name] = Parameter(name, quantity, self)
        elif isinstance(quantity_objects, list):
            for quantity_object in quantity_objects:
                self.addPaperQuantities(quantity_object)
        else:
            raise RecursiveTypeError(quantity_objects, [Function, Parameter])

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

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def getParameterQuantites(self, names: Union[str, List[str]] = None) -> Union[Quantity, Dict[str, Quantity]]:
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

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": dict,
            "default_args": self.getParameterNames()
        }
        return recursiveMethod(**kwargs)

    def getFunctions(
            self,
            names: Union[str, List[str]] = None,
            filter_type: Type[Derivative, Dependent, Independent, Piecewise, NonPiecewise] = None
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

    def loadFunctionsFromFiles(self, ymls: Union[str, List[str]]) -> None:
        """
        Add functions to model by parsing through YML file.

        :param self: :class:`~Function.Model` to add function(s) to
        :param ymls: name(s) of YML file(s) to retrieve function info from
        """
        if isinstance(ymls, str):
            ymls = [ymls]

        for yml in ymls:
            generateFunctionsFromFile(yml, model=self)

    def loadParametersFromFile(self, ymls: Union[str, List[str]]) -> None:
        """
        Add parameters to model by parsing through YML file.

        :param self: :class:`~Function.Model` to add function(s) to
        :param ymls: name(s) of YML file(s) to retrieve parameter info from
        """
        if isinstance(ymls, str):
            ymls = [ymls]

        for yml in ymls:
            generateParametersFromFile(yml, model=self)

    def saveParametersToFile(self, filename: str) -> None:
        """
        Save parameters stored in model into YML file for future retrieval.
        
        :param self: :class:`~Function.Model` to retrieve parameters from
        :param filename: name of file to save parameters into
        """
        parameters = self.getParameterQuantites()
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
        """
        Save time-evolution types stored in model into YML file for future retrieval.

        :param self: :class:`~Function.Model` to retrieve time-evolution types from
        :param filename: name of file to save time-evolution types into
        """
        derivatives = self.getDerivatives()
        time_evolution_types = {
            str(derivative.getVariable()): derivative.getTimeEvolutionType()
            for derivative in derivatives
        }
        file = open(filename, 'w')
        yaml.dump(time_evolution_types, file)

    def getDerivatives(self, time_evolution_types: Union[str, List[str]] = None) -> List[Derivative]:
        """
        Get stored derivatives of given time-evolution type(s).
        
        __Recursion Base__
            get derivatives of single time-evolution type: time_evolution_types [str]
            get all derivatives: time_evolution_types [None]

        :param self: :class:`~Function.Model` to retrieve derivative(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as an optional filter
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

        if isinstance(names, (str, list)):
            equilibria = self.getFunctions(names=names)
            if not all([isinstance(function_object, Derivative) for function_object in equilibria]):
                raise TypeError("names must correspond to Derivative stored in Model")
        elif names is None:
            equilibria = self.getDerivatives(time_evolution_types="Equilibrium")
        else:
            raise RecursiveTypeError(names)

        equilibrium_count = len(equilibria)
        if equilibrium_count == 0:
            return {}
        elif equilibrium_count >= 1:
            equilibrium_variables = list(map(Derivative.getVariable, equilibria))
            equilibrium_derivatives = [equilibrium.getExpression(generations="all") for equilibrium in equilibria]

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
        function_object = self.getDerivativesFromVariableNames(names=name)
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
            getExpression = partial(Function.getExpression, generations="all")
            expressions = list(map(getExpression, functions))
            variables = list(map(Symbol, names))

            substitutions = {}
            if substitute_parameters:
                parameter_names = unique(
                    [
                        parameter_name
                        for function_object in functions
                        for parameter_name in
                        function_object.getFreeSymbols(species="Parameter", generations="all", return_type=str)
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

        :param self: :class:`~Function.Model` to retrieve constant derivative(s) from
        :param names: name(s) of constant derivative(s) to substitute numerical constants in for
        """
        if isinstance(names, (str, list)):
            constant_functions = self.getFunctions(names=names)
            for function_object in constant_functions:
                if not isinstance(function_object, Derivative):
                    raise TypeError("names must correspond to Derivative stored in Model")
        elif names is None:
            constant_functions = self.getDerivatives(time_evolution_types="Constant")
        else:
            raise RecursiveTypeError(names)
        constant_count = len(constant_functions)
        if constant_count == 0:
            return {}
        elif constant_count >= 1:
            substitutions = {
                function_object.getVariable():
                    function_object.getInitialCondition()
                for function_object in constant_functions
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
        quantities = self.getParameterQuantites(names=names)
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
        :param substitute_functions: set True to substitute corresponding expresions for function-type derivatives.
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
            getVariable = partial(Derivative.getVariable, return_type=str)
            names = list(map(getVariable, temporal_derivatives))
        else:
            temporal_derivatives = self.getDerivativesFromVariableNames(names=names)

        derivative_vector = []
        for derivative in temporal_derivatives:
            derivative: Union[Derivative, Function]
            expression = derivative.getExpression(generations="all")

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
            expression = expression.subs({**variable_substitution, **parameter_substitution})

            derivative_vector.append(expression)

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
        
        :param self: :class:`~Function.Model` to retrieve derivatives from
        :param names: name(s) of variables to retrieve values for
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
                derivative: Union[Derivative, Function]
                if derivative.getInitialCondition() == "Equilibrium":
                    variables_append(derivative.getVariable())
                    equations_append(derivative.getExpression(generations="all").subs(substitutions))

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
        Get variables stored in model.
        
        __Recursion Base__
            get symbolic variable associated with single derivative: names [str]

        :param self: :class:`~Function.Model` to retrieve derivative variable(s) from
        :param time_evolution_types: only retrieve derivatives of this type(s), acts as a filter
        :param return_type: class type to return elements in list output as
        """
        if isinstance(time_evolution_types, str):
            derivatives = self.getDerivatives(time_evolution_types=time_evolution_types)
            variables = list(map(Derivative.getVariable, derivatives))
        elif isinstance(time_evolution_types, list):
            variables = [
                self.getVariables(time_evolution_types=time_evolution_type)
                for time_evolution_type in time_evolution_types
            ]
        elif time_evolution_types is None:
            derivatives = self.getDerivatives()
            variables = list(map(Derivative.getVariable, derivatives))
        else:
            raise RecursiveTypeError(time_evolution_types)

        if return_type == Symbol:
            return variables
        elif return_type == str:
            return list(map(str, variables))
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

        def get(name: str) -> Derivative:
            """Base method for :meth:`~Function.Model.getDerivativesFromVariableNames`"""
            for derivative in self.getDerivatives():
                if derivative.getVariable(return_type=str) == str(name):
                    return derivative
            raise ValueError("names input must correspond to some Derivative stored in Model")

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": (str, Symbol),
            "output_type": list
        }
        return recursiveMethod(**kwargs)


class Parent:
    """
    Stores properties for Function qua parent.
    
    :ivar instance_arguments: 2-level dictionary of arguments for children functions.
        Firsy key is name of child function.
        Second key is specie of instance arguments.
        Value is list of symbolic arguments to substitute into child.
        Only used for :class:`~Function.Dependent` children.
    """

    def __init__(self, children: Dict[str, Dict[str, Union[Symbol, List[Symbol]]]]) -> None:
        """
        Constructor for :class:`~Function.Parent`.
        
        :param children: 2-level dictionary of info for children function.
            First key is name of child.
            Second key is specie of instance arguments for child.
            Value is symbols for instance arguments.
        """
        self.instance_arguments = {}

        if children is not None:
            if isinstance(children, Child):
                children = [children]
            self.addChildren(children)

    def addChildren(
            self, children: Dict[str, Dict[str, Union[Symbol, List[Symbol]]]]
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
        return {name: self.addChild(name, arguments) for name, arguments in children.items()}

    def addChild(self, name: str, arguments: Dict[str, List[Symbol]]) -> Dict[str, List[Symbol]]:
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
        Get children functions.
        
        __Recursion Base__
            return all children: names [None]

        :param self: parent to retrieve child(s) from
        :param names: name(s) of child(s) to retrieve from parent.
            Defaults to all children.
        """

        self: Function

        def get(name: str) -> Function:
            """Base method for :meth:`~Function.Model.getChildren`"""
            child = self.getModel().getFunctions(names=name)
            return child

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getChildrenNames()
        }
        return recursiveMethod(**kwargs)

    def getChildrenNames(self) -> List[str]:
        """
        Get names of children functions.
        
        __Recursion Base__
            retrieve name of single parent: functions [Function]
            retrieve names of all parents: functions [None]

        :param self: parent to retrieve child name(s) from
        """
        self: Function
        model = self.getModel()

        free_symbol_names = self.getFreeSymbols(generations=0, substitute_dependents=False, return_type=str)
        dependent_children_names = list(self.instance_arguments.keys())
        instance_argument_function_names = []
        for instance_argument in self.instance_arguments.values():
            try:
                function_names = list(map(str, instance_argument["functions"]))
                independent_function_names = [
                    function_name
                    for function_name in function_names
                    if isinstance(model.getFunctions(names=function_name), Independent)
                ]
                instance_argument_function_names.extend(independent_function_names)
            except KeyError:
                pass

        children_names = dependent_children_names + instance_argument_function_names
        if isinstance(self, Independent):
            self: Function
            model_function_names = model.getFunctionNames()
            model_children_names = [
                name
                for name in free_symbol_names
                if name in model_function_names
            ]
            children_names.extend(model_children_names)

        children_names = unique(children_names)
        return unique(children_names)

    def getInstanceArgumentSpecies(self, name: str) -> List[str]:
        """
        Get all possible species of instance arguments.

        :param self: parent to retrieve instance-argument species from
        :param name: name of child to retrieve instance-argument species for
        """
        return self.instance_arguments[name].keys()

    def getInstanceArguments(self, name: str, specie: str = None) -> Union[Symbol, List[Symbol]]:
        """
        Get instance arguments of given species for function.
        
        __Recursion Base__
            return all instance arguments of given specie: specie [None] and names [None]

        :param self: parent to retrieve instance arguments from
        :param specie: name of instance-argument species to retrieve from parent, acts as an optional filter
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

        def get(name: str) -> Derivative:
            """Base method for :meth:`~Function.Model.getParents`"""
            parents = self.getModel().getFunctions(names=name)
            return parents

        kwargs = {
            "base_method": get,
            "args": names,
            "valid_input_types": str,
            "output_type": list,
            "default_args": self.getParentNames()
        }
        return recursiveMethod(**kwargs)

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
            if name in function_object.getChildrenNames()
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
        self.setModel(model)

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
        :param kwargs: additional arguments to pass into
            :meth:`~Function.Piecewise.getExpression` or into :meth:`~Function.NonPiecewise.getExpression`
        """
        free_symbols = list(self.getExpression(**kwargs).free_symbols)
        if species is None:
            free_symbol_names = list(map(str, free_symbols))
        else:
            unfiltered_names = list(map(str, free_symbols))

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
            return list(map(Symbol, free_symbol_names))
        else:
            raise ValueError("return_type must be Symbol or str")

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
            self, variable: Union[Symbol, str], time_evolution_type: str = "Temporal", initial_condition: float = 0
    ) -> None:
        """
         Constructor for :class:`~Function.Derivative`.
         
        :param variable: variable that derivative is a derivative of
        :param time_evolution_type: time-evolution type of derivative (e.g. "Temporal", "Equilibrium", "Constant")
        :param initial_condition: initial value of associated variable
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
        :param variable: variable to associate with derivative
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
    Stores properties for a function that requires input from another function.
    
    :ivar general_arguments: dictionary of arguments to be substituted from another function.
        Key is specie of arguments.
        Value is list of symbols for arguments of specie.
    """

    def __init__(self, arguments: Union[Symbol, List[Symbol]]) -> None:
        """
        Constructor for :class:`~Function.Dependent`.
        
        :param arguments: general arguments to store in function
        """
        self.general_arguments = {}
        self.setGeneralArguments(arguments)

    def setGeneralArguments(
            self,
            arguments: Union[Union[Symbol, List[Symbol]], Dict[str, Union[Symbol, List[Symbol]]]],
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
            if isinstance(arguments, Symbol):
                self.general_arguments[specie] = [arguments]
            elif isinstance(arguments, list):
                for argument in arguments:
                    if not isinstance(argument, Symbol):
                        raise TypeError("argument must be sympy.Symbol")
                self.general_arguments[specie] = arguments
            elif arguments is None:
                self.general_arguments[specie] = []
            else:
                raise RecursiveTypeError(arguments, Symbol)
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
            self: Function
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
        Get general arguments of function.
        
        __Recursion Base__
            return arguments of single species: species [str] and nested [False]
        
        :param self: :class:`~Function.Dependent` to retrieve arguments from
        :param species: specie(s) of arguments to retrieive, acts as an optional filter
        :param nested: set True to include arguments from children (implicit).
            Set False to only include arguments from self (explicit).
        
        :returns: dictionary of arguments if :paramref:`~Function.Dependent.getGeneralArguments.species` is list.
            Key is specie of argument.
            Value is symbols for argument of specie.
            Symbols for argument
            if :paramref:`~Function.Dependent.getGeneralArguments.species` is str.
        """
        if nested:
            general_arguments = self.getGeneralArguments(species=species, nested=False)
            self: Function
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
            instance_function_symbols = parent.getInstanceArguments(specie=specie, name=self_name)
            instance_names = list(map(str, instance_function_symbols))
            instance_functions = []
            for instance_name in instance_names:
                sibling = parent.getChildren(names=instance_name)
                if isinstance(sibling, Dependent):
                    instance_functions.append(sibling.getInstanceArgumentForm(parent))
                elif isinstance(sibling, Function):
                    instance_functions.append(sibling.getExpression(generations="all"))
                else:
                    raise TypeError(f"sibling for {self_name:s} must be Function")
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
        expression = self.getExpression(generations="all")
        self: Dependent
        species = self.getGeneralSpecies()
        for specie in species:
            substitutions = self.getInstanceArgumentSubstitutions(parent, specie)
            expression = expression.subs(substitutions)
        return expression


class Independent:
    """
    Stores properties for a function that does not require input from another function.
    """

    def __init__(self) -> None:
        """
        Constructor for :class:`~Function.Independent`.
        """
        pass


class Piecewise:
    """
    Stores info pertaining to piecewise function.
    
    :ivar pieces: function symbols constituting piecewise (pieces)
    :ivar conditions: conditions under which to use each piece
    """

    def __init__(self, pieces: List[Function], conditions: List[str]) -> None:
        """
        Constructor for :class:`~Function.Piecewise`.
        
        :param pieces: symbols for function pieces
        :param conditions: conditions corresponding to function pieces
        """
        if len(pieces) != len(conditions):
            raise ValueError("each function must have exactly one corresponding condition")
        self.pieces = pieces
        self.conditions = list(map(eval, conditions))  # eval() to convert str to bool

    def getConditions(self) -> List[bool]:
        """
        Get conditions, to determine which function piece to use.
        
        :param self: :class:`~Function.Piecewise` to retrieve conditions from
        """
        return self.conditions

    def getPieces(
            self, return_type: Type[Union[str, Symbol, Function]]
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
            piece_names = list(map(str, pieces))
            return piece_names
        elif return_type == Symbol:
            return pieces
        elif return_type == Function:
            piece_names = self.getPieces(return_type=str)
            self: Function
            functions = self.getModel().getFunctions(names=piece_names)
            return functions
        else:
            raise ValueError("invalid return type")

    def getPieceCount(self) -> int:
        """
        Get number of pieces constituting piecewise function.
        
        :param self: :class:`~Function.Piecewise` to retrieve pieces from
        """
        return len(self.pieces)

    def getExpression(self, generations: Union[int, str] = 0, **kwargs) -> spPiecewise:
        """
        Get symbolic piecewise expression.

        :param self: :class:`~Function.Piecewise` to retrieve expression for
        :param generations: see :paramref:`~Function.Function.getExpression.generations`
        :param kwargs: additional arguments to substitute into :meth:`~Function.Function.getExpression`
        """
        if generations == "all":
            functions: List[Function] = self.getPieces(return_type=Function)
            getExpression = partial(Function.getExpression, generations="all", **kwargs)
            pieces = list(map(getExpression, functions))
        elif generations >= 1:
            functions: List[Function] = self.getPieces(return_type=Function)
            getExpression = partial(Function.getExpression, generations=generations - 1, **kwargs)
            pieces = list(map(getExpression, functions))
        elif generations == 0:
            pieces = self.getPieces(return_type=Symbol)
        else:
            raise ValueError("generations must be 'all' or some integer greater than or equal to 0")
        conditions = self.getConditions()
        exprconds = [(pieces[i], conditions[i]) for i in range(self.getPieceCount())]
        return spPiecewise(*exprconds)


class NonPiecewise:
    """
    Stores info pertaining to nonpiecewise (standard) function.
    
    :ivar expression: symbolic expression for function
    """

    def __init__(self, expression: Expr) -> None:
        """
        Constructor for :class:`~Function.NonPiecewise`.

        :param expression: symbolic expression for function
        """
        self.expression = expression

    def getExpression(
            self, parent: Function = None, substitute_dependents: bool = None, generations: Union[int, str] = 0
    ) -> Expr:
        """
        Get symbol expression for function.

        :param self: :class:`~Function.Function` to retrieve expression for
        :param parent: function to retrieve input arguments from.
            Only called if self is Dependent function.
        :param substitute_dependents: set True to substitute all dependents into expression.
            Set False to substitute in accordance with :paramref:`~Function.Function.getExpression.generations`.
            Defaults to False if :paramref:`~Function.Function.getExpression.generations`==0.
            Defaults to True otherwise.
        :param generations: number of generations for children to substitute into expression.
            Set to 0 to retrieve original expression without substitutions.
            Set to positive integer n to substitute n generations of children.
            Set to "all" to substitute all generations of children.
        """
        if isinstance(self, Dependent) and isinstance(parent, Independent):
            parent: Function
            return self.getInstanceArgumentForm(parent)

        if substitute_dependents is None:
            if generations == "all":
                substitute_dependents = True
            elif generations >= 1:
                substitute_dependents = True
            elif generations == 0:
                substitute_dependents = False

        expression = self.expression
        self: Function
        if generations == 0:
            has_model = isinstance(self.getModel(), Model)
            if substitute_dependents:
                if has_model:
                    pass
                else:
                    warnings.warn(f"cannot substitute dependents without associated model for {self.getName():s}")
                    return expression
            else:
                return expression

        for child in self.getChildren():
            child_symbol = child.getName(return_type=Symbol)
            if isinstance(child, Dependent):
                child: Function
                if substitute_dependents:
                    child_expression = child.getExpression(parent=self)
                else:
                    child_expression = child_symbol
            elif isinstance(child, Independent):
                child: Function
                if generations == "all":
                    child_expression = child.getExpression(
                        substitute_dependents=substitute_dependents,
                        generations=generations
                    )
                elif generations >= 1:
                    child_expression = child.getExpression(
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
        expression: Union[Expr, List[Function]],
        inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
        **kwargs
) -> Function:
    """
    Get Function object with desired properties/inheritance.
    
    :ivar name: name of function
    :ivar function: symbolic expression for function
        if :class:`~Function.NonPiecewise`.
        Symbols for pieces
        if :class:`~Function.Piecewise`.
    :ivar inheritance: classes for Function object to inherit
    :ivar kwargs: arguments to pass into inheritance classes.
        Key is string name of class.
        Value is dictionary of arguments/parameters to pass into class.
    """

    class FunctionCore(Function, *inheritance):
        """
        Stores info pertinent to generate/simulate function.
        """

        # noinspection PyShadowingNames
        def __init__(
                self,
                name: str,
                expression: Union[Expr, List[Function]],
                inheritance: Tuple[Type[Union[Derivative, Dependent, Independent, Piecewise, NonPiecewise]]] = (),
                **kwargs
        ) -> None:
            """
            Constructor for :class:`~Function.FunctionMaster.FunctionCore`.
            
            :param name: name of function
            :param expression: symbol expression/pieces for function
            :param inheritance: classes for Function object to inherit
            :param kwargs: arguments to pass into inheritance classes.
                Key is string name of class.
                Value is dictionary of arguments/parameters to pass into class.
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
                Piecewise.__init__(self, expression, **kwargs["Piecewise"])
            elif NonPiecewise in inheritance:
                NonPiecewise.__init__(self, expression)
            else:
                raise ValueError("Function must inherit either Piecewise of NonPiecewise")

            Function.__init__(self, name, **kwargs)

    return FunctionCore(name, expression, inheritance, **kwargs)


def getFunctionInfo(info: dict, model: Model = None) -> dict:
    """
    Get ormatted dictionary of info to generate Function object.

    :param info: 2/3-level dictionary of info directly from file.
        First key is name of function.
        Second key is name of property for function.
        Value is string or 1-level dictionary, which indicates property value.
    :param model: model to associated Function object with
    """

    # noinspection PyShadowingNames
    def getVariables(info: dict) -> List[Symbol]:
        """
        Get symbolic variables for function.

        :param info: info for function
        """
        return var(info["variables"])

    # noinspection PyShadowingNames
    def getParameters(info: dict) -> List[Symbol]:
        """
        Get symbolic parameters for function.

        :param info: info for function
        """
        return var(info["parameters"])

    # noinspection PyShadowingNames
    def getProperties(info: dict) -> List[str]:
        """
        Get properties to give function (e.g. piecewise, dependent)

        :param info: info for function
        """
        return info["properties"]

    # noinspection PyShadowingNames
    def getArguments(info: dict) -> Dict[str, List[Symbol]]:
        """
        Get symbolic general arguments for function.

        :param info: info for function
        """
        arguments = info["arguments"]
        return {key: var(arguments[key]) for key in arguments.keys()}

    # noinspection PyShadowingNames
    def getVariable(info: dict) -> Symbol:
        """
        Get associated variable for derivative.

        :param info: info for function
        """
        return Symbol(info["variable"])

    # noinspection PyShadowingNames
    def getChildren(info: dict) -> dict:
        """
        Get info to connect function with child.
        
        :param info: info for function
        :returns: 2-level dictionary of instance arguments for function into child.
            First key is name of child function.
            Second key is specie of instance arguments.
            Value is symbols for instance arguments.
        """
        children_info = info["children"]
        children_names = list(children_info.keys())
        var(children_names)

        children_dict = {}
        for child_name in children_names:
            if (child_info := children_info[child_name]) is not None:
                children_dict[child_name] = {
                    argument_type: var(child_info[argument_type])
                    for argument_type in child_info.keys()
                }
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
        kwargs["children"] = getChildren(info)

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
    """
    Generate all functions from file.
    
    :param filename: name of file to read functions from
    :param kwargs: additional arguments to pass into :meth:`~Function.generateFunction`
    :returns: Generated functions
    """
    file = yaml.load(open(filename), Loader=yaml.Loader)
    filestem = Path(filename).stem
    return [generateFunction(name, file[name], filestem=filestem, **kwargs) for name in file.keys()]


def generateParametersFromFile(filename: str, **kwargs) -> List[Parameter]:
    """
    Generate all parameter from file.

    :param filename: name of file to read parameters from
    :param kwargs: additional arguments to pass into :meth:`~Function.generateParameter`
    :returns: Generated parameters
    """
    file = yaml.load(open(filename), Loader=yaml.Loader)
    filestem = Path(filename).stem
    return [generateParameter(name, file[name], filestem=filestem, **kwargs) for name in file.keys()]


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
        info: Dict[str, Union[str, dict]],
        model: Model = None,
        filestem: str = None
) -> Function:
    """
    Generate Function object.

    :param name: name of function to generate
    :param info: dictionary of info needed to generate function
    :param model: :class:`~Function.Model` to add function into
    :param filestem: stem of filepath where function was loaded form, optional
    :returns: Generated function object
    """
    kwargs = getFunctionInfo(info, model=model)
    kwargs["filestem"] = filestem

    info_keys = info.keys()
    if "form" in info_keys:
        expression = eval(info["form"])
    elif "pieces" in info_keys:
        expression = list(map(Symbol, info["pieces"]))
    else:
        raise ValueError("info from functions_yml file must contain either form or pieces")

    inheritance = getInheritance(kwargs["properties"])
    return FunctionMaster(name, expression, inheritance=tuple(inheritance), **kwargs)


def generateParameter(name: str, info: Dict[str, Union[float, str]], **kwargs) -> Parameter:
    """
    Generate Parameter object.

    :param name: name of function to generate
    :param info: dictionary of info needed to generate function
    :param kwargs: additional arguments to pass into :class:`~Function.Parameter`
    :returns: Generated function object
    """
    value = float(info["value"])
    unit = info["unit"]
    quantity = value * units(unit)
    return Parameter(name, quantity, **kwargs)


def createModel(function_ymls: Union[str, List[str]], parameter_ymls: Union[str, List[str]]) -> Model:
    """
    Create model from YML files.

    :param function_ymls: name(s) of YML file(s) containing info for function
    :param parameter_ymls: name(s) of YML file(s) containing info about parameter values/units
    """
    model = Model()
    model.loadParametersFromFile(parameter_ymls)
    model.loadFunctionsFromFiles(function_ymls)
    return model


def readParameters(filepaths: Union[str, List[str]]) -> Dict[str, Parameter]:
    """
    Read file containing information about parameters

    :param filepaths: name(s) of file(s) containing information
    :returns: Dictionary of parameter quantities.
        Key is name of parameter.
        Value is Quantity containg value and unit.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    parameters = {}
    for filepath in filepaths:
        parameters = yaml.load(open(filepath, 'r'), Loader=yaml.Loader)
        filestem = Path(filepath).stem
        for name, info in parameters.items():
            parameters[name] = generateParameter(name, info, filestem=filestem)
    return parameters


def readFunctions(filepath: Union[str, List[str]]) -> Dict[str, Function]:
    """
    Read file containing information about parameters

    :param filepath: name(s) of file(s) containing information
    :returns: Dictionary of parameter quantities.
        Key is name of parameter.
        Value is Quantity containg value and unit.
    """

    if isinstance(filepath, str):
        filepath = [filepath]

    functions = {}
    for filepath in filepath:
        function_info = yaml.load(open(filepath, 'r'), Loader=yaml.Loader)
        filestem = Path(filepath).stem
        for name, info in function_info.items():
            functions[name] = generateFunction(name, info, filestem=filestem)
    return functions
