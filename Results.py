import pickle
from typing import Dict, List, Tuple, Union

import dill
import numpy as np
from numpy import ndarray
from scipy import fft, signal
from scipy.stats import stats
from sympy import Expr
from sympy import Symbol
from sympy.utilities.lambdify import lambdify

from Function import Model
from macros import commonElement


class Results:
    """
    This class stores results from an ODE simulation.
    Minimally, results for temporal variables are required to be set.
    Other results may be calculated and saved from :class:`~Function.Model` as needed.
    
    :ivar results: 2D dictionary of results.
        First key is indicies of free parameter for current data to present.
        Second key is name of quantity to retrieve results of.
        Value is array of quantitiy values over time.
    :ivar model: :class:`~Function.Model` to calculated results from
    :ivar free_parameter_values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
    :ivar general_equilibrium_expressions: dictionary of symbolic equilibrium expressions.
        Key is name of variable.
        Value is symbolic expression.
        This attribute is so that equilibria only need to be calculated once.
        They are reused after their initial calculation.
    """

    def __init__(self, model: Model, free_parameter_values: Dict[str, ndarray]):
        """
        Constructor for :class:`~Results.Results`

        :param model: :class:`~Function.Model` to calculate results from
        :param free_parameter_values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
        """
        self.results = {}
        self.model = model
        self.free_parameter_values = free_parameter_values
        self.general_equilibrium_expressions = {}

    def getModel(self) -> Model:
        """
        Get associated :class:`~Function.Model`.
        
        :param self: :class:`~Results.Results` to retrieve associated :class:`~Function.Model` from
        """
        return self.model

    def getFreeParameterIndex(self, name: str):
        """
        Get index of free parameter within collection of free-parameter names.
        
        :param self: :class:`~Results.Results` to retreive free-parameter names from
        :param name: name of free parameter to retreive index of
        """
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_index = free_parameter_names.index(name)
        return free_parameter_index

    def getFreeParameterNames(self) -> List[str]:
        """
        Get names of free parameters.

        :param self: :class:`~Results.Results` to retrieve free-parameter names from
        """
        return list(self.free_parameter_values.keys())

    def getFreeParameterValues(self, names: Union[str, List[str]] = None) -> Union[ndarray, Dict[str, ndarray]]:
        """
        Get values for a free parameter.
        
        :param self: :class:`~Results.Results` to retreive value from
        :param names: name(s) of parameter to retreive values for
        """
        if isinstance(names, str):
            free_parameter_values = self.getFreeParameterValues()
            return free_parameter_values[names]
        elif isinstance(names, list):
            return {name: self.getFreeParameterValues(names=name) for name in names}
        elif names is None:
            return self.free_parameter_values
        else:
            raise TypeError("names must be str or list")

    def getFreeParameterSubstitutions(self, index: Union[tuple, Tuple[int, ...]]) -> Dict[Symbol, float]:
        """
        Get substitutions for free parameters at index.

        :param self: :class:`~Results.Results` to retrieve substitutions from
        :param index: index of free parameters to retrieve free-parameter values from
        """
        free_parameter_names = self.getFreeParameterNames()
        free_parameter_substitutions = {}
        for parameter_location, free_parameter_name in enumerate(free_parameter_names):
            parameter_index = index[parameter_location]
            parameter_values = self.getFreeParameterValues(names=free_parameter_name)
            parameter_value = parameter_values[parameter_index]
            free_parameter_substitutions[Symbol(free_parameter_name)] = parameter_value
        print(free_parameter_substitutions)
        return free_parameter_substitutions

    def getParameterSubstitutions(self, index: Union[tuple, Tuple[int, ...]], name: str = None) -> Dict[Symbol, float]:
        """
        Get substutitions from parameter symbol to parameter value.

        :param self: :class:`~Results.Results` to retrieve values from
        :param index: index of free parameters to retrieve free-parameter values from
        :param name: name of function to retrieve parameter names from.
            Returns substitutions for all parameters in model if None.
            Returns substitutions only for parameters in function if str.
        """
        model = self.getModel()

        if name is None:
            parameter_names = None
        else:
            function = model.getFunctions(names=name)
            kwargs = {
                "species": "Parameter",
                "generations": "all",
                "return_type": str
            }
            parameter_names = function.getFreeSymbols(**kwargs)

        substitutions = model.getParameterSubstitutions(parameter_names, skip_parameters=self.getFreeParameterNames())
        substitutions.update(self.getFreeParameterSubstitutions(index))
        return substitutions

    def setEquilibriumExpressions(self, equilibrium_expressions: Dict[Symbol, Expr] = None) -> None:
        """
        Set symbolic expressions for equilibrium variables.
        These expressions are simplified, except variables and free parameters are kept symbolic.
        Equilibria may be set manually or calculated automatically from the stored :class:`~Function.Model`.
        
        :param self: :class:`~Results.Results` to set equilibria for
        :param equilibrium_expressions: dictionary of equilibria for variables.
            Key is symbolic variable.
            Value is symbolic equilibrium expression of variable.
        """
        if equilibrium_expressions is None:
            solutions = self.getModel().getEquilibriumSolutions(skip_parameters=self.getFreeParameterNames())
            solutions = {variable: solution for variable, solution in solutions.items()}
            self.general_equilibrium_expressions = solutions
        else:
            self.general_equilibrium_expressions = equilibrium_expressions

    def getEquilibriumExpression(self, index: Union[tuple, Tuple[int, ...]], name: Union[Symbol, str]) -> Expr:
        """
        Get equilibrium expression for a variable.
        
        :param self: :class:`~Results.Results` to retrieve equilibrium from
        :param index: index of free parameters to retrieve free-parameter values from
        :param name: name of variable to retrieve equilibrium for
        """
        if len(self.general_equilibrium_expressions.keys()) == 0:
            self.setEquilibriumExpressions()

        if isinstance(name, Symbol):
            general_expression = self.general_equilibrium_expressions[name]
        elif isinstance(name, str):
            general_expression = self.general_equilibrium_expressions[Symbol(name)]
        else:
            raise TypeError("name must be sp.Symbol or str")

        parameter_substitutions = self.getParameterSubstitutions(index=index, name=name)
        simplified_expression = general_expression.subs(parameter_substitutions)
        return simplified_expression

    def resetResults(self) -> None:
        """
        Reset results to store a new set of them.

        :param self: :class:`~Results.Results` to reset results for
        """
        self.results = {}

    def getSubstitutedResults(self, index: Union[tuple, Tuple[int]], expression: Expr, name: str = None) -> ndarray:
        """
        Get results from simulation for function, after substituting results from variables.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of free parameters to retrieve substitutive results at
        :param expression: function to substitute results into
        :param name: name of function where expression was derived from
        """
        expression_sub = expression.subs(self.getParameterSubstitutions(index=index, name=name))

        variables = self.getModel().getVariables(time_evolution_types="Temporal")
        function_lambda = lambdify((Symbol('t'), tuple(variables)), expression_sub, "numpy")
        variable_names = [str(variable) for variable in variables]
        temporal_results = self.getResultsOverTime(index, names=variable_names)
        times = self.getResultsOverTime(index, names='t')
        results = [function_lambda(times[i], temporal_results[i]) for i in range(len(times))]
        return np.array(results)

    def getFunctionResults(self, index: Union[tuple, Tuple[int]], name: str) -> ndarray:
        """
        Get results from simulation for function.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of parameter value for free parameter
        :param name: name of function to retrieve results ofZ
        """
        model = self.getModel()
        function = model.getFunctions(names=name)
        expression = function.getExpression(generations="all")

        substitutions = {}
        function_variables = set(function.getFreeSymbols(species="Variable", generations="all"))
        equilibrium_variables = set(model.getVariables(time_evolution_types="Equilibrium"))
        if commonElement(function_variables, equilibrium_variables):
            substitutions.update(model.getEquilibriumSolutions())

        constant_variables = set(model.getVariables(time_evolution_types="Constant"))
        if commonElement(function_variables, constant_variables):
            substitutions.update(model.getConstantSubstitutions())

        derivative_function_variables = set(model.getVariables(time_evolution_types="Function"))
        if commonElement(function_variables, derivative_function_variables):
            substitutions.update(model.getFunctionSubstitutions())

        expression = expression.subs(substitutions)
        updated_results = self.getSubstitutedResults(index, expression)
        return updated_results

    def getEquilibriumVariableResults(self, index: Union[tuple, Tuple[int]], name: str) -> ndarray:
        """
        Get results from simulation for variable in equilibrium.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of parameter value for free parameter
        :param name: name of variable to retrieve results of
        """
        results = self.getSubstitutedResults(index, self.getEquilibriumExpression(index, name))
        return np.array(results)

    def getConstantVariableResults(self, index: Union[tuple, Tuple[int]], name: str) -> ndarray:
        """
        Get results from simulation for constant variable.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of parameter value for free parameter
        :param name: name of variable to retrieve results of
        """
        initial_condition = self.getModel().getDerivativesFromVariableNames(names=name).getInitialCondition()
        results = np.repeat(initial_condition, len(self.getResultsOverTime(index, 't')))
        return results

    def getOscillationFrequency(
            self,
            index: Union[tuple, Tuple[int]],
            name: str,
            calculation_method: str = "autocorrelation",
            condensing_method: str = "average",
            **kwargs
    ) -> float:
        """
        Get oscillation frequency for quantity.

        :param self: :class:`~Results.Results` to retrieve quantity results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param name: name of quantity to retrieve frequency for
        :param calculation_method: method used to calculate frequencies.
            "maxima_separation" uses peak-to-peak separation of waveform.
            "minima_separation" uses trough-to-trough separation of waveform.
            "extrema_separation" uses peaks and troughs separation of waveform.
            "maxima_fourier_[n]" uses peak-to-peak separation of Fourier transform (i.e. separation of harmonics).
            This method uses separations for the first n maxima.
            "autocorrelation" uses autocorrelation of waveform.
            This method uses the argument of the first local maximum, excluding zero.
        :param condensing_method: method used to "average" frequencies.
            "average" uses arithmetic mean of frequencies.
            "maximum" uses maximum of frequencies.
            "minimum" uses minium of frequencies.
            "initial" uses first frequency in frequencies.
            "final" uses last frequency in frequencies.
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        """
        calculation_method = calculation_method.lower()
        condensing_method = condensing_method.lower()
        results = self.getResultsOverTime(index=index, names=name, **kwargs)
        times = self.getResultsOverTime(index=index, names='t', **kwargs)

        if "separation" in calculation_method:
            if "max" in calculation_method or "min" in calculation_method:
                time_to_frequency = lambda initial_time, final_time: 1 / (final_time - initial_time)
            elif "extrema" in calculation_method:
                time_to_frequency = lambda initial_time, final_time: 0.5 / (final_time - initial_time)
            else:
                raise ValueError("separation method must include maxima, minima, xor extrema")

            extrema_indicies = np.array([], dtype=np.int32)
            if "max" in calculation_method or "extrema" in calculation_method:
                maxima_indicies = signal.find_peaks(results)[0]
                extrema_indicies = np.append(extrema_indicies, maxima_indicies)
            if "min" in calculation_method or "extrema" in calculation_method:
                minima_indicies = signal.find_peaks(-results)[0]
                extrema_indicies = np.append(extrema_indicies, minima_indicies)
            extrema_times = times[extrema_indicies]
            frequencies = np.array(
                [time_to_frequency(extrema_times[i], extrema_times[i + 1]) for i in range(len(extrema_times) - 1)]
            )
        elif "max" in calculation_method and "fourier" in calculation_method:
            harmonic_count = int(calculation_method.split('_')[-1])
            print('1', harmonic_count)
            time_count = len(times)
            time_resolution = (times[-1] - times[0]) / (time_count - 1)
            fourier_results, frequencies = abs(fft.rfft(results)), fft.rfftfreq(time_count, time_resolution)

            maxima_indicies = signal.find_peaks(fourier_results)[0]
            if len(maxima_indicies) >= 1:
                harmonic_frequencies = frequencies[maxima_indicies]
                harmonic_frequencies = np.insert(harmonic_frequencies, 0, 0)
                n_harmonic_frequencies = harmonic_frequencies[:harmonic_count]
                frequencies = np.array(
                    [n_harmonic_frequencies[i + 1] - n_harmonic_frequencies[i] for i in
                        range(len(n_harmonic_frequencies) - 1)]
                )
            else:
                frequencies = np.array([])
        elif "autocorrelation" in calculation_method:
            results_count = results.size
            correlation = signal.correlate(results, results, mode="same")[results_count // 2:]
            lags = signal.correlation_lags(results_count, results_count, mode="same")[results_count // 2:]

            argrelmax_correlation = signal.argrelmax(correlation)[0]
            argrelmax_count = argrelmax_correlation.size

            if argrelmax_count >= 1:
                lag = lags[argrelmax_correlation][0]
                delta_time = times[1] - times[0]
                frequencies = np.array([1 / (lag * delta_time)])
            else:
                frequencies = np.array([0])
        else:
            raise ValueError("invalid calculation method")

        frequency_count = frequencies.size
        if frequency_count >= 1:
            if condensing_method == "average":
                frequency = np.mean(frequencies)
            elif condensing_method == "maximum":
                frequency = np.amax(frequencies)
            elif condensing_method == "minimum":
                frequency = np.amin(frequencies)
            elif condensing_method == "initial":
                frequency = frequencies[0]
            elif condensing_method == "final":
                frequency = frequencies[-1]
            else:
                raise ValueError("invalid condensing method")
        else:
            frequency = 0
        return frequency

    def getHolderMean(
            self, index: Union[tuple, Tuple[int]], name: str, order: int = 1, **kwargs
    ) -> float:
        """
        Get Holder mean for results.
        
        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param name: name of quantity to retrieve mean for
        :param order: order of Holder mean
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        """
        results = self.getResultsOverTime(index=index, names=name, **kwargs)

        if order == 1:
            mean = np.mean(results)
        elif order == 2:
            mean = np.sqrt(np.mean(results) ** 2)
        elif order == 0:
            mean = stats.gmean(results)
        # elif order == -1: mean = stats.hmean(results)
        elif np.isinf(order) and np.sign(order) == 1:
            mean = np.amax(results)
        elif np.isinf(order) and np.sign(order) == -1:
            mean = np.amin(results)
        else:
            mean = np.mean(results ** order) ** (1 / order)

        return mean

    def getFourierTransform(self, index: Union[tuple, Tuple[int, ...]], name: str):
        """
        Get Fourier transform of results.
        
        :param self: :class:`~Results.Results` to retreive results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param name: name of quantity to retreive Fourier transform of
        """
        original_results = self.getResultsOverTime(index, names=name)
        # noinspection PyPep8Naming
        N = len(original_results)

        if name == 't':
            initial_time, final_time = original_results[0], original_results[-1]
            time_resolution = (final_time - initial_time) / (N - 1)
            fourier_results = fft.rfftfreq(N, time_resolution)
        else:
            fourier_results = fft.rfft(original_results)
            fourier_results = abs(fourier_results)
        return fourier_results

    def getResultsOverTime(
            self,
            index: Union[tuple, Tuple[int, ...]],
            names: Union[str, List[str]] = None,
            transform_name: str = "None",
            condensor_name: str = "None",
            **condensor_kwargs
    ) -> Union[float, ndarray]:
        """
        Get results for variable or function over time.
        Results are evaluated from simulation.
        Optionally perform a transformation on the quantity.
        Optionally condense variable values into one value.
        
        __Recursion Base__
            return results for single variable: names [str]

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param names: name(s) of variable/function(s) to retrieve results for
        :param transform_name: transform to perform on results.
            "Fourier" - Fourier transform.
        :param condensor_name: analysis to perform on results to reduce into one (or few) floats.
            "Frequency" - calculate frequency of oscillation
        """
        results = self.results[index]
        if isinstance(names, str):
            if condensor_name != "None":
                kwargs = {
                    "index": index,
                    "name": names,
                    "transform_name": transform_name
                }
                if condensor_name == "Frequency":
                    return self.getOscillationFrequency(**kwargs, **condensor_kwargs)
                elif condensor_name == "Mean":
                    return self.getHolderMean(**kwargs, **condensor_kwargs)
                else:
                    raise ValueError("invalid condensor name")

            if transform_name != "None":
                if transform_name == "Fourier":
                    return self.getFourierTransform(index=index, name=names)
                else:
                    raise ValueError("invalid transform name")

            try:
                return results[names]
            except KeyError:
                pass

            model = self.getModel()

            if names in model.getVariables(return_type=str):
                time_evolution_type = model.getDerivativesFromVariableNames(names=names).getTimeEvolutionType()
                results_handles = {
                    "Equilibrium": self.getEquilibriumVariableResults,
                    "Constant": self.getConstantVariableResults,
                    "Function": self.getFunctionResults
                }
                # noinspection PyArgumentList
                updated_results = results_handles[time_evolution_type](index, names)
            elif names in model.getFunctionNames():
                updated_results = self.getFunctionResults(index, names)
            else:
                ValueError("names input must correspond to either variable or function when str ")

            # noinspection PyUnboundLocalVariable
            self.setResults(index, updated_results, names)
            return updated_results
        elif isinstance(names, list):
            kwargs = {
                "index": index,
                "transform_name": transform_name
            }
            new_results = np.array([self.getResultsOverTime(names=name, **kwargs) for name in names])
            transpose = new_results.T
            return transpose
        elif names is None:
            return self.getResultsOverTime(index, names=list(results.keys()))
        else:
            raise TypeError("names input must be str or list")

    def getResultsOverTimePerParameter(
            self,
            index: Union[tuple, Tuple[int]],
            parameter_name: str,
            quantity_names: Union[str, List[str]],
            **kwargs
    ) -> Tuple[ndarray, ...]:
        """
        Get free-parameter values and "averaged" quantity values.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param parameter_name: name of free parameter to average quantity results over.
        :param quantity_names: name of quantity to average results over
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        :returns: tuple of results.
            First index gives parameter values.
            Second-last index gives quantity results; one set of quantity results per index.
        """
        if isinstance(quantity_names, str):
            quantity_names = [quantity_names]

        parameter_index = self.getFreeParameterIndex(parameter_name)
        parameter_values = self.getFreeParameterValues(names=parameter_name)
        parameter_stepcount = len(parameter_values)
        list_index = list(index)
        new_index = lambda i: tuple(list_index[:parameter_index] + [i] + list_index[parameter_index + 1:])

        results = []
        for quantity_name in quantity_names:
            new_results = np.array(
                [
                    self.getResultsOverTime(index=new_index(i), names=quantity_name, **kwargs)
                    for i in range(parameter_stepcount)
                ]
            )
            results.append(new_results)
        return parameter_values, *tuple(results)

    def setResults(
            self, index: Union[tuple, Tuple[int]], results: Union[ndarray, Dict[str, ndarray]], name: str = None
    ) -> None:
        """
        Save results from simulation.

        :param self: :class:`~Results.Results` to save results in
        :param index: index of parameter value for free parameter
        :param results: results to save at :paramref:`~Results.Results.setResults.index`.
            This must be a list of floats if name is specified.
            This must be a dictionary if name is not specified.
            Key is name of variable.
            Value is list of floats for variable over time.
        :param name: name of quantity to set results for
        """
        if isinstance(name, str):
            self.results[index][name] = results
        elif name is None:
            self.results[index] = results

    def saveToFile(self, filename: str) -> None:
        """
        Save results object (self) into file.

        :param self: :class:`~Results.Results` to save into file
        :param filename: name of file to save object into
        """
        file = open(filename, 'wb')
        model = self.getModel()
        save_info = {
            "results": self.results,
            "free_parameter_values": self.getFreeParameterValues(),
            "model_parameters": {
                parameter.getName(): parameter.getQuantity()
                for parameter in model.getParameters()
            },
            "model_functions": {
                function_object.getName():
                    (
                        function_object.getExpression(),
                        function_object.getFreeSymbols(),
                        function_object.instance_arguments
                    )
                for function_object in model.getFunctions()
            }
        }
        dill.dump(save_info, file)
