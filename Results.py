import itertools
from os import remove
from os.path import basename, dirname, join
from typing import Dict, Iterable, List, Tuple, Union
from zipfile import ZipFile

import dill
import numpy as np
import yaml
from numpy import ndarray
from scipy import fft, signal
from scipy.stats import stats
from sympy import Expr
from sympy import Symbol
from sympy.utilities.lambdify import lambdify

from Function import Model
from macros import commonElement, recursiveMethod


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

    def __init__(self, model: Model, free_parameter_values: Dict[str, ndarray], results: dict = None):
        """
        Constructor for :class:`~Results.Results`

        :param model: :class:`~Function.Model` to calculate results from
        :param free_parameter_values: dictionary of values for free parameters.
            Key is name of free parameter.
            Value is possible values for free parameter.
        """
        self.results = {} if results is None else results
        self.model = model
        self.general_function_expressions = {}
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

    def getFreeParameterValues(
            self, names: Union[str, Iterable[str]] = None, output_type: type = list
    ) -> Union[ndarray, Dict[str, ndarray]]:
        """
        Get values for a free parameter.
        
        :param self: :class:`~Results.Results` to retreive value from
        :param names: name(s) of parameter to retreive values for
        :param output_type: iterable to output as
            if :paramref:`~Results.Results.getFreeParameterValues.names` is iterable
        """

        def get(name: str) -> ndarray:
            """Base method for :meth:`~Results.Results.getFreeParameterValues`"""
            return self.free_parameter_values[name]

        kwargs = {
            "args": names,
            "base_method": get,
            "valid_input_types": str,
            "output_type": output_type,
            "default_args": self.getFreeParameterNames()
        }
        return recursiveMethod(**kwargs)

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
        return free_parameter_substitutions

    def getParameterSubstitutions(
            self,
            index: Union[tuple, Tuple[int, ...]] = None,
            name: str = None,
            include_nonfree: bool = True,
            include_free: bool = False
    ) -> Dict[Symbol, float]:
        """
        Get substitutions from parameter symbol to parameter value.

        :param self: :class:`~Results.Results` to retrieve values from
        :param index: index of free parameters to retrieve free-parameter values from
        :param name: name of function to retrieve parameter names from.
            Returns substitutions for all parameters in model if None.
            Returns substitutions only for parameters in function if str.
        :param include_free: set True to include substitutions for free parameters.
            Set False to exclude them.
            :paramref:`~Results.Results.getParameterSubstitutions.index` must be given if set to True.
        :param include_nonfree: set True to include substitutions for non-free parameters.
            Set False to exclude them.
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

        substitutions = {}
        if include_free:
            substitutions.update(self.getFreeParameterSubstitutions(index))
        if include_nonfree:
            nonfree_substitutions = model.getParameterSubstitutions(
                parameter_names,
                skip_parameters=self.getFreeParameterNames()
            )
            substitutions.update(nonfree_substitutions)

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

        parameter_substitutions = self.getParameterSubstitutions(index=index)
        simplified_expression = general_expression.subs(parameter_substitutions)
        return simplified_expression

    def getGeneralFunctionExpression(self, name: str) -> Expr:
        """
        Get expression for function, with values for non-free parameter substituted in.

        :param self: :class:`~Results.Results` to retrieve expression from
        :param name: name of function to retrieve expression for
        """
        try:
            expression = self.general_function_expressions[name]
        except KeyError:
            expression = self.getModel().getFunctions(names=name).getExpression(generations=0)
            parameter_substitutions = self.getParameterSubstitutions(include_free=False)
            expression = expression.subs(parameter_substitutions)
            self.general_function_expressions[name] = expression

        return expression

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
        expression_lambda = lambdify([[Symbol('t'), *variables]], expression_sub, modules=["numpy", "scipy"])
        variable_names = list(map(str, variables))
        temporal_results = self.getResultsOverTime(index, quantity_names=variable_names)
        times = self.getResultsOverTime(index, quantity_names='t')
        times = times.reshape((1, times.size))
        arguments = np.append(times, temporal_results, axis=0)
        results = expression_lambda(arguments)
        return np.array(results)

    def getFunctionResults(self, index: Union[tuple, Tuple[int]], name: str) -> ndarray:
        """
        Get results from simulation for function.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of parameter value for free parameter
        :param name: name of function to retrieve results ofZ
        """
        expression = self.getGeneralFunctionExpression(name)
        parameter_substitutions = self.getParameterSubstitutions(index, include_nonfree=False, include_free=True)
        expression = expression.subs(parameter_substitutions)

        free_symbols = expression.free_symbols
        free_symbol_names = list(map(str, free_symbols))
        expression_lambda = lambdify([free_symbols], expression, modules=["numpy", "scipy"])
        substitutions_results = self.getResultsOverTime(index=index, quantity_names=free_symbol_names)
        updated_results = expression_lambda(substitutions_results)
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
        results = np.repeat(initial_condition, self.getResultsOverTime(index, 't').size)
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
        results = self.getResultsOverTime(index=index, quantity_names=name, **kwargs)
        times = self.getResultsOverTime(index=index, quantity_names='t', **kwargs)

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
            frequencies = time_to_frequency(extrema_times[0:-1], extrema_times[1:])
        elif "max" in calculation_method and "fourier" in calculation_method:
            harmonic_count = int(calculation_method.split('_')[-1])
            time_count = times.size
            time_resolution = (times[-1] - times[0]) / (time_count - 1)
            fourier_results, frequencies = abs(fft.rfft(results)), fft.rfftfreq(time_count, time_resolution)

            maxima_indicies = signal.find_peaks(fourier_results)[0]
            if maxima_indicies.size >= 1:
                harmonic_frequencies = frequencies[maxima_indicies]
                harmonic_frequencies = np.insert(harmonic_frequencies, 0, 0)
                n_harmonic_frequencies = harmonic_frequencies[:harmonic_count]
                frequencies = n_harmonic_frequencies[1:] - n_harmonic_frequencies[0:-1]
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

    def getStandardDeviation(self, index: Union[tuple, Tuple[int]], name: str, **kwargs) -> float:
        """
        Get RMS of deviation for results.

        :param self: :class:`~Results.Results` to retrieve results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param name: name of quantity to retrieve mean for
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        """
        results = self.getResultsOverTime(index=index, quantity_names=name, **kwargs)
        return np.std(results)

    def getHolderMean(self, index: Union[tuple, Tuple[int]], name: str, order: int = 1, **kwargs) -> float:
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
        results = self.getResultsOverTime(index=index, quantity_names=name, **kwargs)

        if order == 1:
            mean = np.mean(results)
        elif order == 2:
            mean = np.sqrt(np.mean(results ** 2))
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

    def getFourierTransform(self, index: Union[tuple, Tuple[int, ...]], quantity_name: str) -> ndarray:
        """
        Get Fourier transform of results.
        
        :param self: :class:`~Results.Results` to retreive results from
        :param index: index of results.
            This is a tuple of indicies.
            The index of the tuple corresponds to the parameter in free parameters.
            The index within the tuple corresponds to the value of the parameter in its set of possible values.
        :param quantity_name: name of quantity to retreive Fourier transform of
        """
        original_results = self.getResultsOverTime(index, quantity_names=quantity_name)
        # noinspection PyPep8Naming
        N = original_results.size

        if quantity_name == 't':
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
            quantity_names: Union[str, List[str]] = None,
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
        :param quantity_names: name(s) of variable/function(s) to retrieve results for
        :param transform_name: transform to perform on results.
            "Fourier" - Fourier transform.
        :param condensor_name: analysis to perform on results to reduce into one (or few) floats.
            "Frequency" - calculate frequency of oscillation
        """
        results = self.results[index]
        if isinstance(quantity_names, str):
            if condensor_name != "None":
                kwargs = {
                    "index": index,
                    "name": quantity_names,
                    "transform_name": transform_name
                }
                if condensor_name == "Frequency":
                    return self.getOscillationFrequency(**kwargs, **condensor_kwargs)
                elif condensor_name == "Mean":
                    return self.getHolderMean(**kwargs, **condensor_kwargs)
                elif condensor_name == "Standard Deviation":
                    return self.getStandardDeviation(**kwargs, **condensor_kwargs)
                else:
                    raise ValueError(f"invalid condensor name ({condensor_name:s})")

            if transform_name != "None":
                if transform_name == "Fourier":
                    return self.getFourierTransform(index=index, quantity_name=quantity_names)
                else:
                    raise ValueError(f"invalid transform name ({transform_name:s})")

            try:
                return results[quantity_names]
            except KeyError:
                pass

            model = self.getModel()

            if quantity_names in model.getVariables(return_type=str):
                time_evolution_type = model.getDerivativesFromVariableNames(names=quantity_names).getTimeEvolutionType()
                results_handles = {
                    "Equilibrium": self.getEquilibriumVariableResults,
                    "Constant": self.getConstantVariableResults,
                    "Function": self.getFunctionResults
                }
                # noinspection PyArgumentList
                updated_results = results_handles[time_evolution_type](index, quantity_names)
            elif quantity_names in model.getFunctionNames():
                updated_results = self.getFunctionResults(index, quantity_names)
            else:
                raise ValueError("quantity_names input must correspond to either variable or function when str")

            # noinspection PyUnboundLocalVariable
            self.setResults(index, updated_results, quantity_names)
            return updated_results
        elif isinstance(quantity_names, list):
            kwargs = {
                "index": index,
                "transform_name": transform_name
            }
            new_results = np.array([self.getResultsOverTime(quantity_names=name, **kwargs) for name in quantity_names])
            return new_results
        else:
            raise TypeError("names input must be str or list")

    def getResultsOverTimePerParameter(
            self,
            index: Union[tuple, Tuple[int]],
            parameter_names: str,
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
        :param parameter_names: name(s) of free parameter(s) to retrieve quantity results over.
        :param quantity_names: name(s) of quantity(s) to average results over
        :param kwargs: additional arguments to pass into :meth:`~Results.Results.getResultsOverTime`
        :returns: tuple of results.
            First index gives array of parameter base values.
                nth index of this array gives values for nth parameter in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.parameter_names`.
            Second index gives matrix quantity results.
                nth index of this matrix gives values for nth quantity in
                :paramref:`~Results.Results.getResultsOverTimePerParameter.quantity_names`.
        """
        if isinstance(quantity_names, str):
            quantity_names = [quantity_names]
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        per_parameter_locations = np.array(list(map(self.getFreeParameterIndex, parameter_names)))
        per_parameter_base_values = tuple(list(self.getFreeParameterValues(names=parameter_names)))
        per_parameter_stepcounts = tuple(map(len, per_parameter_base_values))
        per_parameter_partial_indicies = list(itertools.product(*map(range, per_parameter_stepcounts)))

        default_index = np.array(index)
        times = self.getResultsOverTime(index=index, quantity_names='t', **kwargs)
        if isinstance(times, (float, int)):
            single_result_size = per_parameter_stepcounts
        elif isinstance(times, ndarray):
            single_result_size = (*per_parameter_stepcounts, *times.shape)

        results = np.zeros((len(quantity_names), *single_result_size))
        for quantity_location, quantity_name in enumerate(quantity_names):
            single_results = np.zeros(single_result_size)
            for partial_index in per_parameter_partial_indicies:
                new_index = default_index
                new_index[per_parameter_locations] = partial_index

                single_result = self.getResultsOverTime(index=tuple(new_index), quantity_names=quantity_name, **kwargs)
                single_results[partial_index] = single_result
            results[quantity_location] = single_results
        return per_parameter_base_values, results

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

    def saveToFile(self, filepath: str) -> None:
        """
        Save results object (self) into *.zip file.

        :param self: :class:`~Results.Results` to save into file
        :param filepath: path of file to save object into
        """
        model = self.getModel()

        free_parameter_info = {}
        for name, values in self.getFreeParameterValues(output_type=dict).items():
            unit = str(model.getParameters(names=name).getQuantity().to_base_units().units)
            free_parameter_info[name] = {
                "values": list(map(str, values)),
                "unit": unit
            }

        path_directory = dirname(filepath)

        function_file = model.saveFunctionsToFile(join(path_directory, "Function.yml"))
        parameter_file = model.saveParametersToFile(join(path_directory, "Parameter.yml"))
        time_evolution_type_file = model.saveParametersToFile(join(path_directory, "TimeEvolutionType.yml"))

        free_parameter_file = open(join(path_directory, "FreeParameter.yml"), 'w')
        yaml.dump(free_parameter_info, free_parameter_file, default_flow_style=None)
        free_parameter_file.close()

        results_file = open(join(path_directory, "Results.pkl"), 'wb')
        dill.dump(self.results, results_file)
        results_file.close()

        files = [function_file, parameter_file, free_parameter_file, results_file, time_evolution_type_file]

        with ZipFile(filepath, 'w') as zipfile:
            for file in files:
                filepath = file.name
                filename = basename(filepath)
                zipfile.write(filepath, filename)
        zipfile.close()

        for file in files:
            remove(file.name)
