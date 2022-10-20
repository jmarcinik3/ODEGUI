from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.optimize import OptimizeResult, minimize


def costFunctionRefactor(
    parameters: ndarray,
    input_vector: ndarray,
    output_vector: ndarray,
    fitting_function: Callable[[ndarray, ndarray], ndarray],
    cost_function: Callable[[ndarray, ndarray], float]
) -> float:
    """
    Get cost of fitting parameters to input/output vectors.

    :param parameters: parameter to input into fitting function
    :param input_vector: vector of values to input into fitting function
    :param out_vector: vector of values to fit function outputs to
    :param fitting_function: function form to fit to input/output vectors.
        Takes as input two ndarrays.
        First input is input vector.
        Second input is parameter vector.
    :param cost_function: function to calculate cost between fitting function (using given parameter values) with input/output vectors
    """
    output_fit = fitting_function(input_vector, parameters)
    fit_cost = cost_function(output_vector, output_fit)
    return fit_cost


class MonteCarloMinimization:
    """
    :ivar function_calls: number of calls to fitting function during minimization
    :ivar points_calls: number of points input to fitting function during minimization
    :ivar iterations: number of iterations performed during minimization
    """

    def __init__(
        self,
        fitting_function: Callable[[ndarray, ndarray], ndarray],
        initial_guess: ndarray,
        input_vector: ndarray,
        output_vector: ndarray,
        cost_function: Callable[[ndarray, ndarray], float],
        point_distribution: ndarray = None,
        sample_sizes: Union[int, ndarray] = 3,
        tolerance: float = 1e-6,
        minimization_method: str = "SLSQP",
        jacobian_method: str = "2-point",
        bounds: List[Tuple[float, float]] = None,
        options: Dict[str, Any] = None,
        post_iteration_function: Callable[[], None] = None
    ) -> None:
        """
        :param fitting_function: function to fit input/output vectors to.
            Takes as input two ndarrays.
                First array is vector of input values (chosen from input_vector).
                Second array is vector of parameter values.
        :param initial_guess: vector of initial guess for parameter values
        :param input_vector: vector of input values (e.g. independent variable).
            Must have same shape as output vector, possibly excluding final dimension (indexed at -1).
        :param output_vector: vector of output values (e.g. dependent variable).
            Must have same shape as input vector, possibly excluding final dimension (indexed at -1).
        :param cost_function: function to calculate cost of each parameter fit.
            Goal is to find parameter values that minimize this function.
            Takes as input two ndarrays.
                First array is vector of output values (chosen from output_vector).
                Second array is simulated output vector.
        :param point_distribution: Weights for each point in given input/output vector.
            Used as probability distribution when choosing points to fit.
            Must have same shape as input and output vectors, possibly excluding ending dimension (indexed at -1).
            Defaults to a uniform distribution.
        :param minimization_iterations: number of Monte Carlo iterations to perform.
            Each of these iterations performs a full minimization on a random subset of input/output points.
            Successive iterations perform on larger subset sizes, logarithmically, until the global input/output space is minimized.
        :param tolerance: see scipy.optimize.minimize.tol
        :param sample_sizes: see scipy.optimize.minimize.method if float.
            Collection of custom sample sizes if ndarray.
        :param jacobian_method: see scipy.optimize.minimize.jac
        :param bounds: see scipy.optimize.bounds
        :param options: see scipy.optimize.options
        :param post_iteration_function: functiont to execute after completing one Monte-Carlo iteration
        """
        assert isinstance(input_vector, ndarray)
        input_vector_shape = input_vector.shape
        input_vector_flat_size = np.prod(input_vector_shape[:-1])
        input_vector_flat = input_vector.reshape((input_vector_flat_size, -1))
        self.input_vector_flat = input_vector_flat

        assert isinstance(output_vector, ndarray)
        input_vector_dimension = input_vector.ndim
        output_vector_dimension = output_vector.ndim

        output_vector_shape = output_vector.shape
        if output_vector_dimension == input_vector_dimension:
            assert input_vector_shape[:-1] == output_vector_shape[:-1]
            output_vector_flat_size = np.prod(output_vector_shape[:-1])
            output_vector_flat = output_vector.reshape((input_vector_flat_size, -1))
        elif output_vector_dimension == input_vector_dimension - 1:
            assert input_vector_shape[:-1] == output_vector_shape
            output_vector_flat_size = output_vector.size
            output_vector_flat = output_vector.flatten()
        else:
            raise ValueError("output vector must same or less-one dimension with input vector")

        assert input_vector_flat_size == output_vector_flat_size
        self.vector_flat_size = input_vector_flat_size
        self.output_vector_flat = output_vector_flat

        assert isinstance(initial_guess, ndarray)
        self.initial_guess = initial_guess

        assert isinstance(fitting_function, Callable)
        self.fitting_function = fitting_function
        assert isinstance(cost_function, Callable)
        self.cost_function = cost_function

        if point_distribution is None:
            self.probability_distribution = partial(
                np.random.choice,
                input_vector_flat_size,
                replace=False
            )
        else:
            assert isinstance(point_distribution, ndarray)
            point_distribution_shape = point_distribution.shape
            assert point_distribution_shape == output_vector_shape
            point_probabilities = point_distribution / np.sum(point_distribution)
            point_probabilities_flat = point_probabilities.flatten()

            self.probability_distribution = partial(
                np.random.choice,
                input_vector_flat_size,
                p=point_probabilities_flat,
                replace=False
            )

        if isinstance(sample_sizes, int):
            if sample_sizes == 1:
                sample_sizes = np.array([input_vector_flat_size])
            elif sample_sizes >= 2:
                sample_sizes = np.logspace(
                    0,
                    np.log10(input_vector_flat_size),
                    sample_sizes,
                    endpoint=True,
                    dtype=np.int32
                )
        else:
            assert isinstance(sample_sizes, ndarray)
            for sample_size in sample_sizes:
                assert 1 <= sample_size <= input_vector_flat_size
        self.sample_sizes = sample_sizes

        assert isinstance(tolerance, (float, int))
        self.tolerance = tolerance

        assert isinstance(minimization_method, str)
        self.minimization_method = minimization_method

        assert isinstance(jacobian_method, str)
        self.jacobian_method = jacobian_method

        if bounds is None:
            bounds = ()
        else:
            assert isinstance(bounds, list)
            for bound in bounds:
                assert isinstance(bound, tuple)
                assert len(bound) == 2
                lower_bound, upper_bound = bound
                assert isinstance(lower_bound, float) or lower_bound is None
                assert isinstance(upper_bound, float) or upper_bound is None
        self.bounds = bounds

        if options is None:
            options = {}
        else:
            assert isinstance(options, dict)
            for key, value in options.items():
                assert isinstance(key, str)
        self.options = options

        assert isinstance(post_iteration_function, Callable) or post_iteration_function is None
        self.post_iteration_function = post_iteration_function

        self.function_calls = 0
        self.point_calls = 0
        self.iterations = 0

    def getVectorFlatSize(self) -> int:
        """
        Get size of flattened input points.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve size from
        """
        return self.vector_flat_size

    def getSampleSizes(self) -> ndarray:
        """
        Get sample sizes for Monte Carlo iterations.
        Sizes are increased logarithmically, from 1 to the size of the flattened vector of points.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve sizes from
        """
        return self.sample_sizes

    def getProbabilityDistribution(self) -> Callable[[float], ndarray]:
        """
        Get probability distribution indicating weights for points in random subset.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve distribution from
        """
        return self.probability_distribution

    def getRandomSamples(self, sample_size: int) -> ndarray:
        """
        Get random subset for vector of points, without replacement, sampled from the probability distribution.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve subset from
        :param sample_size: number of points to include in random subset
        """
        probability_distribution = self.getProbabilityDistribution()
        sample_indicies = probability_distribution(size=sample_size).T
        return sample_indicies

    def getInputVector(self) -> ndarray:
        """
        Get vector, with dimension (number of points, number of dimensions per point), of input values.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve vector from
        """
        return self.input_vector_flat

    def getOutputVector(self) -> ndarray:
        """
        Get vector, with dimension (number of points, number of dimensions per point), of output values.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve vector from
        """
        return self.output_vector_flat

    def getFittingFunction(self) -> Callable[[ndarray, ndarray], ndarray]:
        """
        Get function to fit input and output vectors to.
        Takes as input: data input vector and parameter vector.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve function from
        """
        return self.fitting_function

    def getCostFunction(self) -> Callable[[ndarray, ndarray], float]:
        """
        Get cost function to minimize during parameter fit.
        Takes as input: data output vector and simulated output vector.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve function from
        """
        cost_function = self.cost_function

        def costFunction(data: ndarray, data_compare: ndarray) -> float:
            """
            Get cost between data and comparison.

            :param data: vector of points to compare to comparison data (e.g. fit data)
            :param data_compare: vector of points to compare to data (e.g. simulated data)
            """
            is_not_nan = np.logical_and(
                np.logical_not(np.isnan(data)),
                np.logical_not(np.isnan(data_compare))
            )
            data_filtered = data[is_not_nan]
            data_compare_filtered = data_compare[is_not_nan]

            cost = cost_function(data_filtered, data_compare_filtered)
            return cost

        return costFunction

    def getInitialParametersGuess(self) -> ndarray:
        """
        Get initial guess for vector of parameter values.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve guess from
        """
        return self.initial_guess

    def getTolerance(self) -> ndarray:
        """
        Get tolerance for minimization.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve tolerance from
        """
        return self.tolerance

    def getMinimizationMethod(self) -> str:
        """
        Get name of method to minimize cost function.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve method from
        """
        return self.minimization_method

    def getJacobianMethod(self) -> str:
        """
        Get name of method to estimate jacobian.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve method from
        """
        return self.jacobian_method

    def getBounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds to use during minimization.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve bounds from
        """
        return self.bounds

    def getOptions(self) -> Dict[str, Any]:
        """
        Get options to use during minimization.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve bounds from
        """
        return self.options

    def executePostIterationFunction(self) -> None:
        """
        Execute function after completing one Monte-Carlo iteration.

        :param self: :class:`~ParameterFit.MonteCarloMinimization` to retrieve function from
        """
        post_iteration_function = self.post_iteration_function
        post_iteration_function()

    def runMonteCarloIteration(
        self,
        sample_size: int,
        parameters_guess: ndarray = None
    ) -> OptimizeResult:
        """
        Run minimization fit on random subset of points.

        :param sample_size: size of random subset to minimize
        :param parameters_guess: vector of parameter values for initial guess to minimization
        """
        if parameters_guess is None:
            parameters_guess = self.getInitialParametersGuess()

        sample_indicies = self.getRandomSamples(sample_size)

        input_vector_flat = self.getInputVector()
        input_samples = input_vector_flat[sample_indicies]

        output_vector_flat = self.getOutputVector()
        output_samples = output_vector_flat[sample_indicies]

        fitting_function = self.getFittingFunction()
        cost_function = self.getCostFunction()
        minimization_cost_function = partial(
            costFunctionRefactor,
            input_vector=input_samples,
            output_vector=output_samples,
            fitting_function=fitting_function,
            cost_function=cost_function
        )

        tolerance = self.getTolerance()
        minimization_method = self.getMinimizationMethod()
        jacobian_method = self.getJacobianMethod()

        bounds = self.getBounds()
        options = self.getOptions()
        minimize_output = minimize(
            minimization_cost_function,
            parameters_guess,
            bounds=bounds,
            method=minimization_method,
            tol=tolerance,
            jac=jacobian_method,
            options=options
        )

        is_success = minimize_output.success
        if not is_success:
            print('optimization metadata:', minimize_output, sep='\n')
            raise ValueError("minimization failed")

        self.executePostIterationFunction()
        return minimize_output

    def runMonteCarloIterations(self) -> ndarray:
        """
        Run Monte Carlo iterations, up to and including fitting whole set of points.
        """
        self.function_calls = 0
        self.point_calls = 0
        self.iterations = 0

        sample_sizes = self.getSampleSizes()
        initial_sample_size = sample_sizes[0]

        minimize_output = self.runMonteCarloIteration(sample_size=initial_sample_size)
        parameters_fit = minimize_output.x

        if sample_sizes.size >= 2:
            for sample_size in sample_sizes[1:]:
                minimize_output = self.runMonteCarloIteration(
                    sample_size=sample_size,
                    parameters_guess=parameters_fit
                )
                parameters_fit = minimize_output.x

                function_calls = minimize_output.nfev
                point_calls = function_calls * sample_size
                iterations = minimize_output.nit

                self.function_calls += function_calls
                self.point_calls += point_calls
                self.iterations += iterations

        return parameters_fit
