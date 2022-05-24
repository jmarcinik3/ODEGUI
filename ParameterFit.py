from functools import partial
from typing import Callable

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from Transforms.CustomMath import sumSquaresError


def costFunctionRefactor(
    parameters,
    input_vector,
    output_vector,
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
        cost_function: Callable[[ndarray], float],
        point_distribution: ndarray = None,
        minimize_iterations: int = 3,
        tolerance: float = 1e-6,
        minimization_method: str = "SLSQP",
        jacobian_method: str = "2-point"
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
            Takes as input one ndarray of parameter values.
        :param point_distribution: Weights for each point in given input/output vector.
            Used as probability distribution when choosing points to fit.
            Must have same shape as input and output vectors, possibly excluding ending dimension (indexed at -1).
            Defaults to a uniform distribution.
        :param minimization_iterations: number of Monte Carlo iterations to perform.
            Each of these iterations performs a full minimization on a random subset of input/output points.
            Successive iterations perform on larger subset sizes, logarithmically, until the global input/output space is minimized.
        :param tolerance: see scipy.optimize.minimize.tol
        :param minimization_method: see scipy.optimize.minimize.method
        :param jacobian_method: see scipy.optimize.minimize.jac
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

        assert isinstance(minimize_iterations, int)
        self.minimize_iterations = minimize_iterations

        assert isinstance(tolerance, (float, int))
        self.tolerance = tolerance

        assert isinstance(minimization_method, str)
        self.minimization_method = minimization_method

        assert isinstance(jacobian_method, str)
        self.jacobian_method = jacobian_method

        self.function_calls = 0
        self.point_calls = 0
        self.iterations = 0

    def getVectorFlatSize(self) -> int:
        """
        Get size of flattened input points.
        """
        return self.vector_flat_size

    def getMinimizationIterationCount(self) -> int:
        """
        Get number of Monte Carlo iterations to perform.
        """
        return self.minimize_iterations

    def getSampleSizes(self) -> ndarray:
        """
        Get sample sizes for Monte Carlo iterations.
        Sizes are increased logarithmically, from 1 to the size of the flattened vector of points.
        """
        minimize_iteration_count = self.getMinimizationIterationCount()
        vector_flat_size = self.getVectorFlatSize()

        if minimize_iteration_count == 1:
            sample_sizes = np.array([vector_flat_size])
        elif minimize_iteration_count >= 2:
            sample_sizes = np.logspace(
                0,
                np.log10(vector_flat_size),
                minimize_iteration_count,
                endpoint=True,
                dtype=np.int32
            )

        return sample_sizes

    def getProbabilityDistribution(self) -> Callable[[float], ndarray]:
        """
        Get probability distribution indicating weights for points in random subset.
        """
        return self.probability_distribution

    def getRandomSamples(self, sample_size: int) -> ndarray:
        """
        Get random subset for vector of points, without replacement, sampled from the probability distribution.

        :param sample_size: number of points to include in random subset
        """
        probability_distribution = self.getProbabilityDistribution()
        sample_indicies = probability_distribution(size=sample_size).T
        return sample_indicies

    def getInputVector(self) -> ndarray:
        """
        Get vector , with dimension (number of points, number of dimensions per point), of input values.
        """
        return self.input_vector_flat

    def getOutputVector(self) -> ndarray:
        """
        Get vector , with dimension (number of points, number of dimensions per point), of output values.
        """
        return self.output_vector_flat

    def getFittingFunction(self) -> Callable[[ndarray, ndarray], ndarray]:
        """
        Get function to fit input and output vectors to.
        """
        return self.fitting_function

    def getCostFunction(self) -> Callable[[ndarray], float]:
        """
        Get cost function to minimize during parameter fit.
        """
        return self.cost_function

    def getInitialParametersGuess(self) -> ndarray:
        """
        Get initial guess for vector of parameter values.
        """
        return self.initial_guess

    def getTolerance(self) -> ndarray:
        """
        Get tolerance for minimization.
        """
        return self.tolerance

    def getMinimizationMethod(self) -> str:
        """
        Get name of method to minimize cost function.
        """
        return self.minimization_method

    def getJacobianMethod(self) -> str:
        """
        Get name of method to estimate jacobian.
        """
        return self.jacobian_method

    def runMonteCarloIteration(
        self,
        sample_size: int,
        parameters_guess: ndarray = None
    ) -> ndarray:
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

        minimize_output = minimize(
            minimization_cost_function,
            parameters_guess,
            method=minimization_method,
            tol=tolerance,
            jac=jacobian_method
        )
        parameters_fit = minimize_output.x

        function_calls = minimize_output.nfev
        point_calls = function_calls * sample_size
        iterations = minimize_output.nit

        self.function_calls += function_calls
        self.point_calls += point_calls
        self.iterations += iterations

        return parameters_fit

    def runMonteCarloIterations(self) -> ndarray:
        """
        Run Monte Carlo iterations, up to and including fitting whole set of points.
        """
        self.function_calls = 0
        self.point_calls = 0
        self.iterations = 0

        sample_sizes = self.getSampleSizes()
        initial_sample_size = sample_sizes[0]

        parameters_fit = self.runMonteCarloIteration(sample_size=initial_sample_size)

        if sample_sizes.size >= 2:
            for sample_size in sample_sizes[1:]:
                parameters_fit = self.runMonteCarloIteration(
                    sample_size=sample_size,
                    parameters_guess=parameters_fit
                )

        return parameters_fit


if __name__ == "__main__":
    x_sizes = [127, 20]
    p = np.random.randint(-4, 4, 7)

    xs = [
        np.linspace(-1, 1, x_sizes[0]),
        np.linspace(-1, 1, x_sizes[1]),
        # np.linspace(-5, -1, x_sizes[2])
    ]
    x_vec_meshgrids = np.meshgrid(*reversed(xs))
    x_vector = np.moveaxis(
        np.array(x_vec_meshgrids),
        0,
        -1
    )

    def fittingFunction(x, p):
        poly = 0
        for i, pi in enumerate(p):
            poly += pi * x ** i
        partial_sums = np.sum(poly, axis=-1)
        return partial_sums

    proportional_deviation = 0.05
    y_vector = fittingFunction(x_vector, p)
    y_vector *= np.random.normal(1, proportional_deviation, y_vector.shape)

    y_gradient = np.array(np.gradient(y_vector, *xs))
    y_divergence = np.sum(y_gradient, axis=0)
    y_probability = np.nan_to_num(y_divergence, nan=0) ** (-2)

    tolerance = 1e-4
    minimization_method = "SLSQP"
    jacobian_method = "2-point"

    pcallss = []
    iterss = []
    fcallss = []
    ns = np.arange(50)

    for n in ns:
        p0 = 20 * np.random.random(p.size) - 10

        monte_carlo_minimization = MonteCarloMinimization(
            fitting_function=fittingFunction,
            cost_function=sumSquaresError,
            initial_guess=p0,
            input_vector=x_vector,
            output_vector=y_vector,
            point_distribution=y_probability,
            tolerance=tolerance,
            minimize_iterations=3
        )
        parameters_fit = np.round(
            monte_carlo_minimization.runMonteCarloIterations(),
            decimals=3
        )

        fcalls = monte_carlo_minimization.function_calls
        pcalls = monte_carlo_minimization.point_calls
        iters = monte_carlo_minimization.iterations
        fcalls_per_iter = fcalls / iters
        pcalls_per_fcall = pcalls / fcalls

        pcallss.append(pcalls)
        fcallss.append(fcalls)
        iterss.append(iters)

    pcallss = np.array(pcallss)
    iterss = np.array(iterss)
    fcallss = np.array(fcallss)
    fcall_per_iter = fcallss / iterss

    print(
        f"{np.mean(pcallss):.0f} +/- {np.std(pcallss):.0f} point calls",
        f" and {np.mean(fcallss):.0f} +/- {np.std(fcallss):.0f} function calls\n",
        f"{np.mean(fcall_per_iter):.2f} +/- {np.std(fcall_per_iter):.2f} function calls per iteration",
        f" ({np.mean(iterss):.1f} +/- {np.std(iterss):.1f} iterations)\n",
        #f"in {s2 - s1:.3f}s",
        sep=''
    )
    print(parameters_fit)
