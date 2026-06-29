"""
Predictive models and fitting these models
"""

import numpy as np  # for the math
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import optimize
from scipy.linalg import sqrtm

from .kernel_empirical_risk import KernelEmpiricalRisk
from .losses import maker
from ..conformal_prediction.utils import compute_operator_norm


def build_root_output_covariance_estimator_grid(inputs, output_covariance_estimator):
    sample_size = inputs.shape[0]
    return [
        sqrtm(output_covariance_estimator.predict(inputs[i]))
        for i in range(sample_size)
    ]


def build_gram_matrix(inputs, scalar_gram_matrix, root_output_covariance_estimator_grid):
    sample_size = inputs.shape[0]
    gram_matrix = np.concatenate(
        tuple(
            [
                np.concatenate(
                    tuple(
                        [
                            scalar_gram_matrix[i, j]
                            * root_output_covariance_estimator_grid[i]
                            @ root_output_covariance_estimator_grid[j]
                            for j in range(sample_size)
                        ]
                    ),
                    axis=1,
                )
                for i in range(sample_size)
            ]
        ),
        axis=0,
    )
    return gram_matrix


def solve_kernel_regression(
    gram_matrix,
    outputs,
    lam=0.5,
    solver="Newton-CG",
    max_iter=200,
    tol=1e-6,
    loss_name="log_cosh",
    loss_params={"alpha": 1.0},
):
    if type(gram_matrix) is tuple:
        scalar_gram_matrix, output_covariance = gram_matrix
        initial_model_weights = scalar_gram_matrix @ (
            np.random.normal(0, 1, outputs.shape) @ output_covariance
        )
        initial_model_weights = initial_model_weights.T.ravel(order="F")
    else:
        initial_model_weights = gram_matrix @ np.random.normal(0, 1, outputs.size)

    empirical_risk = KernelEmpiricalRisk(loss_name, loss_params)

    if solver not in [
        "L-BFGS-B",
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    ]:
        raise ValueError("Only can handle this for now, sorry")

    elif solver == "L-BFGS-B":
        func = empirical_risk.empirical_risk_gradient
        optimization_result = optimize.minimize(
            func,
            initial_model_weights,
            method=solver,
            jac=True,
            args=(gram_matrix, outputs, lam),
            options={
                "maxiter": max_iter,
                "maxls": 50,  # default is 20
                "gtol": tol,
                "ftol": 64 * np.finfo(float).eps,
            },
        )
        final_model_weights = optimization_result.x
        print(optimization_result.success)
        print(optimization_result.message)

    elif solver in [
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    ]:
        fun = empirical_risk.empirical_risk
        grad = empirical_risk.gradient
        hess = empirical_risk.hessian
        # hessp = empirical_risk.hessian_product

        optimization_result = optimize.minimize(
            fun=fun,
            x0=initial_model_weights,
            method=solver,
            jac=grad,
            hess=hess,
            # hessp=hessp,
            args=(gram_matrix, outputs, lam),
            tol=tol,
        )
        final_model_weights = optimization_result.x
        print(optimization_result.success)
        print(optimization_result.message)

    return final_model_weights


class KernelRegression:
    """
    Performs kernel regression
    """

    def __init__(
        self,
        lam=0.5,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        kernel_max=1.0,
        output_covariance=None,
        loss_name="log_cosh",
        loss_params={"alpha": 1.0},
        solver="Newton-CG",
        max_iter=200,
        tol=1e-6,
    ):
        self.name = "KernelRegression"

        self.lam = lam  # only one is supported
        self.kernel = kernel
        self.kernel_max = kernel_max
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.output_covariance = output_covariance

        self.loss_name = loss_name
        self.loss_params = loss_params

        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

        self.inputs = None
        self.outputs = None
        self.output_dimension = None
        self.scalar_gram_matrix = None
        self.root_output_covariance_estimator_grid = None
        self.model_weights = None

    # I took this function from sklearn's KernelRidge
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0,
            }
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def fit(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.output_dimension = self.outputs.shape[1]

        self.scalar_gram_matrix = self._get_kernel(inputs)
        if self.output_covariance is None:
            self.output_covariance = np.eye(self.output_dimension)

        gram_matrix = (self.scalar_gram_matrix, self.output_covariance)
        if hasattr(self.output_covariance, "predict"):
            self.root_output_covariance_estimator_grid = build_root_output_covariance_estimator_grid(
                self.inputs, self.output_covariance
            )
            gram_matrix = build_gram_matrix(
                self.inputs, self.scalar_gram_matrix, self.root_output_covariance_estimator_grid
            )

        self.model_weights = solve_kernel_regression(
            gram_matrix,
            self.outputs,
            self.lam,
            self.solver,
            self.max_iter,
            self.tol,
            self.loss_name,
            self.loss_params,
        )
        return

    def predict(self, new_inputs):
        if self.model_weights is None:
            raise ValueError("Please fit the model first.")

        if self.output_covariance is None:
            self.output_covariance = np.eye(self.output_dimension)

        prediction_scalar_feature_matrix = self._get_kernel(new_inputs, self.inputs)
        sample_size = prediction_scalar_feature_matrix.shape[1]
        new_sample_size = new_inputs.shape[0]

        if hasattr(self.output_covariance, "predict"):
            new_root_output_covariance_estimator_grid = build_root_output_covariance_estimator_grid(
                new_inputs, self.output_covariance
            )
            prediction_feature_matrix = np.concatenate(
                tuple(
                    [
                        np.concatenate(
                            tuple(
                                [
                                    prediction_scalar_feature_matrix[i, j]
                                    * new_root_output_covariance_estimator_grid[i]
                                    @ self.root_output_covariance_estimator_grid[j]
                                    for j in range(sample_size)
                                ]
                            ),
                            axis=1,
                        )
                        for i in range(new_sample_size)
                    ]
                ),
                axis=0,
            )
            if self.model_weights.ndim != 1:
                self.model_weights = self.model_weights.T.ravel(order="F")
            predictions = prediction_feature_matrix @ self.model_weights
            predictions = predictions.reshape(
                (self.output_dimension, new_sample_size), order="F"
            ).T
        else:
            model_weights = self.model_weights.reshape(
                (self.output_dimension, sample_size), order="F"
            ).T
            predictions = prediction_scalar_feature_matrix @ (
                model_weights @ self.output_covariance
            )

        return predictions

    def compute_predictor_stability_bound(self, new_input):
        if self.scalar_gram_matrix is None:
            raise ValueError("Please fit the model first.")
        sample_size = self.scalar_gram_matrix.shape[0]

        if self.output_covariance is None:
            self.output_covariance = np.eye(self.output_dimension)

        if hasattr(self.output_covariance, "predict"):
            output_covariance_operator_norm = compute_operator_norm(
                self.output_covariance.predict(new_input)
            )
        else:
            output_covariance_operator_norm = compute_operator_norm(
                self.output_covariance
            )
        loss_ = maker(self.loss_name)(**self.loss_params)

        loss_Lipschitz_constant = loss_["lams"]["rho"]

        new_scalar_gram_value = self._get_kernel(new_input).item()

        predictor_stability_bounds = (
            loss_Lipschitz_constant
            * np.sqrt(output_covariance_operator_norm)
            * np.sqrt(new_scalar_gram_value)
            / (2 * self.lam * (sample_size + 1))
        )

        return predictor_stability_bounds

    def compute_loss_stability_bound(self):
        if self.scalar_gram_matrix is None:
            raise ValueError("Please fit the model first.")
        sample_size = self.scalar_gram_matrix.shape[0]

        if self.output_covariance is None:
            self.output_covariance = np.eye(self.output_dimension)

        if hasattr(self.output_covariance, "predict"):
            kernel_root_operator_norms = np.array([
                    np.sqrt(
                        self.scalar_gram_matrix[iter_index, iter_index].item()
                    ) * compute_operator_norm(
                        self.root_output_covariance_estimator_grid[iter_index]
                    )
                    for iter_index, input_value in enumerate(self.inputs)
                ]
            )
            average_kernel_root_operator_norm = np.mean(kernel_root_operator_norms)
        else:
            output_covariance_operator_norm = compute_operator_norm(
                self.output_covariance
            )
            average_kernel_root_operator_norm = np.mean(
                np.sqrt(
                    np.diagonal(self.scalar_gram_matrix)
                    * output_covariance_operator_norm
                )
            )

        loss_ = maker(self.loss_name)(**self.loss_params)
        loss_Lipschitz_constant = loss_["lams"]["rho"]

        loss_stability_bounds = (
            loss_Lipschitz_constant**2
            * average_kernel_root_operator_norm
            / (2 * self.lam * sample_size)
        )

        return loss_stability_bounds

    def get_parameters(self):
        result = {
            "name": self.name,
            "lam":  self.lam,
            "kernel":  self.kernel,
            "kernel_max": self.kernel_max,
            "gamma": self.gamma,
            "degree":  self.degree,
            "coef0": self.coef0,
            "kernel_params": self.kernel_params,
            "loss_name": self.loss_name,
            "loss_params": self.loss_params,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "output_dimension": self.output_dimension,
            "scalar_gram_matrix": self.scalar_gram_matrix,
            "root_output_covariance_estimator_grid": self.root_output_covariance_estimator_grid,
            "model_weights": self.model_weights,
        }
        return result

    def initiate(self,
        name="KernelRegression",
        lam=0.5,
        kernel="linear",
        kernel_max=1.0,
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        loss_name="log_cosh",
        loss_params={"alpha": 1.0},
        solver="Newton-CG",
        max_iter=200,
        tol=1e-6,
        inputs=None,
        outputs=None,
        output_dimension=None,
        scalar_gram_matrix=None,
        root_output_covariance_estimator_grid=None,
        model_weights=None
    ):
        self.name = name
        self.lam = lam  # only one is supported
        self.kernel = kernel
        self.kernel_max = kernel_max
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

        self.loss_name = loss_name
        self.loss_params = loss_params

        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

        self.inputs = inputs
        self.outputs = outputs
        self.output_dimension = output_dimension
        self.scalar_gram_matrix = scalar_gram_matrix
        self.root_output_covariance_estimator_grid = root_output_covariance_estimator_grid
        self.model_weights = model_weights
