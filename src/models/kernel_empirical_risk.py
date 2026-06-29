import numpy as np
from .losses import maker


def function_norm(model_weights, gram_matrix):
    if type(gram_matrix) is tuple:
        scalar_gram_matrix, output_covariance = gram_matrix
        sample_size = scalar_gram_matrix.shape[0]
        output_dimension = output_covariance.shape[0]
        model_weights = model_weights.reshape(
            (output_dimension, sample_size), order="F"
        ).T
        return np.linalg.trace(
            output_covariance @ (model_weights.T @ (scalar_gram_matrix @ model_weights))
        )
    else:
        return (model_weights.T @ (gram_matrix @ model_weights)).item()


def compute_emp_risk(loss, outputs, predictions, gram_matrix, lam, model_weights):
    return np.mean(
        [
            loss(output_point, prediction)
            for output_point, prediction in zip(outputs, predictions)
        ]
    ) + lam * function_norm(model_weights, gram_matrix)


def compute_grad_emp_risk(dloss, outputs, predictions, gram_matrix, lam):
    if type(gram_matrix) is tuple:
        scalar_gram_matrix, output_covariance = gram_matrix
        return (
            np.mean(
                [
                    np.outer(
                        scalar_gram_vec, output_covariance @ dloss(output, prediction)
                    )
                    for scalar_gram_vec, output, prediction in zip(
                        scalar_gram_matrix, outputs, predictions
                    )
                ],
                axis=0,
            )
            + 2 * lam * predictions
        ).T.ravel(order="F")
    else:
        sample_size, output_dimension = predictions.shape
        return np.mean(
            [
                gram_matrix[:, i * output_dimension : (i + 1) * output_dimension]
                @ dloss(output, prediction)
                for i, output, prediction in zip(
                    range(sample_size), outputs, predictions
                )
            ],
            axis=0,
        ) + 2 * lam * predictions.T.ravel(order="F")


def compute_hess_emp_risk(d2loss, outputs, predictions, gram_matrix, lam):
    if type(gram_matrix) is tuple:
        scalar_gram_matrix, output_covariance = gram_matrix
        return np.mean(
            [
                np.kron(gram_vec.reshape(-1, 1), output_covariance)
                @ d2loss(output, prediction)
                @ np.kron(gram_vec.reshape(1, -1), output_covariance)
                for gram_vec, output, prediction in zip(
                    scalar_gram_matrix, outputs, predictions
                )
            ],
            axis=0,
        ) + 2 * lam * np.kron(scalar_gram_matrix, output_covariance)
    else:
        sample_size, output_dimension = predictions.shape
        return (
            np.mean(
                [
                    gram_matrix[:, i * output_dimension : (i + 1) * output_dimension]
                    @ d2loss(output, prediction)
                    @ gram_matrix[i * output_dimension : (i + 1) * output_dimension, :]
                    for i, output, prediction in zip(
                        range(sample_size), outputs, predictions
                    )
                ],
                axis=0,
            )
            + 2 * lam * gram_matrix
        )


def predict(gram_matrix, model_weights, output_dimension):
    if type(gram_matrix) is tuple:
        scalar_gram_matrix, output_covariance = gram_matrix
        sample_size = scalar_gram_matrix.shape[0]
        model_weights = model_weights.reshape(
            (output_dimension, sample_size), order="F"
        ).T
        predictions = scalar_gram_matrix @ (model_weights @ output_covariance)

    else:
        sample_size = np.int64(gram_matrix.shape[0] / output_dimension)
        predictions = (
            (gram_matrix @ model_weights)
            .reshape((output_dimension, sample_size), order="F")
            .T
        )

    return predictions


class KernelEmpiricalRisk:
    def __init__(self, loss_name="log_cosh", loss_params={"alpha": 1.0}):
        self.loss_name = loss_name
        loss_ = maker(loss_name)(**loss_params)
        self.loss = loss_["f"]
        self.dloss = loss_["df"]
        self.d2loss = loss_["d2f"]

    def empirical_risk(self, model_weights, gram_matrix, outputs, lam):
        """Computes the regularized empirical risk

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        empirical_risk_ : float
            Weighted average of losses per sample, plus penalty.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        emp_risk = compute_emp_risk(
            self.loss, outputs, predictions, gram_matrix, lam, model_weights
        )
        return emp_risk

    def empirical_risk_gradient(self, model_weights, gram_matrix, outputs, lam):
        """Computes the regularized empirical risk, and its gradient w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        empirical_risk : float
            Weighted average of losses per sample, plus penalty.

        gradient : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        emp_risk = compute_emp_risk(
            self.loss, outputs, predictions, gram_matrix, lam, model_weights
        )
        grad = compute_grad_emp_risk(self.dloss, outputs, predictions, gram_matrix, lam)
        return emp_risk, grad

    def gradient(self, model_weights, gram_matrix, outputs, lam):
        """Computes the gradient of the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        gradient : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        grad = compute_grad_emp_risk(self.dloss, outputs, predictions, gram_matrix, lam)

        return grad

    def gradient_hessian(self, model_weights, gram_matrix, outputs, lam):
        """Computes the gradient and the hessian of the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        grad : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.

        hess : ndarray of shape (model_weights.shape, model_weights.shape)
             The hessian of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        grad = compute_grad_emp_risk(self.dloss, outputs, predictions, gram_matrix, lam)
        hess = compute_hess_emp_risk(
            self.d2loss, outputs, predictions, gram_matrix, lam
        )

        return grad, hess

    def hessian(self, model_weights, gram_matrix, outputs, lam):
        """Computes the gradient and the hessian of the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        hess : ndarray of shape (model_weights.shape, model_weights.shape)
             The hessian of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        hess = compute_hess_emp_risk(
            self.d2loss, outputs, predictions, gram_matrix, lam
        )

        return hess

    def gradient_hessian_product(self, model_weights, gram_matrix, outputs, lam):
        """Computes the gradient and the hessian vector product function of
        the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------

        grad : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.

        hessp : callable
            The hessian vector product function of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)
        grad = compute_grad_emp_risk(self.dloss, outputs, predictions, gram_matrix, lam)

        def hessp(vec):
            if type(gram_matrix) is tuple:
                scalar_gram_matrix, output_covariance = gram_matrix
                sample_size, output_dimension = outputs.shape
                vec = vec.reshape((output_dimension, sample_size), order="F").T
                return (
                    np.mean(
                        [
                            np.outer(gram_vec, gram_vec)
                            @ (vec @ output_covariance)
                            @ (self.d2loss(output, prediction) @ output_covariance)
                            for gram_vec, output, prediction in zip(
                                scalar_gram_matrix, outputs, predictions
                            )
                        ],
                        axis=0,
                    )
                    + 2 * lam * scalar_gram_matrix @ (vec @ output_covariance)
                ).T.ravel(order="F")
            else:
                sample_size, output_dimension = outputs.shape
                return (
                    np.mean(
                        [
                            gram_matrix[
                                :, i * output_dimension : (i + 1) * output_dimension
                            ]
                            @ self.d2loss(output, prediction)
                            @ (
                                gram_matrix[
                                    i * output_dimension : (i + 1) * output_dimension, :
                                ]
                                @ vec
                            )
                            for i, output, prediction in zip(
                                range(sample_size), outputs, predictions
                            )
                        ],
                        axis=0,
                    )
                    + 2 * lam * gram_matrix @ vec
                )

        return grad, hessp

    def hessian_product(self, model_weights, vector, gram_matrix, outputs, lam):
        """Computes the hessian vector product function of
        the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : (sample_size * output_dimension, )
            Model parameters

        vector : (sample_size * output_dimension, )
            Model parameters

        gram_matrix : ndarray of shape (sample_size, sample_size)

        outputs : ndarray of shape (sample_size, output_dimension)

        lam : float
            The regularization parameter

        Returns
        -------
        hessp : callable
            The hessian vector product function of the regularized empirical risk.
        """
        output_dimension = outputs.shape[1]
        predictions = predict(gram_matrix, model_weights, output_dimension)

        if type(gram_matrix) is tuple:
            scalar_gram_matrix, output_covariance = gram_matrix
            sample_size, output_dimension = outputs.shape
            vector = vector.reshape((output_dimension, sample_size), order="F").T
            hessp = (
                np.mean(
                    [
                        np.outer(gram_vec, gram_vec)
                        @ (vector @ output_covariance)
                        @ (self.d2loss(output, prediction) @ output_covariance)
                        for gram_vec, output, prediction in zip(
                            scalar_gram_matrix, outputs, predictions
                        )
                    ],
                    axis=0,
                )
                + 2 * lam * scalar_gram_matrix @ (vector @ output_covariance)
            ).T.ravel(order="F")
        else:
            sample_size, output_dimension = outputs.shape
            hessp = (
                np.mean(
                    [
                        gram_matrix[
                            :, i * output_dimension : (i + 1) * output_dimension
                        ]
                        @ self.d2loss(output, prediction)
                        @ (
                            gram_matrix[
                                i * output_dimension : (i + 1) * output_dimension, :
                            ]
                            @ vector
                        )
                        for i, output, prediction in zip(
                            range(sample_size), outputs, predictions
                        )
                    ],
                    axis=0,
                )
                + 2 * lam * gram_matrix @ vector
            )

        return hessp
