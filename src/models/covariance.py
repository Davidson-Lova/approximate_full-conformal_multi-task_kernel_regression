import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize(weights):
    return weights / np.sum(weights)


def estimate_mean(outputs, weights):
    sample_size = outputs.shape[0]
    if weights is None:
        weights = np.ones(sample_size).reshape(-1, 1)
    weights = normalize(weights)
    mean = np.sum(weights * outputs, axis=0).reshape(-1, 1)
    return mean, weights


def estimate_covariance_and_mean(outputs, regularization, weights):
    mean, weights = estimate_mean(outputs, weights)
    covariance = outputs.T @ (weights * outputs) - mean @ mean.T
    covariance.flat[:: covariance.shape[1] + 1] += regularization
    return covariance, mean


def make_covariance_and_mean_estimator(
    outputs,
    regularization,
    neighbor_predictor,
):
    def mean_estimator(input):
        if input.ndim == 1:
            input = input.reshape(1, -1)
        weights = neighbor_predictor.kneighbors_graph(input).T.toarray()
        mean, _ = estimate_mean(outputs, weights)
        return mean

    def covariance_estimator(input):
        if input.ndim == 1:
            input = input.reshape(1, -1)
        weights = neighbor_predictor.kneighbors_graph(input).T.toarray()
        estimated_covariance, _ = estimate_covariance_and_mean(
            outputs, regularization, weights
        )
        return estimated_covariance

    return (covariance_estimator, mean_estimator)


def make_diminished_covariance_and_mean_estimator(
    outputs,
    regularization,
    neighbor_predictor,
):
    def r_estimator(input):
        if input.ndim == 1:
            input = input.reshape(1, -1)
        weights = neighbor_predictor.kneighbors_graph(input).T.toarray()
        fact = 1 - normalize(weights)[-1, :].item()
        return fact, weights

    def mean_estimator(input):
        if input.ndim == 1:
            input = input.reshape(1, -1)
        weights = neighbor_predictor.kneighbors_graph(input).T.toarray()
        diminished_weights = weights[:-1, :]
        mean, _ = estimate_mean(outputs, diminished_weights)
        return mean

    def covariance_estimator(input):
        fact, weights = r_estimator(input)
        diminished_weights = weights[:-1, :]
        estimated_covariance, _ = estimate_covariance_and_mean(
            outputs, regularization * fact, diminished_weights
        )
        return estimated_covariance

    return (
        covariance_estimator,
        mean_estimator,
        lambda input: r_estimator(input)[0],
    )


class Covariance:
    """
    Estimates output covariance matrix

    Example
    -------
    ```
        inputs, outputs = ...

        output_covariance_estimator = Covariance(...)

        output_covariance_estimator.fit(inputs, outputs)

        new_input, new_output = ...
        estimated_output_covariance = output_covariance_estimator.predict(new_input)
    ```
    """

    def __init__(
        self,
        covariance_estimator=None,
        mode="fixed",
        regularization=1.0,
        neighbor_number=1,
        weights=None,
    ):
        self.covariance_estimator = covariance_estimator
        self.mean_estimator = None
        self.r_estimator = None

        self.mode = mode
        if self.mode not in ["fixed", "global", "local"]:
            if self.covariance_estimator is None:
                self.mode = "global"
                print("Mode set to global.")
            else:
                self.mode = "fixed"
                print("Mode set to fixed.")

        self.regularization = regularization
        self.neighbor_number = neighbor_number
        if self.mode == "local":
            self.neighbor_predictor = NearestNeighbors(
                n_neighbors=self.neighbor_number, algorithm="ball_tree"
            )

        self.weights = weights

    def fit(self, inputs, outputs, diminished=False):
        if self.mode == "global":
            sample_size = outputs.shape[0]
            fact = ((sample_size + 1) / sample_size) if diminished else 1.0
            self.covariance_estimator, self.mean_estimator = (
                estimate_covariance_and_mean(
                    outputs, self.regularization * fact, self.weights
                )
            )

        elif self.mode == "local":
            self.neighbor_predictor.fit(inputs)
            if diminished:
                self.covariance_estimator, self.mean_estimator, self.r_estimator = (
                    make_diminished_covariance_and_mean_estimator(
                        outputs, self.regularization, self.neighbor_predictor
                    )
                )
            else:
                self.covariance_estimator, self.mean_estimator = (
                    make_covariance_and_mean_estimator(
                        outputs, self.regularization, self.neighbor_predictor
                    )
                )

    def predict(self, new_input):
        if self.covariance_estimator is None:
            raise ValueError("Please fit the model first or provide a value.")
        if self.mode in ["fixed", "global"]:
            return self.covariance_estimator
        elif self.mode == "local":
            return self.covariance_estimator(new_input)

    def predict_mean(self, new_input):
        if self.mean_estimator is None:
            raise ValueError("Please fit the model first or provide a value.")
        if self.mode in ["fixed", "global"]:
            return self.mean_estimator
        elif self.mode == "local":
            return self.mean_estimator(new_input)
