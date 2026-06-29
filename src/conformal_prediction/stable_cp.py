from .utils import *


class StableConformalPredictor:
    """
    Performs stable conformal prediction

    Example
    -------
    ```
        inputs, outputs = ...

        output_covariance_estimator = Covariance(...)
        predictor = KernelRegression(...)

        conformal_predictor = StableConformalPredictor(
            predictor, output_covariance_estimator
        )

        new_input, new_output = ...

        region_predictor = conformal_predictor.fit_predict(
            new_input, new_output,
            inputs, outputs
        )
    ```
    """

    def __init__(self, predictor, output_covariance_estimator):
        self.predictor = predictor
        self.output_covariance_estimator = output_covariance_estimator
        self.name = "StableCP"

    def fit_predict(self, new_input, inputs, outputs):
        augmented_inputs = np.concatenate((inputs, new_input), axis=0)
        sample_size = inputs.shape[0]

        self.predictor.lam = self.predictor.lam * ((sample_size + 1) / sample_size)
        self.predictor.fit(inputs, outputs)
        self.output_covariance_estimator.fit(augmented_inputs, outputs, diminished=True)

        predictions = self.predictor.predict(inputs)
        new_prediction = self.predictor.predict(new_input)

        non_conformity_scores = compute_non_conformity_scores(
            inputs, outputs, predictions, self.output_covariance_estimator
        )

        (
            new_non_conformity_score_stability_bound,
            non_conformity_score_stability_bounds,
        ) = compute_non_conformity_score_stability_bounds(
            new_input, inputs, self.predictor, self.output_covariance_estimator
        )

        upper_non_conformity_scores = [
            score + bound
            for score, bound in zip(
                non_conformity_scores, non_conformity_score_stability_bounds
            )
        ]

        def upper_region_predictor(confidence_control_level):
            quantile_level = compute_quantile_level(
                sample_size, confidence_control_level
            )

            if self.output_covariance_estimator.mode == "fixed":
                quantile_value = (
                    np.quantile(
                        upper_non_conformity_scores, quantile_level, method="higher"
                    )
                    + new_non_conformity_score_stability_bound
                )

            return (
                new_prediction,
                quantile_value,
                self.output_covariance_estimator.predict(new_input),
            )

        lower_non_conformity_scores = [
            score - bound
            for score, bound in zip(
                non_conformity_scores, non_conformity_score_stability_bounds
            )
        ]

        def lower_region_predictor(confidence_control_level):
            quantile_level = compute_quantile_level(
                sample_size, confidence_control_level
            )

            quantile_value = (
                np.quantile(
                    lower_non_conformity_scores, quantile_level, method="higher"
                )
                - new_non_conformity_score_stability_bound
            )

            return (
                new_prediction,
                quantile_value,
                self.output_covariance_estimator.predict(new_input),
            )

        self.predictor.lam = self.predictor.lam * (sample_size / (sample_size + 1))
        return {"upper": upper_region_predictor, "lower": lower_region_predictor}
