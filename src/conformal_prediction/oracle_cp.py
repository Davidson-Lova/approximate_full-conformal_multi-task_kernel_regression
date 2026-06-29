from .utils import *


class OracleConformalPredictor:
    """
    Performs oracle conformal prediction

    Example
    -------
    ```
        inputs, outputs = ...

        output_covariance_estimator = Covariance(...)
        predictor = KernelRegression(...)

        conformal_predictor = OracleConformalPredictor(
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
        self.name = "OracleCP"

    def fit_predict(self, new_input, new_output, inputs, outputs):
        sample_size = inputs.shape[0]
        augmented_inputs = np.concatenate((inputs, new_input), axis=0)
        augmented_outputs = np.concatenate((outputs, new_output), axis=0)

        self.predictor.fit(augmented_inputs, augmented_outputs)
        self.output_covariance_estimator.fit(augmented_inputs, augmented_outputs)

        predictions = self.predictor.predict(inputs)
        new_prediction = self.predictor.predict(new_input)

        non_conformity_scores = compute_non_conformity_scores(
            inputs, outputs, predictions, self.output_covariance_estimator
        )

        def region_predictor(confidence_control_level):
            quantile_level = compute_quantile_level(
                sample_size, confidence_control_level
            )
            quantile_value = np.quantile(
                non_conformity_scores, quantile_level, method="higher"
            )
            return (
                new_prediction,
                quantile_value,
                self.output_covariance_estimator.predict(new_input),
            )

        return region_predictor
