from .utils import *


class SplitConformalPredictor:
    """
    Performs split conformal prediction and
    is not responsible for the splitting of the data

    Example
    -------
    ```
        train_inputs, calibration_inputs, train_outputs, calibration_outputs = ...

        output_covariance_estimator = Covariance(...)
        predictor = KernelRegression(...)

        conformal_predictor = SplitConformalPredictor(
            predictor, output_covariance_estimator
        )

        conformal_predictor.fit(train_inputs, train_outputs)

        new_input, new_output = ...
        region_predictor = conformal_predictor.predict(
            new_input, calibration_inputs, calibration_outputs
        )
    ```
    """

    def __init__(
        self,
        predictor,
        output_covariance_estimator,
    ):
        self.predictor = predictor
        self.output_covariance_estimator = output_covariance_estimator
        self.name = "SplitCP"

    def fit(self, inputs, outputs):
        self.predictor.fit(inputs, outputs)
        self.output_covariance_estimator.fit(inputs, outputs, False)

    def predict(self, new_input, calibration_inputs, calibration_outputs):
        calibration_predictions = self.predictor.predict(calibration_inputs)
        new_prediction = self.predictor.predict(new_input)

        calibration_non_conformity_scores = compute_non_conformity_scores(
            calibration_inputs,
            calibration_outputs,
            calibration_predictions,
            self.output_covariance_estimator,
        )
        calibration_sample_size = calibration_non_conformity_scores.shape[0]

        def region_predictor(confidence_control_level):
            quantile_level = compute_quantile_level(
                calibration_sample_size, confidence_control_level
            )
            quantile_value = np.quantile(
                calibration_non_conformity_scores, quantile_level, method="higher"
            )
            return (
                new_prediction,
                quantile_value,
                self.output_covariance_estimator.predict(new_input),
            )

        return region_predictor
