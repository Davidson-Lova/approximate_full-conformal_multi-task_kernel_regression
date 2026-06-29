from .utils import *
from scipy.optimize import root_scalar


class LocalEllipsoidConformalPredictor:
    """
    Performs Local Ellipsoid conformal prediction

    Example
    -------
    ```
        inputs, outputs = ...

        output_covariance_estimator = Covariance(...)
        predictor = KernelRegression(...)

        conformal_predictor = LocalEllipsoidConformalPredictor(
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

    def fit_predict(self, new_input, inputs, outputs):
        sample_size = inputs.shape[0]

        self.predictor.lam = self.predictor.lam * ((sample_size + 1) / sample_size)
        self.predictor.fit(inputs, outputs)

        augmented_inputs = np.concatenate((inputs, new_input), axis=0)
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

        new_r_value, r_values = compute_r_values(
            new_input, inputs, self.output_covariance_estimator
        )

        predictor_stability_bound = self.predictor.compute_predictor_stability_bound(new_input)
        print("predictor_stability: ", predictor_stability_bound)
        print("alignments: ",  non_conformity_score_stability_bounds / predictor_stability_bound)
        print("ratio r_value",[np.sqrt(new_r_value / r_value) for r_value in r_values])

        upper_non_conformity_scores = [
            np.sqrt(new_r_value / r_value) * (score + bound)
            for r_value, score, bound in zip(
                r_values, non_conformity_scores, non_conformity_score_stability_bounds
            )
        ]

        t_value = compute_t_value(
            new_input, new_prediction, self.output_covariance_estimator
        )

        lower_bound, df_lower_bound, d2f_lower_bound = make_lower_bound(
            new_non_conformity_score_stability_bound, t_value, 1 - new_r_value
        )

        def upper_region_predictor(confidence_control_level):
            quantile_level = compute_quantile_level(
                sample_size, confidence_control_level
            )

            upper_quantile_value = np.quantile(
                upper_non_conformity_scores, quantile_level, method="higher"
            )
            if upper_quantile_value >= (1 / (1 - new_r_value)) ** 0.5:
                quantile_value = np.inf
            else:
                initial_quantile_value = (
                    upper_quantile_value + new_non_conformity_score_stability_bound
                )
                root_res = root_scalar(
                    f=lambda x: (lower_bound(x) - upper_quantile_value),
                    method="newton",
                    fprime=df_lower_bound,
                    fprime2=d2f_lower_bound,
                    x0=initial_quantile_value,
                )
                quantile_value = root_res.root
                print(root_res.converged)
                print(root_res.flag)

            return (
                new_prediction,
                quantile_value,
                self.output_covariance_estimator.predict(new_input),
            )

        return {"upper": upper_region_predictor}
