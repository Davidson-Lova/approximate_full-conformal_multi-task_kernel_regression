import numpy as np
import portion as P
from scipy.optimize import root_scalar
from scipy.special import gamma


def mahalanobis_norm(vec, covariance_matrix):
    vec = vec.reshape(-1, 1)
    invCovVec = np.linalg.solve(covariance_matrix, vec)
    invCovVec = invCovVec.reshape(-1, 1)
    return np.sqrt(vec.T @ invCovVec)


def compute_operator_norm(matrix):
    operator_norm = np.linalg.svdvals(matrix)[0].item()
    return operator_norm


def compute_alignment(inner_matrix, outer_matrix):
    U, outer_singular_values, Vh = np.linalg.svd(outer_matrix)
    outer_inverse_root = np.sqrt(1 / outer_singular_values).reshape(-1, 1)
    alignment = compute_operator_norm(
        U @ (outer_inverse_root * (Vh @ inner_matrix @ U) * outer_inverse_root.T) @ Vh
    )
    return alignment


def compute_non_conformity_scores(
    inputs,
    outputs,
    predictions,
    output_covariance_estimator,
):
    result = np.array(
        [
            mahalanobis_norm(
                output - prediction,
                output_covariance_estimator.predict(input.reshape(1, -1)),
            ).item()
            for input, output, prediction in zip(inputs, outputs, predictions)
        ]
    )
    return result


def compute_r_values(new_input, inputs, output_covariance_estimator):
    new_r_value = output_covariance_estimator.r_estimator(new_input)
    r_values = np.array(
        [output_covariance_estimator.r_estimator(input) for input in inputs]
    )
    return new_r_value, r_values


def compute_t_value(test_input, new_prediction, output_covariance_estimator):
    result = mahalanobis_norm(
        new_prediction.T - output_covariance_estimator.predict_mean(test_input),
        output_covariance_estimator.predict(test_input),
    )
    return result


def make_lower_bound(tau, t, w):
    def lower_bound(x):
        num = x - tau
        den = (w * ((x + t) ** 2) + 1) ** 0.5
        return num / den

    def df_lower_bound(x):
        num = w * (tau + t) * (x + t) + 1
        den = (w * ((x + t) ** 2) + 1) ** (1.5)
        return num / den

    def d2f_lower_bound(x):
        num1 = w * (tau + t) * (w * ((x + t) ** 2) + 1)
        num2 = -3 * w * (x + t) * (w * (tau + t) * (x + t) + 1)
        den = (w * ((x + t) ** 2) + 1) ** (2.5)
        return (num1 + num2) / den

    return lower_bound, df_lower_bound, d2f_lower_bound


def make_upper_bound(tau, t, w, a):
    def upper_bound(x):
        return (x - tau) * (w * ((a * x + t) ** 2) + 1) ** 0.5

    def df_upper_bound(x):
        num = w * (a * x + t) * (a * (2 * x + tau) + t) + 1
        den = (w * ((a * x + t) ** 2) + 1) ** 0.5
        return num / den

    def d2f_upper_bound(x):
        num = (
            a
            * w
            * (
                a
                * (
                    x * (2 * w * ((a**2) * (x**2) + 3 * a * t * x + 3 * (t**2)) + 3)
                    + tau
                )
                + 2 * t * ((t**2) * w + 1)
            )
        )
        den = (w * ((x + t) ** 2) + 1) ** (1.5)
        return num / den

    return upper_bound, df_upper_bound, d2f_upper_bound


def compute_coverage(prediction_region, new_output):
    oracle_score = mahalanobis_norm(
        prediction_region[0] - new_output, prediction_region[-1]
    ).item()
    return oracle_score <= prediction_region[1]


def compute_quantile_level(sample_size, confidence_control_level):
    return np.ceil((sample_size + 1) * (1 - confidence_control_level)) / sample_size


def compute_non_conformity_score_stability_bounds(
    new_input, inputs, predictor, output_covariance_matrix
):
    scalar_gram_diag = np.diagonal(predictor.scalar_gram_matrix)
    predictor_stability_bound = predictor.compute_predictor_stability_bound(new_input)

    if hasattr(predictor.output_covariance, "predict"):
        sample_size = inputs.shape[0]
        kernel_score_covariance_alignments = np.array(
            [
                np.sqrt(
                    scalar_gram_diag[j]
                    * compute_alignment(
                        output_covariance_matrix.predict(inputs[j]),
                        predictor.root_output_covariance_estimator_grid[j]
                        @ predictor.root_output_covariance_estimator_grid[j]
                    )
                ).item()
                for j in range(sample_size)
            ]
        )

        new_scalar_gram_value = predictor._get_kernel(new_input).item()
        new_non_conformity_score_stability_bound = (
            np.sqrt(
                new_scalar_gram_value
                * compute_alignment(
                    predictor.output_covariance.predict(new_input),
                    output_covariance_matrix.predict(new_input)
                )
            ).item()
            * predictor_stability_bound
        )
    else:
        kernel_score_covariance_alignments = np.array(
            [
                np.sqrt(
                    scalar_gram_value
                    * compute_alignment(
                        predictor.output_covariance, output_covariance_matrix.predict(input)
                    )
                ).item()
                for input, scalar_gram_value in zip(inputs, scalar_gram_diag)
            ]
        )

        new_scalar_gram_value = predictor._get_kernel(new_input).item()
        new_non_conformity_score_stability_bound = (
            np.sqrt(
                new_scalar_gram_value
                * compute_alignment(
                    predictor.output_covariance, output_covariance_matrix.predict(new_input)
                )
            ).item()
            * predictor_stability_bound
        )

    non_conformity_score_stability_bounds = (
        kernel_score_covariance_alignments * predictor_stability_bound
    )

    return (
        new_non_conformity_score_stability_bound,
        non_conformity_score_stability_bounds,
    )


def compute_volume_ellipsoid(covariance, radius):
    p = covariance.shape[0]
    det_root = np.linalg.det(covariance) ** 0.5
    return det_root * (np.pi ** (p / 2)) * (radius**p) / gamma(p / 2 + 1)


def inter_finder(objective, y_min, y_max, y_hat):
    """
    Computes the interval contains within (y_min, y_max)
    where the objective function is non-negative

    Args:
        objective (callable): objective function
        y_min (float): output sample minimum
        y_max (float): output sample maximum
        y_hat (float): prediction

    Returns:
        (P.open): interval where the objective function is non-negative

    """
    lb = y_min
    ub = y_max

    fy_min = objective(y_min)
    fy_max = objective(y_max)
    fy_hat = objective(y_hat)
    fy_0 = objective(0)

    if fy_min * fy_max < 0:
        mid_finder = root_scalar(objective, bracket=[y_min, y_max])
        if fy_min < 0:
            lb = mid_finder.root
        else:
            ub = mid_finder.root
        return P.closed(lb, ub)
    else:
        if fy_min >= 0:
            if fy_min * fy_0 < 0:
                mid_finder_left = root_scalar(objective, bracket=[y_min, 0])
                mid_finder_right = root_scalar(objective, bracket=[0, y_max])
                return P.closed(lb, mid_finder_left.root) | P.closed(
                    mid_finder_right.root, ub
                )
            else:
                return P.closed(lb, ub)
        else:
            if fy_min * fy_hat < 0:
                lb_finder = root_scalar(objective, bracket=[y_min, y_hat])
                ub_finder = root_scalar(objective, bracket=[y_hat, y_max])
                return P.closed(lb_finder.root, ub_finder.root)
            else:
                print("Prediction not in the predictive region")
                return P.open(0, 0)


def interval_length(interval):
    """
    Compute the length of an interval

    Args:
        interval (P.closed): interval

    Returns:
        (float): interval length
    """
    if interval.empty:
        return 0
    length = 0
    for subinterval in interval:
        length += subinterval.upper - subinterval.lower
    return length
