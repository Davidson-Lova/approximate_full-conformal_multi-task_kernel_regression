# %%
import json
import pickle
import sys

from tqdm import tqdm

sys.path.append("..")

# %%
from generate_data import *
from plot_data import *

# %%
from src.conformal_prediction.global_ellipsoid_cp import (
    GlobalEllipsoidConformalPredictor,
)
from src.models.kernel_regression import KernelRegression
from src.models.covariance import Covariance


# %%
def reader(path_params):
    with open(path_params, "r") as file:
        params = json.load(file)
    print(params)
    return params


def load_list_with_pickle(file_name):
    with open(file_name, "rb") as file:
        result = pickle.load(file)
    return result


def save_list_with_pickle(my_list, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(my_list, file)


class RadiusTransformation2:
    def __init__(self, d, beta=None):
        if beta is None:
            beta = np.random.randn(d)
        self.beta = beta

    def get(self, x):
        return (
            ((np.linalg.norm(x)) / 2.0 + (np.dot(self.beta, x)) ** 2 / 10) / 2
        ) / 0.5 + 0.15


class NonLinearFunction2:
    def __init__(self, d, k, beta=None):
        if beta is None:
            beta = np.random.randn(d, k)
        self.beta = beta
        self.proj = np.zeros((d, k))
        self.proj[0, 0] = 1.0
        self.proj[1, 1] = 1.0

    def get(self, x):
        nonlinear_term = (
            np.sin(np.dot(x, self.beta))
            + 0.5 * np.tanh(np.dot(x**2, self.beta))
            + np.dot(x, self.proj)
        )
        return nonlinear_term * 2


def make_data_generator(
    sample_size, input_dimension, output_dimension, covariance_matrix, pert, n_anchors
):
    f_star = NonLinearFunction2(input_dimension, output_dimension)
    radius = RadiusTransformation2(input_dimension)
    local_perturbation = LocalPerturbation(
        input_dimension,
        output_dimension,
        n_anchors=n_anchors,
        radius_transformation=radius,
        seed=None,
    )
    data_generator = DataGenerator(
        input_dimension,
        output_dimension,
        pert,
        f_star=f_star,
        local_perturbation=local_perturbation,
        covariance_matrix=np.array(covariance_matrix),
        bias=False,
        seed=None,
    )

    return data_generator


def run(params_path, result_path):
    params = reader(params_path)
    predictors = {
        lam: KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam,
        )
        for lam in params["lams"]
    }
    new_inputs = []
    new_outputs = []

    region_predictors = {lam: [] for lam in params["lams"]}
    control_levels = np.exp2(np.linspace(np.log2(2 / params["data"]["sample_size"]), np.log2(0.25), 10))
    prediction_regions = {
        lam: {control_level: [] for control_level in control_levels}
        for lam in params["lams"]
    }

    output_covariance_estimator = Covariance(
        regularization=params["regularization"], mode="global"
    )

    for iter_id in tqdm(range(params["rep_number"])):
        data_generator = make_data_generator(**params["data"])
        inputs, outputs = data_generator.generate(params["data"]["sample_size"])
        new_input, new_output = data_generator.generate(1)
        new_inputs += [new_input]
        new_outputs += [new_output]

        for lam in params["lams"]:
            conformal_predictor = GlobalEllipsoidConformalPredictor(
                predictors[lam], output_covariance_estimator
            )
            region_predictor = conformal_predictor.fit_predict(
                new_input, inputs, outputs
            )
            region_predictors[lam] += [region_predictor]
            for control_level in control_levels:
                prediction_regions[lam][control_level] += [
                    region_predictor["upper"](control_level)
                ]

        results = {
            "new_inputs": new_inputs,
            "new_outputs": new_outputs,
            "prediction_regions": prediction_regions,
        }
        save_list_with_pickle(results, result_path + "result.pkl")

    return


params_path = "params/evaluate_coverage_global.json"
result_path = "results/evaluate_coverage/global/"

run(params_path, result_path)
