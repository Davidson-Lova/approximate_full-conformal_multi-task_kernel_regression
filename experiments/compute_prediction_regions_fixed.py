# %%
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import json
from tqdm import tqdm

import sys

sys.path.append("..")

# %%
from utils import *
from generate_data import *
from plot_data import *
from time import time

# %%
from src.conformal_prediction.split_cp import SplitConformalPredictor
from src.conformal_prediction.oracle_cp import OracleConformalPredictor
from src.conformal_prediction.stable_cp import StableConformalPredictor
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


def run(params_path, input_path, result_path):
    params = reader(params_path)
    lam = load_list_with_pickle(input_path + "lam.pkl")
    result = {
        "new_inputs": [],
        "new_outputs": [],
        "oracle_prediction_regions": [],
        "split_prediction_regions": [],
        "upper_prediction_regions": [],
        "oracle_prediction_run_times": [],
        "split_prediction_run_times": [],
        "upper_prediction_run_times": [],
    }

    for iter in tqdm(range(params["rep_number"])):
        predictor_oracle = KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam["lam_upper"],
        )
        predictor_upper = KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam["lam_upper"],
        )
        predictor_split = KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam["lam_split"],
        )

        output_covariance_estimator = Covariance(
            covariance_estimator=np.array(params["data"]["covariance_matrix"]), mode="fixed"
        )

        data_generator = make_data_generator(**params["data"])
        inputs, outputs = data_generator.generate(params["data"]["sample_size"])
        new_input, new_output = data_generator.generate(1)

        result["new_inputs"] += [new_input]
        result["new_outputs"] += [new_output]

        oracle_conformal_predictor = OracleConformalPredictor(
            predictor_oracle, output_covariance_estimator
        )
        train_inputs, calibration_inputs, train_outputs, calibration_outputs = (
            train_test_split(inputs, outputs, random_state=0)
        )
        split_conformal_predictor = SplitConformalPredictor(
            predictor_split, output_covariance_estimator
        )
        conformal_predictor = StableConformalPredictor(
            predictor_upper, output_covariance_estimator
        )

        tic = time()
        oracle_region_predictor = oracle_conformal_predictor.fit_predict(
            new_input, new_output, inputs, outputs
        )
        oracle_prediction_region = oracle_region_predictor(
            params["confidence_control_level"]
        )
        tac = time()
        result["oracle_prediction_regions"] += [oracle_prediction_region]
        result["oracle_prediction_run_times"] += [tac - tic]

        tic = time()
        split_conformal_predictor.fit(train_inputs, train_outputs)
        split_region_predictor = split_conformal_predictor.predict(
            new_input, calibration_inputs, calibration_outputs
        )
        split_prediction_region = split_region_predictor(
            params["confidence_control_level"]
        )
        tac = time()
        result["split_prediction_regions"] += [split_prediction_region]
        result["split_prediction_run_times"] += [tac - tic]

        tic = time()
        region_predictor = conformal_predictor.fit_predict(new_input, inputs, outputs)
        upper_prediction_region = region_predictor["upper"](
            params["confidence_control_level"]
        )
        tac = time()
        result["upper_prediction_regions"] += [upper_prediction_region]
        result["upper_prediction_run_times"] += [tac - tic]

        save_list_with_pickle(result, result_path + "result.pkl")


# %%
indices = [1]
for index in indices:
    print("Starting index {}\n".format(index))

    params_path = "params/n{}.json".format(index)
    input_path = "results/best_predictor/n{}/".format(index)
    result_path = "results/prediction_regions/n{}/fixed/".format(index)
    run(params_path, input_path, result_path)

    print("Ending index {}\n".format(index))
