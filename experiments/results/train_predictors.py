# %%
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import json

import sys

sys.path.append("..")

# %%
from utils import *
from generate_data import *
from plot_data import *

# %%
from src.models.kernel_regression import KernelRegression


# %%
def reader(path_params):
    with open(path_params, "r") as file:
        params = json.load(file)
    print(params)
    return params


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


def generate_data(
    sample_size, input_dimension, output_dimension, covariance_matrix, pert, n_anchors
):
    f_star = NonLinearFunction2(input_dimension, output_dimension)
    radius = RadiusTransformation2(input_dimension)
    local_perturbation = LocalPerturbation(
        input_dimension,
        output_dimension,
        n_anchors=n_anchors,
        radius_transformation=radius,
    )
    data_generator = DataGenerator(
        input_dimension,
        output_dimension,
        pert,
        f_star=f_star,
        local_perturbation=local_perturbation,
        covariance_matrix=np.array(covariance_matrix),
        bias=False,
        seed=42,
    )
    inputs, outputs = data_generator.generate(sample_size)
    return inputs, outputs


def run(params_path, result_path):
    params = reader(params_path)

    inputs, outputs = generate_data(**params["data"])
    train_inputs, calibration_inputs, train_outputs, calibration_outputs = (
        train_test_split(inputs, outputs, random_state=0)
    )
    dataset = {
        "inputs": inputs,
        "outputs": outputs,
        "train_inputs": train_inputs,
        "train_outputs": train_outputs,
        "calibration_inputs": calibration_inputs,
        "calibration_outputs": calibration_outputs,
    }
    save_list_with_pickle(dataset, result_path + "dataset.pkl")

    lams = np.exp2(
        np.linspace(
            np.log2(params["lam_min"]),
            np.log2(params["lam_max"]),
            params["lams_number"],
        )
    )
    predictors_upper = [
        KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam,
        )
        for lam in lams
    ]

    predictors_split = [
        KernelRegression(
            solver=params["solver"],
            loss_name=params["loss_name"],
            loss_params=params["loss_params"],
            output_covariance=np.array(params["data"]["covariance_matrix"]),
            kernel=params["kernel"],
            kernel_max=params["kernel_max"],
            lam=lam,
        )
        for lam in lams
    ]

    for index in range(params["lams_number"]):
        predictors_upper[index].fit(inputs, outputs)
        result_name = "predictor_upper_{}.pkl".format(index)
        save_list_with_pickle(predictors_upper[index], result_path + result_name)

        predictors_split[index].fit(train_inputs, train_outputs)
        result_name = "predictor_split_{}.pkl".format(index)
        save_list_with_pickle(predictors_split[index], result_path + result_name)

    return


# %%
indices = [1]
for index in indices:
    print("Starting index{}.\n".format(index))
    params_path = "params/n{}.json".format(index)
    result_path = "results/train_predictors/n{}/".format(index)
    run(params_path, result_path)
    print("Done with index{}.\n".format(index))
