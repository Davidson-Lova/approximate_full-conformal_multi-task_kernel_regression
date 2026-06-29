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
    input_dimension, output_dimension, covariance_matrix, pert, n_anchors
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
    sample_sizes = np.flip(
        np.int64(
            np.exp2(
                np.linspace(
                    np.log2(params["min_sample_size"]),
                    np.log2(params["max_sample_size"]),
                    params["samples_number"],
                )
            )
        )
    )

    for index in [0, 1]:
        if params["lam_decay_rates"][index] == 0:
            lams = params["lam_0s"][index] * np.ones(params["samples_number"])
        else:
            lams = (
                params["lam_0s"][index]
                * (1 / sample_sizes) ** params["lam_decay_rates"][index]
            )

        output_covariance_estimator = Covariance(mode="global", regularization=1e-6)

        new_inputs = {sample_size: [] for sample_size in sample_sizes}

        new_outputs = {sample_size: [] for sample_size in sample_sizes}

        upper_prediction_regions = {sample_size: [] for sample_size in sample_sizes}

        lower_prediction_regions = {sample_size: [] for sample_size in sample_sizes}

        for sample_size, lam in zip(sample_sizes, lams):
            print(
                "################################### sample_size:{} #############################".format(
                    sample_size
                )
            )
            print(
                "################################### lam:{} ###################################".format(
                    lam
                )
            )

            for iter_id in tqdm(range(params["rep_number"])):
                data_generator = make_data_generator(**params["data"])
                inputs, outputs = data_generator.generate(sample_size)
                new_input, new_output = data_generator.generate(1)

                new_inputs[sample_size] += [new_input]
                new_outputs[sample_size] += [new_output]

                predictor = KernelRegression(
                    solver=params["solver"],
                    loss_name=params["loss_name"],
                    loss_params=params["loss_params"],
                    output_covariance=np.array(params["data"]["covariance_matrix"]),
                    kernel=params["kernel"],
                    kernel_max=params["kernel_max"],
                    lam=lam,
                )

                conformal_predictor = GlobalEllipsoidConformalPredictor(
                    predictor, output_covariance_estimator
                )
                print(predictor.lam)

                region_predictor = conformal_predictor.fit_predict(
                    new_input, inputs, outputs
                )
                upper_prediction_region = region_predictor["upper"](0.1)
                lower_prediction_region = region_predictor["lower"](0.1)
                upper_prediction_regions[sample_size] += [upper_prediction_region]
                lower_prediction_regions[sample_size] += [lower_prediction_region]

            results = {
                "new_inputs": new_inputs,
                "new_outputs": new_outputs,
                "upper_prediction_regions": upper_prediction_regions,
                "lower_prediction_regions": lower_prediction_regions,
            }

            save_list_with_pickle(results, result_path + "result_{}.pkl".format(index))
            print("Saving completed :)")


params_path = "params/evaluate_consistency_global.json"
result_path = "results/evaluate_consistency/global/"

run(params_path, result_path)
