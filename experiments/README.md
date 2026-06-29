# Choosing the regularization parameters
You can run the following command,
```sh
$ python train_predictors.py
```
which computes a set of predictors given the full data set and another set of predictors given the split data set. Each set contains predictors integratting different regularization parameters $\lambda$. It has a correspoding `.json` file in the `\params` folder (for example `n1.json`).
Its outputs will be contained in `results/train_predictors/n1/`.

Then, you can use the notebook `best_predictor.ipynb`
to visualize the results and to choose the best predictor
given the full data set and given the split data set.


# StableCP
## Evolution of the empirical coverage probability
You can run the following command,
```sh
$ python evaluate_coverage_fixed.py
```
to compute an empirical estimation (across 100 repetitions)
of the coverage probability of the StableCP-region
across a range of values control levels $\alpha$
and for some regularization values $\lambda$.
It has a correspoding `.json` file in the `\params` folder.
Its outputs will be contained in `results/evaluate_coverage/fixed/`.


Then, you can use the notebook `evaluate_coverage_fixed.ipynb` to visualize the results.


> It takes about an hour for the script to end on my computer so one must be prepared to wait and check the duration displayed by `tqdm`.


## Evolution of empirical upper-bound on the thickness


`evaluate_consistency_fixed.py` 

You can run the following command,
```sh
$ python evaluate_consistency_fixed.py
```
to computes the empirical upper-bound on the thickness
across a range of sample-sizes for a fixed regularization parameter $\lambda$ (and then, for $\lambda \propto \frac{1}{\sqrt{n}}$).
It has a correspoding `.json` file in the `\params` folder.
Its outputs will be contained in `results/evaluate_consistency/fixed/`.

Then, you can use the notebook `evaluate_consistency_fixed.ipynb` to visualize the results.

> It takes about four hours for the script to end on my computer so one must be prepared to wait and check the duration displayed by `tqdm`.


## Comparing StableCP and SplitCP
Before comparing the two methods,
one must first choose the regularization parameter
corresponding to each method. So, one must have run the command at the top of this file.

You can run the following command,
```sh
$ python compute_prediction_regions_fixed.py
```
to compute, across 100 repetitions, the StableCP-region, the SplitCP-region and the OracleCP-region. It has a correspoding `.json` file in the `\params` folder (for example `n1.json`).

Then, you can use the notebook `compare_cp_fixed.ipynb` to visualize the results.


# G-EllipsoidCP

See the previous section and replace all `fixed` with `global` (for example `evaluate_coverage_global.py` instead of `evaluate_coverage_fixed.py`).