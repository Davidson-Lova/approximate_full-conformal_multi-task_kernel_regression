# Approximate full-conformal multi-task kernel regression

## How to start
You can clone or download this repo.
Then, you can run the following command,
```sh
$ python -m venv .venv
```
to create a virtual environment
and then activate said environment
using
```sh
$ .venv/Scripts/activate
```
if you are on windows.
(Its different if you are on mac or linux so do check the relevant documentation.)

After your virtual environment is activated,
you can run the following command,
```sh
$ pip install -r requirements.txt
```
to install all the required packages.


## Structure
After all the required packages are installed, you can go to the folder `experiments\`
if you wish to reproduce the results in the paper.
Wherein lies a dedicated `README.md` file.

`src\conformal_prediction\` contains the implementation of conformal prediction methods.

`src\models\` contains an implementation of
multi-task regression with a reproducing kernel.

## If there is any issue

Do contact me at `davidson-lova.razafindrakoto@proton.me`.