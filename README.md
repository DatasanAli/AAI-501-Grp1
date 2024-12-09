# Credit Card Defaults Prediction Models
The problem we’re addressing is predicting credit card defaults for a financial institution, aiming
to minimize the risk of approving loans or credit lines to high-risk customers. This requires
accurately predicting the probability of default while providing interpretable insights into what
drives these predictions, which is crucial for regulatory and operational transparency.
We plan to investigate several machine learning and AI algorithms:

1. Artificial Neural Network (ANN) integrated with the Sorting Smoothing Method (SSM)
can enhance accuracy in predicting credit card defaults and estimating default
probabilities by effectively capturing complex nonlinear relationships in the data while
reducing noise and improving model stability
2. XGBoost provides robust and efficient classification for credit card default prediction,
while pairing it with SHAP (SHapley Additive ExPlanations) ensures interpretability by
highlighting the importance and contribution of each feature to the model's predictions.
3. Naive Bayes can be used to predict credit card defaults by calculating the probability of
default based on features like payment history, billing amounts, and customer
demographics, under the assumption of conditional independence between these
features

The system will handle data preprocessing, model training, evaluation, and interpretability
outputs. This includes balancing the dataset using SMOTE techniques, tuning parameters, and
comparing models on metrics like accuracy, recall, and interpretability.

## Dataset
You can find the dataset stored in this repo, however it comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

There are 30,000 instances in this dataset with 23 features. We rename the features for redability. The data was aggregated in .

### Features

This table lists the old feature names and their corresponding new names used in the dataset:

| Old Feature Name | New Feature Name      | Description                                                                 |
|-------------------|-----------------------|-----------------------------------------------------------------------------|
| X1               | CREDIT_LIMIT          | Credit limit (NT dollar)                                                   |
| X2               | GENDER                | Gender (1 = male; 2 = female)                                              |
| X3               | EDUCATION_LEVEL       | Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others) |
| X4               | MARITAL_STATUS        | Marital status (1 = married; 2 = single; 3 = others)                       |
| X5               | AGE                   | Age (years)                                                                |
| X6               | SEPT_PAY_STATUS       | Repayment status in September  (-1 = pay duly; 1-9 = months delayed)   |
| X7               | AUG_PAY_STATUS        | Repayment status in August  (-1 = pay duly; 1-9 = months delayed)      |
| X8               | JULY_PAY_STATUS       | Repayment status in July  (-1 = pay duly; 1-9 = months delayed)        |
| X9               | JUNE_PAY_STATUS       | Repayment status in June  (-1 = pay duly; 1-9 = months delayed)        |
| X10              | MAY_PAY_STATUS        | Repayment status in May  (-1 = pay duly; 1-9 = months delayed)         |
| X11              | APRIL_PAY_STATUS      | Repayment status in April  (-1 = pay duly; 1-9 = months delayed)       |
| X12              | SEPT_BILL             | Amount of bill statement in September  (NT dollar)                     |
| X13              | AUG_BILL              | Amount of bill statement in August  (NT dollar)                        |
| X14              | JULY_BILL             | Amount of bill statement in July  (NT dollar)                          |
| X15              | JUNE_BILL             | Amount of bill statement in June  (NT dollar)                          |
| X16              | MAY_BILL              | Amount of bill statement in May  (NT dollar)                           |
| X17              | APRIL_BILL            | Amount of bill statement in April  (NT dollar)                         |
| X18              | SEPT_PAYMENT          | Amount paid in September  (NT dollar)                                  |
| X19              | AUG_PAYMENT           | Amount paid in August  (NT dollar)                                     |
| X20              | JULY_PAYMENT          | Amount paid in July  (NT dollar)                                       |
| X21              | JUNE_PAYMENT          | Amount paid in June  (NT dollar)                                       |
| X22              | MAY_PAYMENT           | Amount paid in May  (NT dollar)                                        |
| X23              | APRIL_PAYMENT         | Amount paid in April  (NT dollar)                                      |

## Cloning this repository:

If you would like to clone this repository and actively develop new features for this, here are the steps to clone this to your local machine and get started.

```bash
$ cd <some>/<dir>/
$ git clone https://github.com/amarchini5339/credit-card_defaults_prediction.git
$ cd ./credit-card_defaults_prediction
```

## Prerequisites: 

### Python Version
For this project we are using Python version 3.10.15, conda automatically will install and set the correct python version for the project so there is nothing that needs to be done.

### 1. Install Miniconda

If you are already using Anaconda or any other conda distribution, feel free to skip this step.

Miniconda is a minimal installer for `conda`, which we will use for managing environments and dependencies in this project. Follow these steps to install Miniconda or go [here](https://docs.anaconda.com/miniconda/install/) to reference the documentation: 

1. Open your terminal and run the following commands:
```bash
   $ mkdir -p ~/miniconda3

   <!-- If using Apple Silicon chip M1/M2/M3 -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
   <!-- If using intel chip -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh

   $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   $ rm ~/miniconda3/miniconda.sh
```

2. After installing and removing the installer, refresh your terminal by either closing and reopening or running the following command.
```bash
$ source ~/miniconda3/bin/activate
```

3. Initialize conda on all available shells.
```bash
$ conda init --all
```

You know conda is installed and working if you see (base) in your terminal. Next, we want to actually use the correct environments and packages.

### 2. Install Make

Make is a build automation tool that executes commands defined in a Makefile to streamline tasks like compiling code, setting up environments, and running scripts. [more information here](https://formulae.brew.sh/formula/make)

#### Installation

`make` is often pre-installed on Unix-based systems (macOS and Linux). To check if it's installed, open a terminal and type:
```bash
make -v
```

If it is not installed, simply use brew:
```bash
$ brew install make
```

#### Available Commands

The following commands are available in this project’s `Makefile`:

- **Set up the environment**:

    This will create the environment from the environment.yml file in the root directory of the project.

    ```bash
      $ make create
    ```

- **Update the environment**:

    This will update the environment from the environment.yml file in the root directory of the project. Useful if pulling in new changes that have updated the environment.yml file.

    ```bash
      $ make update
    ```

- **Remove the environment**:

    This will remove the environment from your shell. You will need to recreate and reinstall the environment with the setup command above.

    ```bash
    $ make clean
    ```

- **Activate the environment**:

    This will activate the environment in your shell. Keep in mind that make will not be able to actually activate the environment, this command will just tell you what conda command you need to run in order to start the environment.

    Please make sure to activate the environment before you start any development, we want to ensure that all packages that we use are the same for each of us.

    ```bash
    $ make activate
    ```

    Command you actually need to run in your terminal:
    ```bash
    $ conda activate credit_card_defaults_prediction
    ```

- **Deactivate the environment**:

    This will Deactivate the environment in your shell.

    ```bash
    $ make deactivate
    ```

- **run jupyter notebook**:

    This command will run jupyter notebook from within the conda environment. This is important so that we can make sure the package versions are the same for all of us! Please make sure that you have activated your environment before you run the notebook.

    ```bash
    $ make notebook
    ```

- **Export packages to env file**:

    This command will export any packages you install with either `conda install ` or `pip install` to the environment.yml file. This is important because if you add any packages we want to make sure that everyones machine knows to install it.

    ```bash
    $ make freeze
    ```

- **Verify conda environment**:

    This command will list all of your conda envs, the environment with the asterick next to it is the currently activated one. Ensure it is correct.

    ```bash
    $ make verify
    ```


#### Example workflows:

To simplify knowing which commands you need to run and when you can follow these instructions:

- **First time running, no env installed**:

    In the scenario where you just cloned this repo, or this is your first time using conda. These are the commands you will run to set up your environment.

    ```bash
    <!-- Make sure that conda is initialized -->
    $ conda init --all

    <!-- Next create the env from the env file in the root directory. -->
    $ make create

    <!-- After the environment was successfully created, activate the environment. -->
    $ conda activate credit_card_defaults_prediction

    <!-- verify the conda environment -->
    $ make verify

    <!-- verify the python version you are using. This should automatically be updated to the correct version 3.10.15 when you enter the environment. -->
    $ python --version

    <!-- Run jupyter notebook and have some fun! -->
    $ make notebook
    ```

- **Installing a new package**:

    While we are developing, we are going to need to install certain packages that we can utilize. Here is a sample workflow for installing packages. The first thing we do is verify the conda environment we are in to ensure that only the required packages get saved to the environment. We do not want to save all of the python packages that are saved onto our system to the `environment.yml` file. 

    Another thing to note is that if the package is not found in the conda distribution of packages you will get a `PackagesNotFoundError`. This is okay, just use pip instead of conda to install that specific package. Conda thankfully adds them to the environment properly.

    ```bash
    <!-- verify the conda environment -->
    $ make verify

    <!-- Install the package using conda -->
    $ conda install <package_name>

    <!-- If the package is not found in the conda channels, install the package with pip. -->
    $ pip install <package_name>

    <!-- If removing a package. -->
    $ conda remove <package_name>
    $ pip remove <package_name>

    <!-- Export the package names and versions that you downloaded to the environment.yml file -->
    make freeze
    ```

- **Daily commands to run before starting development**:

    Here is a sample workflow for the commands to run before starting development on any given day. We want to first pull all the changes from github into our local repository, 

    ```brew
    <!-- Pull changes from git -->
    $ git pull origin main

    <!-- Update env based off of the env file. It is best to deactivate the conda env before you do this step-->
    $ conda deactivate
    $ make update
    $ conda activate credit_card_defaults_prediction

    $ make notebook
    ```

- **Daily commands to run after finishing development**:

    Here is a sample workflow for the commands to run after finishing development for any given day.

    ```brew
    $ conda deactivate

    <!-- If you updated any of the existing packages, freeze to the environment.yml file. -->
    $ make freeze

    <!-- Commit changes to git -->
    $ git add .
    $ git commit -m "This is my commit message!"
    $ git push origin <branch_name>
    ```
## Directory Structure
```
credit-card_defaults_prediction/
    ├── notebooks/
        ├── models/
            ├── ann/
                ├── ann_report.joblib
                └── ann_roc_auc.joblib
            ├── naive_bayes/
                ├── naive_bayes_ensemble.joblib
                ├── test_features.joblib
                └── test_targets.joblib
            └── xgboost/
                ├── xbg_model.joblib
                ├── test_features.joblib
                └── test_targets.joblib
        ├── util/
            ├── get_data.py
            ├── model_eval.py
            └── shap.py
        ├── cc-default-prediction.ipynb
        ├── model-comparison.ipynb
        ├── naive-bayes.ipynb
        ├── neural-network.ipynb
        └── xgboost.ipynb
    ├── proposal/
        └── Final Project Proposal.pdf
    ├── .gitignore
    ├── README.md
    ├── Makefile
    ├── default_of_credit_card_clients.csv
    └── environment.yml
```

### notebooks
This directory holds all of the relevant information for the notebooks, such as utility functions, and exported model information using joblib. 

There are 5 main notebooks:
- One for basic EDA of our data
- Three for creating our models
- One for comparing the three models together

The models directory contains .joblib files for each of the models that we can use to easily export and compare against each other.

the util directory holds helper functions that we can use across all of the models, to simplify our code and ensure certain things are the same for each model.

### proposal
This directory holds our project proposal saved as a PDF. It explains our contributions and intial information regarding our reasoning behind this project.

### default_of_credit_card.csv
This is the raw datafile for our dataset. Imported into the repo to ensure smoother analysis and training of our models.

### environment.yml
This is our conda file that stores all of our dependencies, including python version and package versions.

### Makefile
This makefile defines some custom commands that we can use to help ensure our environments are the same across all of our machines.

### README.md
This has basic information regarding creating and setting up the project for replicability, as well as our reasoning behind the project yourself. You are reading this in the README.

## Contributors

<style>
table, td{
    border:0;
    text-align: center;
}
</style>

<table>
  <tr>
    <td>
      <a href="https://github.com/amarchini5339">
        <img src="https://github.com/amarchini5339.png" width="100" height="100" alt="Alex Marchini"/><br />
        <sub><b>Alex Marchini</b></sub>
      </a>
    </td>
    <td>
      <a href="https://github.com/AntonioRecaldeRusso">
        <img src="https://github.com/AntonioRecaldeRusso.png" width="100" height="100" alt="Antonio Recalde Russo"/><br />
        <sub><b>Antonio Recalde Russo</b></sub>
      </a>
    </td>
     <td>
      <a href="https://github.com/omarsagoo.png">
        <img src="https://github.com/omarsagoo.png" width="100" height="100" alt="Omar Sagoo"/><br />
        <sub><b>Omar Sagoo</b></sub>
      </a>
    </td>
    <td>
      <a href="https://github.com/DatasanAli">
        <img src="https://github.com/DatasanAli.png" width="100" height="100" alt="Hassan Ali"/><br />
        <sub><b>Hassan Ali</b></sub>
      </a>
    </td>
  </tr>
</table>