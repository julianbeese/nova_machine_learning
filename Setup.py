import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocessing(data, target, multicoll_columns, categoric_columns):
    '''
    Preprocessing function

    Required import:
    import pandas as pd

    :param data: Input Dataframe
    :param target: Target variable
    :param multicoll_columns: columns which show mulitcollinearity to another column
    :param categoric_columns: columns with categorical data
    :return: df with preprocessed features (X) and target variable y
    '''
    #drop columns showing multicollinearity
    data.drop(multicoll_columns, axis=1, inplace=True)

    # One hot enconding for categorical features
    data = pd.get_dummies(data, columns=categoric_columns)

    # Splitting features and target variable
    X = data.drop(target, axis=1)
    y = data[target]

    return X, y

def tune_and_evaluate_models(X_train, y_train, X_test, y_test, models, param_grids, cv, dataset_name):
    '''
    Tune and evaluate models using GridSearchCV.

    Required import:
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.metrics import accuracy_score,recall_score,precision_score
    from sklearn.metrics import f1_score,roc_auc_score

    Parameters:
    :param X_train: Training data features.
    :param y_train: training data target variable.
    :param X_test: test data features.
    :param y_test: test data target variable.
    :param models: dictionary of models to tune and evaluate.
    :param param_grids: dictionary of parameter grids to tune.
    :param cv: cross-validation strategy.
    :param dataset_name: string to identify dataset.
    :return: Dictionary of results for each model.
    '''

    # Dictionary to store results
    results = {}

    # Metrics to calculate

    metrics = {
        'Mean Absolute Error': mean_absolute_error,
        'Mean Squared Error': mean_squared_error,
        'R2 Score': r2_score
    }

    # Tune and evaluate each model
    for name, model in models.items():
        print(f"\n--- Tuning {name} on {dataset_name} Dataset ---")

        # Perform Grid Search
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='neg_mean_squared_error', # metrics,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)

        # get best model
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Predictions
        y_pred = best_model.predict(X_test)

        # Calculate Metrics
        model_metrics = {}
        for metric_name, metric_func in metrics.items():
            if metric_name in ['Mean Squared Error']:
                model_metrics[metric_name] = metric_func(y_test, y_pred)
            else:
                model_metrics[metric_name] = metric_func(y_test, y_pred)

        # Store results
        results[name] = {
            'Best Model': best_model,
            'Best Params': best_params,
            'Metrics': model_metrics
        }

    return results


def print_results(results):
    """
    Prints the structured output of model evaluation results.

    This function iterates through a dictionary of model results, printing key evaluation
    metrics for each model, including the best model name, the best hyperparameters,
    evaluation metrics, confusion matrix, and classification report.

    :param results:
        Dictionary where keys are model names (str) and values are dictionaries
        containing model evaluation details such as
        'Best Model', 'Best Params', 'Metrics', 'Confusion Matrix',
        and 'Classification Report'.
    """
    for model_name, model_results in results.items():
        print(f"--- Results for {model_name} ---")
        print(f"Best Model: {model_results['Best Model']}")
        print(f"Best Params: {model_results['Best Params']}")
        print("Metrics: ")
        for metric, value in model_results['Metrics'].items():
            print(f"{metric}: {value}")
        print(f"Confusion Matrix:\n{model_results['Confusion Matrix']}")
        print(f"Classification Report:\n{model_results['Classification Report']}")


def r_forest(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates a Random Forest model using the provided training and
    testing datasets. The function tunes hyperparameters of the Random Forest
    model using cross-validation and outputs the evaluation results.

    :param x_train: Training data features.
    :type x_train: array-like
    :param y_train: Training data labels.
    :type y_train: array-like
    :param x_test: Testing data features.
    :type x_test: array-like
    :param y_test: Testing data labels.
    :type y_test: array-like
    :return: None
    """

    param_grids = {
        'Random Forest': {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }

    model = {'Random Forest': RandomForestRegressor(random_state=42)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = tune_and_evaluate_models(x_train, y_train, x_test, y_test, model, param_grids, cv, 'Cars Dataset')
    print_results(results)


def ridge(x_train, y_train, x_test, y_test):
    """
    Performs Ridge Regression on the given training and test dataset, applies hyperparameter tuning, evaluates the model,
    and prints results. The function utilizes cross-validation for model tuning using predefined hyperparameter grids.

    :param x_train: Training feature set
    :param y_train: Training target variable
    :param x_test: Test feature set
    :param y_test: Test target variable
    :return: None
    """
    param_grids = {'Ridge': {'alpha': [0.01, 0.1, 1, 10]}    }
    model = {'Ridge': Ridge(random_state=42)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = tune_and_evaluate_models(x_train, y_train, x_test, y_test, model, param_grids, cv, 'Cars Dataset')
    print_results(results)

def lasso(x_train, y_train, x_test, y_test):
    """
    Performs Lasso regression using cross-validation to find the best hyperparameters and evaluates the model on
    the provided data. The function internally uses a grid search for hyperparameter tuning, leveraging a particular
    dataset and returns evaluation metrics for the model.

    :param x_train: Training data features.
    :param y_train: Training data target labels.
    :param x_test: Testing data features.
    :param y_test: Testing data target labels.
    :return: None. Prints the results of the model tuning and evaluation process.
    """
    param_grids = {'Lasso': {'alpha': [0.01, 0.1, 1, 10]}    }
    model = {'Lasso': Lasso(random_state=42)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = tune_and_evaluate_models(x_train, y_train, x_test, y_test, model, param_grids, cv, 'Cars Dataset')
    print_results(results)

def elastic_net(x_train, y_train, x_test, y_test):
    """
    Elastic Net Regression Model Training and Evaluation

    This function performs training and evaluation of an Elastic Net regression model on the
    provided dataset using specified hyperparameters, cross-validation strategy, and evaluation
    pipeline. It uses grid search to tune the hyperparameters and computes results for the
    Elastic Net regression task. The function also outputs key evaluation metrics for later analysis.

    :param x_train: Training feature set
    :param y_train: Target labels corresponding to the training feature set
    :param x_test: Testing feature set
    :param y_test: Target labels corresponding to the testing feature set
    :return: None; results from the evaluation process will be printed in the console
    """
    param_grids = {'Elastic Net': {'alpha': [0.01, 0.1, 1, 10]}    }
    model = {'Elastic Net': Ridge(random_state=42)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = tune_and_evaluate_models(x_train, y_train, x_test, y_test, model, param_grids, cv, 'Cars Dataset')
    print_results(results)

def xgboost(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates an XGBoost classifier model using the provided training
    and testing data. The function tunes hyperparameters for the XGBoost algorithm
    and performs cross-validation to optimize and evaluate the model's performance.
    It prints the results of the model evaluation.

    :param x_train: Training feature matrix for model training.
    :param y_train: Training target labels for model training.
    :param x_test: Test feature matrix for model evaluation.
    :param y_test: Test target labels for model evaluation.
    :return: Results of the evaluated model, including performance metrics.
    """
    param_grids = {
        'XGBoost': {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.5]
        }
    }
    model = {'XGBoost': xgb.XGBRegressor(random_state=42)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = tune_and_evaluate_models(x_train, y_train, x_test, y_test, model, param_grids, cv, 'Cars Dataset')
    print_results(results)
