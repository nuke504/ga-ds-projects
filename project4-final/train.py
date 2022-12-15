import numpy as np
import pandas as pd
import argparse

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

from typing import List, Tuple

from helper.features import generate_features
from helper.preprocess import merge_data, process_data, read_dataframes, add_beer_category
from helper.visual import plot_confusion_matrix, plot_feature_importance

BEER_DS_URI = "Path to CSV"
BREWERY_DS_URI = "Path to CSV"
STATE_DS_URI = "Path to CSV"

# def compute_metrics(
#     y:np.array,
#     y_pred:np.array, 
#     print_metrics:bool = True,
#     metric_prefix:str="",
#     logger = None
# ):
#     """
#     Print RMSE and R2
#     """
#     # Compute Metrics
#     rmse = mean_squared_error(y, y_pred, squared = False)
#     r2 = r2_score(y, y_pred)
    
#     if print_metrics:
#         print(f"RMSE: {rmse:.3f}")
#         print(f"R2: {r2:.3f}")

#     if logger:

#         logger.log_metric(f"{metric_prefix}RMSE", rmse)
#         logger.log_metric(f"{metric_prefix}R2", r2)
    
#     return r2, rmse

def compute_metrics(
    y:np.array,
    y_pred:np.array, 
    cm_labels:List[str] = ["light","regular","strong"],
    print_cm:bool = True
) -> Tuple[np.array, dict]:
    """
    Compute Metrics

    - confusion matrix
    - precision of categories
    - recall of categories
    """
    
    cm = confusion_matrix(y, y_pred, labels=cm_labels)
    df_cm = pd.DataFrame(cm, index = cm_labels, columns = cm_labels)

    if print_cm:
        print("Confusion Matrix")
        print(df_cm)

    # Compute recall and precision
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
    }

    for i in range(cm.shape[0]):
        metrics[f"recall-{cm_labels[i]}"] = float(cm[i,i]/cm[i].sum())
        metrics[f"precision-{cm_labels[i]}"] = float(cm[i,i]/cm[:,i].sum())
    
    return df_cm, metrics

def compute_logloss(
    y:np.array, 
    y_prob:np.array,
    sample_weight: np.array,
    labels: np.array,
    gamma: float = 2,
    epsilon: float = 1e-7,
):
    """
    Compute the log loss
    """

    if sample_weight is None:
        weight_arr = np.ones(shape=y.shape)
    else:
        weight_arr = sample_weight.ravel()

    lb = LabelEncoder()
    labels_new = lb.fit_transform(labels)
    y_truth = lb.transform(y.ravel())

    # Add epsilon to prevent numerical errors
    predicted_ep = y_prob + epsilon
    log_prob = -1 * np.log(predicted_ep)
    cross_entropy = np.array(
        [log_prob[row_idx, col_idx] for row_idx, col_idx in enumerate(y_truth)]
    )
    prob_power = np.power((1 - predicted_ep), gamma)
    loss_weight = np.array(
        [
            prob_power[row_idx, col_idx]
            for row_idx, col_idx in enumerate(y_truth)
        ]
    )

    loss_arr = weight_arr * cross_entropy * loss_weight

    return np.sum(loss_arr) / weight_arr.sum()



def log_numpy_array(
    arr:np.array,
    arr_name:str
):
    """
    Log the numpy array
    """
    with open(f"{arr_name}.npy", "wb") as outfile:
        np.save(outfile, arr)

    mlflow.log_artifact(f"{arr_name}.npy")

def main():
    """
    Main function
    """

    parser = argparse.ArgumentParser()

    # Arguments for Features
    parser.add_argument('--svd_n', type=int, default=3,
                        help="Vector dimension for SVD")

    # Arguments for Light GBM
    parser.add_argument('--n_estimators', type=int, default=100,
                        help="Number of Estimators")
    parser.add_argument('--min_child_samples', type=int, default=20,
                        help="Min number of samples in child leaf")
    parser.add_argument('--num_leaves', type=int, default=31,
                        help="Number of leaves for base learner")
    parser.add_argument('--max_depth', type=int, default=-1,
                        help="Max tree depth")
    parser.add_argument('--reg_alpha', type=float, default=0.,
                        help="L1 Weights")
    parser.add_argument('--reg_lambda', type=float, default=0.,
                        help="L2 Weights")
    parser.add_argument('--boosting_type', type=str, default="gbdt",
                        help="Boosting Mechanism")
 
    # General Arguments
    parser.add_argument('--cross_validation', type=int, default=5,
                        help="Number of CV Folds")
    parser.add_argument('--test_split', type=float, default=0.15,
                        help="Test State")
    parser.add_argument('--random_state', type=int, default=101,
                        help="Random state used")

    args = parser.parse_args()

    mlflow.start_run()
    
    # enable autologging
    mlflow.sklearn.autolog()

    # Read all datasets and preprocess them
    data_dict = read_dataframes(BEER_DS_URI, BREWERY_DS_URI, STATE_DS_URI)

    df_bb = merge_data(data_dict["beer"], data_dict["brewery"], data_dict["state"])

    df_bb = process_data(df_bb)
    
    df_bb = add_beer_category(df_bb)

    X, y, column_names, encoder_dict  = generate_features(df_bb, svd_elements=args.svd_n)

    feature_args = {
        "svd_dimension": args.svd_n
    }

    # Get those indices which are NA for y and set them aside
    print(f"Total rows: {X.shape[0]}")

    na_indices = pd.Series(y).isna()

    y = y[~na_indices]

    X_exp = X[~na_indices]

    print(f"Non-NA rows: {X_exp.shape[0]}")

    # Keep this for later
    X_na = X[na_indices]

    # Do train-test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_exp, y, 
        test_size=args.test_split,
        random_state=args.random_state
    )

    scaler = StandardScaler()

    model_kwargs = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "boosting_type": args.boosting_type,
        "min_child_samples": args.min_child_samples,
        "num_leaves ": args.num_leaves 
    }
    
    # Scale X
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    kf = KFold(n_splits=args.cross_validation, shuffle=True)

    # Apply K-Fold Cross Validation
    cv_models = {}
    cv_train_folds = {}
    cv_train_metrics = {}
    cv_val_metrics = {}

    for i, (train_index, test_index) in enumerate(kf.split(X_train_val)):
        print(f"==== Fold {i} ====")
        X_train, y_train = X_train_val[train_index], y_train_val[train_index]
        X_val, y_val = X_train_val[test_index], y_train_val[test_index]

        weights = dict(**(1/(3*pd.Series(y_train).value_counts())))
        weights_arr = pd.Series(y_train).apply(lambda x: weights[x]).to_numpy()

        model = LGBMClassifier(random_state=args.random_state, **model_kwargs)

        model.fit(
            X = X_train,
            y = y_train,
            sample_weight = weights_arr,
            eval_set = [(X_train, y_train),(X_val, y_val),],
            eval_names = [f"Train fold {i}", f"Val fold {i}"],
            feature_name = column_names
        )

        train_loss = model.best_score_[f"Train fold {i}"]["multi_logloss"]
        val_loss = model.best_score_[f"Val fold {i}"]["multi_logloss"]

        y_train_pred = model.predict(X = X_train)
        y_val_pred = model.predict(X = X_val)

        _, train_metrics = compute_metrics(y_train, y_train_pred)
        _, val_metrics = compute_metrics(y_val, y_val_pred)

        y_val_prob = model.predict_proba(X = X_val)
        weights_val_arr = pd.Series(y_val).apply(lambda x: weights[x]).to_numpy()
        val_focal_loss = compute_logloss(y_val, y_val_prob, weights_val_arr, ["light","regular","strong"])

        cv_models[i] = model
        cv_train_metrics[i] = {
            "loss": float(train_loss),
            **train_metrics
        }

        cv_val_metrics[i] = {
            "loss": float(val_loss),
            "focal_loss": float(val_focal_loss),
            **val_metrics
        }

        cv_train_folds[i] = X_train

        print("")

    mlflow.log_dict(cv_train_metrics, "cv_train_metrics.yaml")
    mlflow.log_dict(cv_val_metrics, "cv_eval_metrics.yaml")

    # cv_results = cross_validate(
    #     model,
    #     X_train_val,
    #     y_train_val,
    #     cv=kf,
    #     scoring=["r2","neg_mean_squared_error"],
    #     return_train_score=True,
    #     return_estimator=True
    # )

    # metrics = {
    #     "Train RMSE": (np.sqrt(-1*cv_results["train_neg_mean_squared_error"])).mean(),
    #     "Train RMSE Std": (np.sqrt(-1*cv_results["train_neg_mean_squared_error"])).std(),
    #     "Train R2": cv_results["train_r2"].mean(),
    #     "Validation RMSE": (np.sqrt(-1*cv_results["test_neg_mean_squared_error"])).mean(),
    #     "Validation RMSE Std": (np.sqrt(-1*cv_results["test_neg_mean_squared_error"])).std(),
    #     "Validation R2": cv_results["test_r2"].mean()
    # }
    
    metrics = {
        "Train Loss": np.mean([metric["loss"] for metric in cv_train_metrics.values()]),
        "Train Accuracy": np.mean([metric["accuracy"] for metric in cv_train_metrics.values()]),
        "Val Loss": np.mean([metric["loss"] for metric in cv_val_metrics.values()]),
        "Val Focal Loss": np.mean([metric["focal_loss"] for metric in cv_val_metrics.values()]),
        "Val Accuracy": np.mean([metric["accuracy"] for metric in cv_val_metrics.values()])
    }

    # Log Metrics and Params
    mlflow.log_metrics(metrics)

    mlflow.log_params(model_kwargs)
    mlflow.log_params(feature_args)

    # Get Predicted Y for Test
    argmin_idx = np.argmin([metric["loss"] for metric in cv_val_metrics.values()])

    trained_model = cv_models[argmin_idx]
    y_test_pred = trained_model.predict(X_test)

    # fig = plot_error_terms(y_test, y_test_pred, "Test Error Plot")
    # mlflow.log_figure(fig, "test_error.png")

    confusion_matrix_labels = ["light","regular","strong"]
    cf_matrix = confusion_matrix(y_test, y_test_pred, labels=confusion_matrix_labels)
    fig = plot_confusion_matrix(cf_matrix, confusion_matrix_labels)
    mlflow.log_figure(fig, "confusion_matrix.png")

    fig = plot_feature_importance(trained_model.feature_importances_, np.array(column_names))
    mlflow.log_figure(fig, "feature_importance.png")

    print("===== Test Metrics =====")
    _, test_metrics = compute_metrics(y_test, y_test_pred)
    for metric, val in test_metrics.items():
        print(f"Test {metric}",val)
        mlflow.log_metric(f"Test {metric}",val)
    # compute_metrics(y_test, y_test_pred, True, "Test ", mlflow)
    
    # Save Model to workspace
    print("Registering the model via MLFlow")
    
    registered_model_name = "craft-beer-classification-lgbm"
    mlflow.lightgbm.log_model(
        lgb_model=trained_model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    log_numpy_array(cv_train_folds[argmin_idx], "X_train_val")
    log_numpy_array(X_test, "X_test")
    log_numpy_array(y_test, "y_test")

    # End the run
    mlflow.end_run()


if __name__ == "__main__":
    main()
