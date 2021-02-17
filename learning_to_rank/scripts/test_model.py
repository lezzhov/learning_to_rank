from parse import get_tune_args
import pandas as pd
from zipfile import ZipFile
import urllib.request
import numpy as np
from sklearn.metrics import ndcg_score
import optuna.integration.lightgbm as lgb
import optuna
from sklearn import preprocessing
from preprocess import get_data
import parse


def main():
    # Evaluate model performance
    # Get the "ideal" order of y_test by sorting in descending order.

    args = parse.get_test_args()

    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(args["data_path"])

    gbm = lgb.Booster(model_file=args["model_path"])

    true_relevance = y_test.sort_values(ascending=False)

    # Get the actual order of y_test by sorting it according to our model's predictions.

    test_pred = gbm.predict(X_test)
    y_test = pd.DataFrame({"relevance_score": y_test, "predicted_ranking": test_pred})

    relevance_score = y_test.sort_values("predicted_ranking", ascending=False)

    # Use computed variables to calculate the nDCG score
    print(
        "nDCG score: ",
        ndcg_score(
            [true_relevance.to_numpy()], [relevance_score["relevance_score"].to_numpy()]
        ),
    )


if __name__ == "__main__":
    main()
