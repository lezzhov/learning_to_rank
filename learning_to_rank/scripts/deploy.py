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


def preprocess(data_path):
    df = pd.read_csv(f"{data_path}", delimiter=" ")

    df.columns = np.arange(len(df.columns))
    df.drop(
        columns=df.columns[df.isna().all()].tolist(), inplace=True
    )  # delete columns where all data is missing
    print(
        df.columns[df.isna().any()].tolist()
    )  # check if any remaining columns have missing data
    print(df.head)  # just taking a peak at the data... looks good so far

    split = {}

    df = df.iloc[:, 1:]

    # According to a LASSO regression analysis in
    # "Feature Selection and Model Comparisonon Microsoft Learning-to-Rank Data Sets",
    # (https://arxiv.org/pdf/1803.05127.pdf),
    # variance features, as well as Inverse Document Frequency (IDF) based features,
    # appear to be less useful (E.g. IDF based features seem not to be able to capture the web page quality well enough).
    # Therefore, I will train the model on the more relevant features instead.

    # fmt: off
    columns_to_remove = [41, 42, 43, 44, 45, 66, 67, 68, 69, 70,
                         91, 92, 93, 94, 95, 16, 17, 18, 19, 20,
                         71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

    # fmt: on

    # Get rid of irrelevant information at the beginning of each feature value
    df = df.applymap(lambda x: x.split(":", 1)[-1])
    # convert data into float format to conform to LGBMRanker input standard
    df = df.astype(float)
    # get rid of the query ID column since it is not a feature
    df = df.drop(columns=1)
    # rename column indices for convenience
    df.columns = [i for i in range(1, 137)]
    # drop less useful features
    df = df.drop(columns=columns_to_remove)

    return df


def main():
    # Evaluate model performance
    # Get the "ideal" order of y_test by sorting in descending order.

    args = parse.get_deploy_args()

    X_df = preprocess(args["data_path"])

    gbm = lgb.Booster(model_file=args["model_path"])

    test_pred = gbm.predict(X_df)


if __name__ == "__main__":
    main()
