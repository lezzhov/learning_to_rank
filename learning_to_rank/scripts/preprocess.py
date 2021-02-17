from parse import get_train_args
import pandas as pd
import numpy as np


def get_data(data_path):
    dfs = {
        "train": pd.read_csv(f"{data_path}/train.txt", delimiter=" "),
        "vali": pd.read_csv(f"{data_path}/vali.txt", delimiter=" "),
        "test": pd.read_csv(f"{data_path}vali.txt", delimiter=" "),
    }

    for df in dfs.values():
        df.columns = np.arange(len(df.columns))
        df.drop(
            columns=df.columns[df.isna().all()].tolist(), inplace=True
        )  # delete columns where all data is missing

    split = {}

    split["X_train"] = dfs["train"].iloc[:, 1:]
    split["X_val"] = dfs["vali"].iloc[:, 1:]
    split["X_test"] = dfs["test"].iloc[:, 1:]

    y_train = dfs["train"].iloc[:, 0]
    y_val = dfs["vali"].iloc[:, 0]
    y_test = dfs["test"].iloc[:, 0]

    # In order to use the Light GBM framework, we need to create variables group_train and group_vali, which contain
    # number of examples for each query ID. This will allow LGBMRanker to group examples by query during training.

    g = split["X_train"].groupby(by=1)
    size = g.size()
    group_train = size.to_list()

    g = split["X_val"].groupby(by=1)
    size = g.size()
    group_vali = size.to_list()

    # According to a LASSO regression analysis in "Feature Selection and Model Comparisonon Microsoft
    # Learning-to-Rank Data Sets", (https://arxiv.org/pdf/1803.05127.pdf), variance features, as well as Inverse
    # Document Frequency (IDF) based features, appear to be less useful (E.g. IDF based features seem not to be able
    # to capture the web page quality well enough). Therefore, I will train the model on the more relevant features
    # instead.

    # fmt: off
    columns_to_remove = [41, 42, 43, 44, 45, 66, 67, 68, 69, 70,
                         91, 92, 93, 94, 95, 16, 17, 18, 19, 20,
                         71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    # fmt: on

    for name, df in split.items():
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

        split[name] = df

    return (
        split["X_train"],
        split["X_test"],
        split["X_val"],
        y_train,
        y_test,
        y_val,
        group_vali,
        group_train,
    )
