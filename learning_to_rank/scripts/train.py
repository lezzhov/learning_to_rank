from parse import get_train_args
import pandas as pd
import optuna.integration.lightgbm as lgb
import optuna
from preprocess import get_data


def main():
    args = get_train_args()

    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(
        args["data_path"]
    )

    # Now that we found the best hyperparameters, let's use them to train our model for a longer time.
    gbm = lgb.LGBMRanker(
        n_estimators=10000,
        num_leaves=args["num_leaves"],
        learning_rate=args["learning_rate"],
        reg_lambda=args["reg_lambda"],
    )

    gbm.fit(
        X_train,
        y_train,
        group=group_train,
        eval_group=[group_vali],
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,
    )

    gbm.booster_.save_model(args["output_file_name"], num_iteration=gbm.best_iteration)


if __name__ == "__main__":
    main()
