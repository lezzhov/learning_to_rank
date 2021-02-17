from parse import get_tune_args
import pandas as pd
import optuna.integration.lightgbm as lgb
import optuna
from preprocess import get_data
from parse import get_tune_args


def main():
    args = get_tune_args()
    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(
        args["data_path"]
    )

    # To train the model, I will be using LightGBM due to its simplicity and support of learning to rank tasks.

    # Before we train our model, let's find the best performing hyperparameters.
    # We will use Optuna, which supports integration with LightGBM.

    def objective(trial):
        # num_leaves is the first hyperparameter that LightGBM manual suggests to optimize.
        # It is the main parameter to control the complexity of the tree model.
        # Learning rate and L2 regularization parameter lambda are the standard and most important hyperparameters for tuning.

        param = {
            "num_leaves": trial.suggest_int("num_leaves", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.3),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }

        gbm = lgb.LGBMRanker(
            num_leaves=param["num_leaves"],
            learning_rate=param["learning_rate"],
            reg_lambda=param["reg_lambda"],
        )

        gbm.fit(
            X_train,
            y_train,
            group=group_train,
            eval_group=[group_vali],
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
        )

        return gbm.best_score_["valid_0"]["ndcg@1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params

    print("Number of finished trials:", len(study.trials))
    print("Best trial:", best_params)


if __name__ == "__main__":
    main()
