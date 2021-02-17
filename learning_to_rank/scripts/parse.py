import argparse


def get_train_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--num_leaves", type=int, default=66)
    parser.add_argument("--learning_rate", type=float, default=0.2447211446488824)
    parser.add_argument("--reg_lambda", type=float, default=2.3763155117571912)
    args = parser.parse_args()
    return args


def get_tune_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    return vars(args)


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    return args


def get_deploy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    return args
