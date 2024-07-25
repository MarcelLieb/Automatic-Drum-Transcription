import logging
import sys

import optuna
from settings import DatasetSettings, TrainingSettings, CNNMambaSettings, EvaluationSettings
from main import main as train_model
import torch

def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 32)
    frame_length = trial.suggest_float("segment_length", 1, 10)

    train_settings = TrainingSettings(
        epochs=20,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        dataset_version="M",
        early_stopping=7,
    )
    dataset_settings = DatasetSettings(frame_length=frame_length)
    model_settings = CNNMambaSettings(causal=True, num_channels=16)
    evaluate_settings = EvaluationSettings(pr_points=200)

    # train model
    model, score = train_model(train_settings, dataset_settings, evaluate_settings, model_settings)
    del model

    torch.cuda.empty_cache()

    return score


def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cnn_mamba_train"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)

if __name__ == '__main__':
    main()