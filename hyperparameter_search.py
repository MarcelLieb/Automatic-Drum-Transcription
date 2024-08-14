import logging
import sys
from dataclasses import asdict

import optuna
from calflops import calculate_flops
from torch import nn

from model.cnn import CNN
from settings import DatasetSettings, TrainingSettings, CNNMambaSettings, EvaluationSettings, AudioProcessingSettings, \
    CNNSettings, AnnotationSettings
from main import main as train_model
import torch


def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 4e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    frame_length = trial.suggest_float("segment_length", 0.15, 0.4)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    num_channels = trial.suggest_int("num_channels", 16, 64)
    num_feature_layer = trial.suggest_int("num_feature_layer", 1, 3)
    if num_feature_layer == 1:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    else:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    classifier_dim = trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 11, log=True)
    activation = trial.suggest_categorical("activation", ["selu", "relu", "elu", "silu"])

    match activation:
        case "selu":
            activation = nn.SELU()
        case "relu":
            activation = nn.ReLU()
        case "elu":
            activation = nn.ELU()
        case "silu":
            activation = nn.SiLU()

    train_settings = TrainingSettings(
        epochs=20,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        dataset_version="M",
        early_stopping=7,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        audio_settings=AudioProcessingSettings(n_mels=n_mels),
        annotation_settings=AnnotationSettings(time_shift=0.015),
    )
    model_settings = CNNSettings(
        num_channels=num_channels,
        num_residual_blocks=0,
        num_feature_layers=num_feature_layer,
        channel_multiplication=channel_multiplication,
        dropout=dropout,
        causal=True,
        flux=flux,
        classifier_dim=classifier_dim,
        down_sample_factor=down_sample_factor,
        activation=activation,
    )
    evaluate_settings = EvaluationSettings(pr_points=400)

    test_model = CNN(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes, n_mels=n_mels)
    input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
                                          print_results=False)
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("macs", macs)
    trial.set_user_attr("params", params)

    del test_model

    # train model
    model, score = train_model(train_settings, dataset_settings, evaluate_settings, model_settings)
    del model

    torch.cuda.empty_cache()

    return score, flops


def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cnn_feature_model_with_flops"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.enqueue_trial({
        "lr": 6e-5,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "segment_length": 0.25,
        "n_mels": 84,
        "flux": False,
        "num_channels": 32,
        "num_feature_layer": 2,
        "channel_multiplication": 2,
        "dropout": 0.1,
        "down_sample_factor": 2,
        "classifier_dim": 420,
        "activation": "selu"
    }, skip_if_exists=True)

    study.enqueue_trial({
        "lr": 2e-4,
        "weight_decay": 1e-3,
        "batch_size": 256,
        "segment_length": 0.32,
        "n_mels": 128,
        "flux": False,
        "num_channels": 25,
        "num_feature_layer": 3,
        "channel_multiplication": 3,
        "dropout": 0.34,
        "down_sample_factor": 3,
        "classifier_dim": 520,
        "activation": "selu"
    }, skip_if_exists=True)

    study.enqueue_trial({
        "lr": 8e-4,
        "weight_decay": 4e-5,
        "batch_size": 1024,
        "segment_length": 0.17,
        "n_mels": 128,
        "flux": True,
        "num_channels": 64,
        "num_feature_layer": 1,
        "channel_multiplication": 1,
        "dropout": 0.28,
        "down_sample_factor": 1,
        "classifier_dim": 320,
        "activation": "selu"
    }, skip_if_exists=True)

    study.optimize(objective, n_trials=5, catch=(torch.cuda.OutOfMemoryError,), gc_after_trial=True)


if __name__ == '__main__':
    main()
