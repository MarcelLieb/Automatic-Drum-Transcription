import logging
import sys
from dataclasses import asdict

import numpy as np
import optuna
from calflops import calculate_flops
from torch import nn

from model.CRNN import CRNN
from model.cnnA import CNNAttention
from settings import DatasetSettings, TrainingSettings, CNNMambaSettings, EvaluationSettings, AudioProcessingSettings, \
    CRNNSettings, AnnotationSettings, Config, CNNAttentionSettings
from main import main as train_model
import torch


def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 4e-5, 6e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 5e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 1)
    beta_2 = trial.suggest_float("beta_2", 0.5, 1)
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    # decoupled_weight_decay = trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 4, 64, log=True)
    frame_length = trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_overlap = trial.suggest_float("segment_overlap", 0.1, min(frame_length, 2), step=0.1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    # num_channels = trial.suggest_int("num_channels", 16, 64)
    num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 48])
    # num_attention_layers = trial.suggest_int("num_attention_layers", 1, 8)
    num_attention_layers = 5
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16])
    use_relative_pos = trial.suggest_categorical("use_relative_pos", [True, False])
    # down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    # max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    # num_conv_layers = trial.suggest_int("num_conv_layer", 1, max_layers)
    # if num_conv_layers == 1:
    #     channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    # else:
    #     channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    # classifier_dim = trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 11, log=True)
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
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        # decoupled_weight_decay=decoupled_weight_decay,
        dataset_version="S",
        early_stopping=7,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels),
        annotation_settings=AnnotationSettings(time_shift=0.015),
    )
    model_settings = CNNAttentionSettings(
        num_channels=num_channels,
        dropout=dropout,
        num_attention_blocks=num_attention_layers,
        num_heads=n_heads,
        activation=activation,
        flux=flux,
        use_relative_pos=use_relative_pos,
    )
    evaluate_settings = EvaluationSettings(pr_points=200, min_test_score=0.48)

    test_model = CNNAttention(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes, n_mels=n_mels)
    input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
                                          print_results=False)
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("macs", macs)
    trial.set_user_attr("params", params)

    del test_model

    # train model
    model, score = train_model(
        Config(training=train_settings, dataset=dataset_settings, model=model_settings, evaluation=evaluate_settings),
        trial=trial)
    del model

    torch.cuda.empty_cache()

    return score, flops


def main():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "CNN Attention subset"
    storage_name = "sqlite:///hyperparameters.db"
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.enqueue_trial({
        "activation": "silu",
        "batch_size": 57,
        "beta_1": 0.535394233881171,
        "beta_2": 0.530662776036198,
        "dropout": 0.5,
        "lr": 0.00018497419766305348,
        "segment_length": 9,
        "segment_overlap": 0.7,
        "weight_decay": 1.9160206379176343e-7,
        "n_mels": 96,
        "num_channels": 24,
        "flux": False,
        "n_heads": 4,
    }, skip_if_exists=True)

    study.optimize(objective, n_trials=200, catch=(torch.cuda.OutOfMemoryError, RuntimeError), gc_after_trial=True)


if __name__ == '__main__':
    main()
