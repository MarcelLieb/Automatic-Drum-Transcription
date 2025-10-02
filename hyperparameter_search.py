import logging
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
import optuna
from optuna import (
    storages,
    load_study,
    Study,
    create_study,
    Trial,
    terminator,
    samplers,
    logging as optuna_logging,
)
import optuna.pruners as pruners
import optunahub
from optuna.samplers import BruteForceSampler, TPESampler
from optuna.terminator import (
    TerminatorCallback,
    report_cross_validation_scores,
    Terminator,
    EMMREvaluator,
    CrossValidationErrorEvaluator,
    RegretBoundEvaluator,
)
import torch
from calflops import calculate_flops
from optuna.trial import TrialState
from torch import nn

from dataset import DrumMapping
from hyperparameters import final_experiment_params
from main import main as train_model
from model.CRNN import CRNN
from model.cnn import CNN
from model.cnnA import CNNAttention
from model.cnnM2 import CNNMambaFast
from settings import DatasetSettings, TrainingSettings, CNNMambaSettings, EvaluationSettings, AudioProcessingSettings, \
    CRNNSettings, AnnotationSettings, Config, CNNAttentionSettings, CNNSettings

# Varying batch size and segment length can lead to different results, so we set a fixed cache size limit
torch._dynamo.config.cache_size_limit = 1024


def attention_objective(trial: Trial):
    lr = trial.suggest_float("lr", 5e-6, 6e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-15, 5e-3, log=True)
    beta_1 = 0.9
    beta_2 = 0.999
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = True
    batch_size = trial.suggest_int("batch_size", 4, 64)
    frame_length = 4.0 # round(trial.suggest_float("segment_length", 1.0, 8.0, step=0.1), 1)
    frame_overlap = 0.5 # round(trial.suggest_float("segment_overlap", 0.1, min(frame_length / 2, 2), step=0.1), 1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = False # trial.suggest_categorical("flux", [True, False])
    # num_channels = trial.suggest_int("num_channels", 16, 64)
    # num_attention_layers = trial.suggest_int("num_attention_layers", 1, 5)
    num_attention_layers = 5
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16])
    # expansion_factor = trial.suggest_int("expansion_factor", 1, 4)
    expansion_factor = trial.suggest_categorical("expansion_factor", [2, 4])
    use_relative_pos = False
    down_sample_factor = trial.suggest_categorical("down_sample_factor", [3, 4]) # trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = 2 # trial.suggest_int("num_conv_layer", 0, max_layers)
    # if num_conv_layers > 0:
    #     if num_conv_layers == 1:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    #     else:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    #     num_channels = trial.suggest_categorical("num_channels", [16, 32])
    # else:
    #     num_channels = 1
    #     channel_multiplication = 1
    channel_multiplication = 2
    num_channels = 32 # trial.suggest_categorical("num_channels", [16, 24, 32])
    # hidden_units = trial.suggest_int("hidden_units", 2 ** 4, 2 ** 11, log=True)
    cnn_dropout = round(trial.suggest_float("cnn_dropout", 0.0, 0.9, step=0.05), 2)
    dense_dropout = round(trial.suggest_float("dense_dropout", 0.0, 0.9, step=0.05), 2)
    attention_dropout = round(trial.suggest_float("attention_dropout", 0.0, 0.9, step=0.05), 2)
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
        epochs=50,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=5,
        ema=False,
        min_save_score=0.80,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02, mapping=DrumMapping.THREE_CLASS),
        dataset_version="S",
        full_length_test=False,
        train_set="a2md_train",
        splits=[0.8, 0.2, 0.0]
    )
    model_settings = CNNAttentionSettings(
        num_channels=num_channels,
        cnn_dropout=cnn_dropout,
        dense_dropout=dense_dropout,
        attention_dropout=attention_dropout,
        activation=activation,
        flux=flux,
        num_attention_blocks=num_attention_layers,
        num_heads=n_heads,
        context_size=200,
        expansion_factor=expansion_factor,
        use_relative_pos=use_relative_pos,
        down_sample_factor=down_sample_factor,
        num_conv_layers=num_conv_layers,
        causal=True,
        channel_multiplication=channel_multiplication,
    )
    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.65, detect_tolerance=0.05)

    # test_model = CNNAttention(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
    #                           n_mels=n_mels)
    # input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    # with torch.inference_mode():
    #     test_model.eval()
    #     flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
    #                                           print_results=False)
    # trial.set_user_attr("flops", flops)
    # trial.set_user_attr("macs", macs)
    # trial.set_user_attr("params", params)
    # del test_model

    # train model
    scores = []
    for i in range(1, 4):
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=model_settings, evaluation=evaluate_settings),
            trial=trial if i == 1 else None, metric_to_track="F-Score/Sum/Validation", seed=i * 20
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)

    torch.cuda.empty_cache()

    return score


def cnn_objective(trial: Trial):
    lr = trial.suggest_float("lr", 5e-6, 6e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-30, 5e-3, log=True)
    beta_1 = 0.9
    beta_2 = 0.999
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = True
    batch_size = trial.suggest_int("batch_size", 4, 64)
    frame_length = 4.0 # round(trial.suggest_float("segment_length", 1.0, 8.0, step=0.1), 1)
    frame_overlap = 0.5 # round(trial.suggest_float("segment_overlap", 0.1, min(frame_length / 2, 2), step=0.1), 1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = False # trial.suggest_categorical("flux", [True, False])
    # num_channels = trial.suggest_int("num_channels", 16, 64)
    # num_attention_layers = 5
    down_sample_factor = trial.suggest_int("down_sample_factor", 2, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 8, 8)
    num_conv_layers = trial.suggest_int("num_conv_layer", 2, 3)
    # if num_conv_layers > 0:
    #     if num_conv_layers == 1:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    #     else:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    #     num_channels = trial.suggest_categorical("num_channels", [16, 24, 32])
    # else:
    #     num_channels = 1
    #     channel_multiplication = 1
    channel_multiplication = 2
    num_channels = 32 # trial.suggest_categorical("num_channels", [16, 24, 32])
    cnn_dropout = round(trial.suggest_float("cnn_dropout", 0.0, 0.9, step=0.05), 2)
    dense_dropout = round(trial.suggest_float("dense_dropout", 0.0, 0.9, step=0.05), 2)
    # classifier_dim = trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 11, log=True)
    classifier_dim = trial.suggest_categorical("classifier_dim", [64 ,128, 256, 512])
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
        epochs=50,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=5,
        ema=False,
        min_save_score=0.80,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02),
        dataset_version="S",
        full_length_test=True,
    )
    model_settings = CNNSettings(
        num_channels=num_channels,
        cnn_dropout=cnn_dropout,
        dense_dropout=dense_dropout,
        activation=activation,
        flux=flux,
        causal=True,
        num_feature_layers=num_conv_layers,
        num_residual_blocks=0,
        down_sample_factor=down_sample_factor,
        channel_multiplication=channel_multiplication,
        classifier_dim=classifier_dim,
    )
    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.55, detect_tolerance=0.05)

    # test_model = CNN(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
    #                           n_mels=n_mels)
    # input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    # with torch.inference_mode():
    #     test_model.eval()
    #     flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
    #                                           print_results=False)
    # trial.set_user_attr("flops", flops)
    # trial.set_user_attr("macs", macs)
    # trial.set_user_attr("params", params)
    # del test_model

    # train model
    scores = []
    # train model
    for i in range(1, 4):
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=model_settings, evaluation=evaluate_settings),
            trial=trial if i == 1 else None, metric_to_track="F-Score/Sum/Validation", seed=i * 20
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)
    torch.cuda.empty_cache()

    return score


def mamba_objective(trial: Trial):
    lr = 5e-4 # trial.suggest_float("lr", 7e-5, 9e-4)
    # weight_decay = trial.suggest_float("weight_decay", 1e-30, 5e-3, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-20, 1e-10, 1e-6, 1e-3])
    beta_1 = 0.9 # trial.suggest_float("beta_1", 0.8, 1)
    beta_2 = 0.999 # trial.suggest_float("beta_2", 0.8, 1)
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = True # trial.suggest_categorical("decoupled_weight_decay", [True, False])
    # frame_length = trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_length = trial.suggest_categorical("segment_length", [4.0, 8.0])
    batch_size = 8 if frame_length == 4.0 else 4 # trial.suggest_int("batch_size", 4, 64)
    frame_overlap = 0 # trial.suggest_float("segment_overlap", 0.1, min(frame_length - 0.1, 2), step=0.1)
    n_mels = 64 # trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = False # trial.suggest_categorical("flux", [True, False])
    n_layers = 5 # trial.suggest_int("n_layers", 3, 8)
    expand = 2 # trial.suggest_int("expansion_factor", 1, 4)
    down_sample_factor = trial.suggest_int("down_sample_factor", 2, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = 2 # trial.suggest_int("num_conv_layer", 0, max_layers)
    # if num_conv_layers > 0:
    #     if num_conv_layers == 1:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    #     else:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    #     # num_channels = trial.suggest_int("num_channels", 16, 64)
    #     num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 48])
    # else:
    #     num_channels = 1
    #     channel_multiplication = 1

    num_channels = 32
    channel_multiplication = 2

    # hidden_units = trial.suggest_int("hidden_units", 2 ** 4, 2 ** 12, log=True)
    hidden_units = trial.suggest_categorical("hidden_units", [64, 128, 256, 512])
    d_state = trial.suggest_categorical("d_state", [16, 32, 64])

    cnn_dropout = 0.3
    mamba_dropout = 0.5 # round(trial.suggest_float("mamba_dropout", 0.0, 0.9, step=0.1), 1)
    dense_dropout = 0.5 # round(trial.suggest_float("dense_dropout", 0.0, 0.9, step=0.1), 1)

    # classifier_dim = trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 11, log=True)
    classifier_dim = trial.suggest_categorical("classifier_dim", [64, 128, 256, 512])
    activation = trial.suggest_categorical("activation", ["selu", "relu", "elu", "silu"])

    for t in trial.study.trials:
        if t.state != TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            raise optuna.TrialPruned()

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
        epochs=50,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=7,
        ema=False,
        min_save_score=0.80,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02),
        dataset_version="S",
        full_length_test=True,
        train_set="a2md_train",
        k_folds=5,
    )

    model_settings = CNNMambaSettings(
        d_state=d_state,
        d_conv=4,
        expand=expand,
        flux=flux,
        activation=activation,
        causal=True,
        num_channels=num_channels,
        n_layers=n_layers,
        cnn_dropout=cnn_dropout,
        mamba_dropout=mamba_dropout,
        dense_dropout=dense_dropout,
        down_sample_factor=down_sample_factor,
        num_conv_layers=num_conv_layers,
        channel_multiplication=channel_multiplication,
        classifier_dim=classifier_dim,
        hidden_units=hidden_units,
    )

    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.65, detect_tolerance=0.05)

    # test_model = CNNMambaFast(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
    #                           n_mels=n_mels)
    # test_model = test_model.to("cuda")
    # input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate
    # with torch.inference_mode():
    #     flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
    #                                           print_results=False)
    # trial.set_user_attr("flops", flops)
    # trial.set_user_attr("macs", macs)
    # trial.set_user_attr("params", params)
    # del test_model

    seed = trial.number
    trial.set_user_attr("seed", seed)

    scores = []
    # train model
    for i in range(3):
        dataset_settings.fold = i
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=model_settings, evaluation=evaluate_settings),
            trial=trial if i == 0 else None, metric_to_track="Loss/Validation", seed=seed
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)

    torch.cuda.empty_cache()

    return score


def crnn_objective(trial: Trial):
    lr = 5e-4 # trial.suggest_float("lr", 9e-6, 4e-3, log=True)
    weight_decay = 1e-10 # trial.suggest_float("weight_decay", 1e-15, 5e-3, log=True)
    beta_1 = 0.9
    beta_2 = 0.999
    decoupled_weight_decay = True
    batch_size = 8 # trial.suggest_int("batch_size", 4, 64)
    frame_length = 4.0 # round(trial.suggest_float("segment_length", 1.0, 8.0, step=0.1), 1)
    frame_overlap = 0.0 # round(trial.suggest_float("segment_overlap", 0.1, round(min(frame_length / 2, 2), 1), step=0.1), 1)
    n_mels = 64 # trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = False
    num_channels = 32 # trial.suggest_categorical("num_channels", [16, 24, 32])
    num_rnn_layers = trial.suggest_int("num_rnn_layers", 3, 5)
    # hidden_units = 128 # trial.suggest_int("hidden_units", 16, 256, log=True)
    hidden_units = trial.suggest_categorical("hidden_units", [64, 128, 256])
    down_sample_factor = trial.suggest_int("down_sample_factor", 2, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = 2 # trial.suggest_int("num_conv_layer", 0, max_layers)
    # if num_conv_layers == 1:
    #     channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    # else:
    #     channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    channel_multiplication = 2
    # classifier_dim = 256 # trial.suggest_int("classifier_dim", 2 ** 4, 256, log=True)
    classifier_dim = trial.suggest_categorical("classifier_dim", [64, 128, 256])
    activation = trial.suggest_categorical("activation", ["selu", "relu", "elu", "silu"])

    cnn_dropout = 0.3 # round(trial.suggest_float("cnn_dropout", 0.0, 0.8, step=0.05), 2)
    rnn_dropout = 0.0 # round(trial.suggest_float("rnn_dropout", 0.0, 0.8, step=0.1), 1)
    dense_dropout = 0.5 # round(trial.suggest_float("dense_dropout", 0.0, 0.8, step=0.1), 1)

    seed = trial.number
    trial.set_user_attr("seed", seed)

    for t in trial.study.trials:
        if t.state != TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            raise optuna.TrialPruned()

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
        epochs=50,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=7,
        ema=False,
        min_save_score=0.80,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02),
        dataset_version="S",
        full_length_test=True,
        train_set="a2md_train",
        k_folds=5,
    )
    model_settings = CRNNSettings(
        num_channels=num_channels,
        num_conv_layers=num_conv_layers,
        num_rnn_layers=num_rnn_layers,
        rnn_units=hidden_units,
        channel_multiplication=channel_multiplication,
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        dense_dropout=dense_dropout,
        causal=True,
        flux=flux,
        classifier_dim=classifier_dim,
        down_sample_factor=down_sample_factor,
        activation=activation,
    )
    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.65, detect_tolerance=0.05)

    # test_model = CRNN(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes, n_mels=n_mels)
    # input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    # flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
    #                                       print_results=False)
    # trial.set_user_attr("flops", flops)
    # trial.set_user_attr("macs", macs)
    # trial.set_user_attr("params", params)
    # del test_model

    # train model
    scores = []
    # train model
    for i in range(3):
        dataset_settings.fold = i
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=model_settings, evaluation=evaluate_settings),
            trial=trial if i == 0 else None, metric_to_track="F-Score/Sum/Validation", seed=seed
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)

    torch.cuda.empty_cache()

    return score


def loss_mamba_objective(trial: Trial):
    lr = trial.suggest_float("lr", 5e-5, 2e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-15, 5e-3, log=True)
    beta_1 = 0.9
    beta_2 = 0.999
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = True
    batch_size = trial.suggest_int("batch_size", 4, 32)
    frame_length = 8.0 # trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_overlap = 1.0 # trial.suggest_float("segment_overlap", 0.1, min(round(frame_length / 2, 1), 2), step=0.1)
    n_mels = 96 # trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = False
    n_layers = 5 # trial.suggest_int("n_layers", 1, 10)
    expand = 2 # trial.suggest_int("expansion_factor", 1, 4)
    down_sample_factor = 2 # trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = 2 # trial.suggest_int("num_conv_layer", 0, max_layers)
    # if num_conv_layers > 0:
    #     if num_conv_layers == 1:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    #     else:
    #         channel_multiplication = trial.suggest_int("channel_multiplication", 1, 2)
    #     # num_channels = trial.suggest_int("num_channels", 16, 64)
    #     num_channels = trial.suggest_categorical("num_channels", [16, 32])
    # else:
    #     num_channels = 1
    #     channel_multiplication = 1

    num_channels = 32
    channel_multiplication = 2

    hidden_units = 128 # trial.suggest_int("hidden_units", 2 ** 4, 2 ** 8, log=True)
    d_state = 16 # trial.suggest_categorical("d_state", [16, 32, 64, 128])
    cnn_dropout = round(trial.suggest_float("cnn_dropout", 0.0, 0.8, step=0.05), 2)
    mamba_dropout = round(trial.suggest_float("mamba_dropout", 0.0, 0.8, step=0.1), 1)
    dense_dropout = round(trial.suggest_float("dense_dropout", 0.0, 0.8, step=0.1), 1)

    classifier_dim = 64 # trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 8, log=True)
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
        epochs=30,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=5,
        ema=False,
        min_save_score=0.8,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02),
        dataset_version="S",
        full_length_test=True,
        splits=[0.8, 0.2, 0.0],
    )

    model_settings = CNNMambaSettings(
        d_state=d_state,
        d_conv=4,
        expand=expand,
        flux=flux,
        activation=activation,
        causal=True,
        num_channels=num_channels,
        n_layers=n_layers,
        backbone="cnn",
        cnn_dropout=cnn_dropout,
        mamba_dropout=mamba_dropout,
        dense_dropout=dense_dropout,
        down_sample_factor=down_sample_factor,
        num_conv_layers=num_conv_layers,
        channel_multiplication=channel_multiplication,
        classifier_dim=classifier_dim,
        hidden_units=hidden_units,
    )

    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.65, detect_tolerance=0.050)

    # test_model = CNNMambaFast(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
    #                           n_mels=n_mels)
    # test_model = test_model.to("cuda")
    # input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate
#
    # with torch.inference_mode():
    #     flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
    #                                           print_results=False)
    # trial.set_user_attr("flops", flops)
    # trial.set_user_attr("macs", macs)
    # trial.set_user_attr("params", params)
#
    # del test_model

    # trial.set_user_attr("seed", trial.number)

    scores = []
    # train model
    for i in range(3):
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=model_settings, evaluation=evaluate_settings),
            trial=trial if i == 0 else None, metric_to_track="F-Score/Sum/Validation", seed=i * 20
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)

    torch.cuda.empty_cache()

    return score


def vogl_objective(trial: Trial):
    lr = trial.suggest_float("lr", 1e-4, 9e-4)
    weight_decay = trial.suggest_float("weight_decay", 1e-30, 5e-3, log=True)
    beta_1 = 0.9 # trial.suggest_float("beta_1", 0.8, 1)
    beta_2 = 0.999 # trial.suggest_float("beta_2", 0.8, 1)
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = True # trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 2, 64)
    frame_length = trial.suggest_categorical("segment_length", [4.0, 8.0]) # round(trial.suggest_float("segment_length", 1.0, 10.0, step=0.1), 1)
    frame_overlap = trial.suggest_categorical("segment_overlap", [0.0, 0.5, 1.0]) # round(trial.suggest_float("segment_overlap", 0.1, min(frame_length / 2, 2), step=0.1), 1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])

    train_settings = TrainingSettings(
        epochs=50,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        early_stopping=7,
        ema=False,
        model_settings="vogl",
        min_save_score=0.8,
    )

    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=2048),
        annotation_settings=AnnotationSettings(time_shift=0.02),
        dataset_version="S",
        full_length_test=True,
        k_folds=5,
        splits=[0.8, 0.2, 0.0],
    )


    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.6, detect_tolerance=0.05)

    scores = []
    # train model
    for i in range(3): # high variance seeds
        dataset_settings.fold = i
        model, score = train_model(
            Config(training=train_settings, dataset=dataset_settings,
                   model=None, evaluation=evaluate_settings),
            trial=trial if i == 0 else None, metric_to_track="F-Score/Sum/Validation", seed=i
        )
        del model
        scores.append(score)

    report_cross_validation_scores(trial, scores)

    trial.set_user_attr("std", np.std(scores))

    score = np.mean(scores)

    torch.cuda.empty_cache()

    return score



def final_configs_objective(trial: Trial):
    configs = final_experiment_params.keys()
    selected_config = trial.suggest_categorical("config", configs)

    seed = trial.number if "seed" not in trial.user_attrs else trial.user_attrs["seed"]
    trial.set_user_attr("seed", seed)

    params: dict[str, Any] = final_experiment_params[selected_config]
    config = Config.from_flat_dict(params.copy())

    model, score = train_model(config=config, trial=trial, metric_to_track="F-Score/Validation", seed=seed)

    del model

    torch.cuda.empty_cache()

    return score


def time_shift_objective(trial: Trial):
    configs = final_experiment_params.keys()
    selected_config = trial.suggest_categorical("config", configs)

    seed = trial.number if "seed" not in trial.user_attrs else trial.user_attrs["seed"]
    trial.set_user_attr("seed", seed)

    params: dict[str, Any] = final_experiment_params[selected_config]
    params["time_shift"] = trial.suggest_categorical("time_shift", [0.005, 0.035])
    config = Config.from_flat_dict(params.copy())

    if selected_config == "Mamba best":
        readd_trial("time_shift", {"config": selected_config, "time_shift": params["time_shift"]}, user_attrs={"seed": seed, "number": trial.user_attrs["number"]})
        return None

    model, score = train_model(config=config, trial=trial, metric_to_track="F-Score/Validation", seed=seed)

    del model

    torch.cuda.empty_cache()

    return score


def readd_trial(study_name: str, params: dict[str, Any], user_attrs: dict):
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study: Study = load_study(
        study_name=study_name,
        storage=storage,
        sampler=samplers.RandomSampler(),
        pruner=None
    )

    study.enqueue_trial(params=params, user_attrs=user_attrs, skip_if_exists=False)


def activation_dist(x, y):
    order = ["relu", "silu", "elu", "selu"]
    return float(abs(order.index(x) - order.index(y)))


def attention():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cnn_attention_fixed_model"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )

    module = optunahub.load_module(package="samplers/auto_sampler")
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=module.AutoSampler(),
        pruner=pruners.HyperbandPruner(min_resource=1, max_resource=50),
    )

    study.optimize(attention_objective, n_trials=400, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def vogl():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "vogl_cross_validation_stacked_plus_dense"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"

    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )

    module = optunahub.load_module(package="samplers/auto_sampler")
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=module.AutoSampler(),
        # sampler=samplers.TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=40),
        # pruner=opt.pruners.PatientPruner(opt.pruners.HyperbandPruner(min_resource=1, max_resource=50), patience=3),
        pruner=pruners.HyperbandPruner(min_resource=1, max_resource=50),
    )

    terminator = Terminator(improvement_evaluator=RegretBoundEvaluator(), error_evaluator=CrossValidationErrorEvaluator(), min_n_trials=60)

    study.optimize(vogl_objective, n_trials=300, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True, timeout=60 * 60 * 48, callbacks=[TerminatorCallback(terminator=terminator)])


def cnn():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cnn_fixed_model"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )

    module = optunahub.load_module(package="samplers/auto_sampler")
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=module.AutoSampler(),
        pruner=pruners.PatientPruner(pruners.HyperbandPruner(min_resource=1, max_resource=50, bootstrap_count=10), patience=3),
    )

    study.optimize(cnn_objective, n_trials=200, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def mamba():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mamba_grid_search_tpe_loss_less_params"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )

    dist_funcs = defaultdict(lambda: lambda x, y: float(np.abs(x - y)))
    dist_funcs["activation"] = activation_dist

    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=samplers.TPESampler(multivariate=True, group=True, constant_liar=True, categorical_distance_func=dist_funcs, n_startup_trials=60),
        pruner=pruners.HyperbandPruner(min_resource=1, max_resource=50),
    )

    study.optimize(mamba_objective, n_trials=400, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def crnn():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "crnn_grid_search_tpe"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    module = optunahub.load_module(package="samplers/auto_sampler")

    dist_funcs = {
        "hidden_units": lambda x, y: float(np.abs(x - y)),
        "classifier_dim": lambda x, y: float(np.abs(x - y)),
        "down_sample_factor": lambda x, y: float(np.abs(x - y)),
        "num_rnn_layers": lambda x, y: float(np.abs(x - y)),
        "activation": activation_dist,
    }
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=30, multivariate=True, group=True, constant_liar=True, categorical_distance_func=dist_funcs),
        pruner=pruners.HyperbandPruner(min_resource=1, max_resource=50, bootstrap_count=0)
    )


    study.optimize(crnn_objective, n_trials=324, catch=(torch.cuda.OutOfMemoryError, RuntimeError), gc_after_trial=True)


def loss():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mamba_constant_small_model_artist_splits"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=module.AutoSampler(),
        # sampler=opt.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=40),
        pruner=pruners.HyperbandPruner(min_resource=1, max_resource=30, bootstrap_count=10),
    )

    study.optimize(loss_mamba_objective, n_trials=200, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True) # , callbacks=[TerminatorCallback()]


def final_configs():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "final_configs_fixed"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=samplers.RandomSampler(),
        pruner=None
    )

    repetitions_to_add = 0
    for i in range(repetitions_to_add):
        for config in final_experiment_params.keys():
            study.enqueue_trial({
                "config": config
            }, skip_if_exists=False, user_attrs={"number": i})

    # study.enqueue_trial({
    #     "config": "Attention no conv",
    # }, skip_if_exists=False, user_attrs={"number": 2, "seed": 40})

    study.optimize(final_configs_objective, n_trials=1, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)



def time_shift():
    optuna_logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "time_shift"
    storage_name = "postgresql://BachelorarbeitSync:BachelorarbeitSyncPlsDontHackMe@192.168.2.206:5432"
    storage = storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study: Study = create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=samplers.RandomSampler(),
        pruner=None
    )

    configs = ["Mamba best", "Attention no conv", "Mamba no conv", "CRNN best", "CRNN no conv"]
    shifts = [0.005, 0.035]

    repetitions_to_add = 0
    for i in range(repetitions_to_add):
        for shift in shifts:
            for config in configs:
                study.enqueue_trial({
                    "config": config,
                    "time_shift": shift
                }, skip_if_exists=False, user_attrs={"number": i})

    # for trial in study.get_trials(deepcopy=True, states=[opt.trial.TrialState.FAIL]):
    #     study.enqueue_trial(params=trial.params, user_attrs=trial.user_attrs)

    # study.enqueue_trial({
    #     "config": "Mamba no conv",
    #     "time_shift": 0.035
    # }, skip_if_exists=False, user_attrs={"number": 2, "seed": 27})

    study.optimize(time_shift_objective, n_trials=1, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)

if __name__ == '__main__':
    mamba()
