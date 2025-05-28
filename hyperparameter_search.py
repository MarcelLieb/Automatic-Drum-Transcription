import logging
import sys
from dataclasses import asdict
from typing import Any

import numpy as np
import optuna
from calflops import calculate_flops
import torch
from torch import nn

from model.CRNN import CRNN
from model.cnnA import CNNAttention
from model.cnnM2 import CNNMambaFast
from settings import DatasetSettings, TrainingSettings, CNNMambaSettings, EvaluationSettings, AudioProcessingSettings, \
    CRNNSettings, AnnotationSettings, Config, CNNAttentionSettings
from main import main as train_model
from hyperparameters import final_experiment_params


def attention_objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 4e-5, 6e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 5e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 1)
    beta_2 = trial.suggest_float("beta_2", 0.5, 1)
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 4, 64, log=True)
    frame_length = trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_overlap = trial.suggest_float("segment_overlap", 0.1, min(frame_length, 2), step=0.1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    # num_channels = trial.suggest_int("num_channels", 16, 64)
    num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 48])
    num_attention_layers = trial.suggest_int("num_attention_layers", 1, 8)
    # num_attention_layers = 5
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16])
    expansion_factor = trial.suggest_int("expansion_factor", 1, 4)
    use_relative_pos = trial.suggest_categorical("use_relative_pos", [True, False])
    down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = trial.suggest_int("num_conv_layer", 0, max_layers)
    if num_conv_layers == 1:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    else:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    # hidden_units = trial.suggest_int("hidden_units", 2 ** 4, 2 ** 11, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.9, step=0.05)
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
        epochs=30,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        dataset_version="M",
        early_stopping=7,
        full_length_test=False,
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
    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.45)

    test_model = CNNAttention(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
                              n_mels=n_mels)
    input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    with torch.inference_mode():
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


def mamba_objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 6e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 5e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.8, 1)
    beta_2 = trial.suggest_float("beta_2", 0.8, 1)
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 4, 64, log=True)
    frame_length = trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_overlap = trial.suggest_float("segment_overlap", 0.1, min(frame_length - 0.1, 2), step=0.1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    n_layers = trial.suggest_int("n_layers", 1, 20)
    expand = trial.suggest_int("expansion_factor", 1, 4)
    down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = trial.suggest_int("num_conv_layer", 0, max_layers)
    if num_conv_layers > 0:
        if num_conv_layers == 1:
            channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
        else:
            channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
        # num_channels = trial.suggest_int("num_channels", 16, 64)
        num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 48])
    else:
        num_channels = 1
        channel_multiplication = 1

    hidden_units = trial.suggest_int("hidden_units", 2 ** 4, 2 ** 12, log=True)
    d_state = trial.suggest_categorical("d_state", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.7, step=0.05)
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
        epochs=30,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        # epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        dataset_version="M",
        early_stopping=7,
        full_length_test=False,
        ema=True,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=1024),
        annotation_settings=AnnotationSettings(time_shift=0.015),
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
        dropout=dropout,
        down_sample_factor=down_sample_factor,
        num_conv_layers=num_conv_layers,
        channel_multiplication=channel_multiplication,
        classifier_dim=classifier_dim,
        hidden_units=hidden_units,
    )

    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.45)

    test_model = CNNMambaFast(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
                              n_mels=n_mels)
    test_model = test_model.to("cuda")
    input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    with torch.inference_mode():
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


def crnn_objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 4e-5, 6e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 5e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.3, 1)
    beta_2 = trial.suggest_float("beta_2", 0.3, 1)
    epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 4, 64, log=True)
    frame_length = trial.suggest_float("segment_length", 1.0, 8.0, step=0.1)
    frame_overlap = trial.suggest_float("segment_overlap", 0.1, 1.0, step=0.1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    num_channels = trial.suggest_int("num_channels", 16, 64)
    num_rnn_layers = trial.suggest_int("num_rnn_layers", 1, 8)
    hidden_units = trial.suggest_int("hidden_units", 16, 2 ** 11, log=True)
    down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = trial.suggest_int("num_conv_layer", 0, max_layers)
    if num_conv_layers == 1:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
    else:
        channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
    dropout = trial.suggest_float("dropout", 0.3, 0.9, step=0.05)
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
        epochs=30,
        learning_rate=lr,
        scheduler=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        decoupled_weight_decay=decoupled_weight_decay,
        dataset_version="M",
        early_stopping=7,
        full_length_test=True,
        train_set="a2md_train",
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=1024),
        annotation_settings=AnnotationSettings(time_shift=0.015),
    )
    model_settings = CRNNSettings(
        num_channels=num_channels,
        num_conv_layers=num_conv_layers,
        num_rnn_layers=num_rnn_layers,
        rnn_units=hidden_units,
        channel_multiplication=channel_multiplication,
        dropout=dropout,
        causal=True,
        flux=flux,
        classifier_dim=classifier_dim,
        down_sample_factor=down_sample_factor,
        activation=activation,
    )
    evaluate_settings = EvaluationSettings(pr_points=400, min_test_score=0.48)

    test_model = CRNN(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes, n_mels=n_mels)
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


def loss_objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 6e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 5e-3, log=True)
    beta_1 = 0.9
    beta_2 = 0.999
    # epsilon = trial.suggest_float("epsilon", 1e-10, 1e-6, log=True)
    decoupled_weight_decay = trial.suggest_categorical("decoupled_weight_decay", [True, False])
    batch_size = trial.suggest_int("batch_size", 4, 64, log=True)
    frame_length = trial.suggest_float("segment_length", 1.0, 10.0, step=0.1)
    frame_overlap = trial.suggest_float("segment_overlap", 0.1, min(frame_length - 0.1, 2), step=0.1)
    n_mels = trial.suggest_categorical("n_mels", [64, 84, 96, 128])
    flux = trial.suggest_categorical("flux", [True, False])
    n_layers = trial.suggest_int("n_layers", 1, 20)
    expand = trial.suggest_int("expansion_factor", 1, 4)
    down_sample_factor = trial.suggest_int("down_sample_factor", 1, 4)
    max_layers = min(int(np.emath.logn(down_sample_factor, n_mels)) if down_sample_factor > 1 else 4, 4)
    num_conv_layers = trial.suggest_int("num_conv_layer", 0, max_layers)
    if num_conv_layers > 0:
        if num_conv_layers == 1:
            channel_multiplication = trial.suggest_int("channel_multiplication", 1, 1)
        else:
            channel_multiplication = trial.suggest_int("channel_multiplication", 1, 4)
        # num_channels = trial.suggest_int("num_channels", 16, 64)
        num_channels = trial.suggest_categorical("num_channels", [16, 24, 32, 48])
    else:
        num_channels = 1
        channel_multiplication = 1

    hidden_units = trial.suggest_int("hidden_units", 2 ** 4, 2 ** 7, log=True)
    d_state = trial.suggest_categorical("d_state", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.7, step=0.05)
    classifier_dim = trial.suggest_int("classifier_dim", 2 ** 4, 2 ** 8, log=True)
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
        dataset_version="S",
        early_stopping=3,
        full_length_test=False,
        ema=True,
    )
    dataset_settings = DatasetSettings(
        frame_length=frame_length,
        frame_overlap=frame_overlap,
        audio_settings=AudioProcessingSettings(n_mels=n_mels, fft_size=1024),
        annotation_settings=AnnotationSettings(time_shift=0.015),
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
        dropout=dropout,
        down_sample_factor=down_sample_factor,
        num_conv_layers=num_conv_layers,
        channel_multiplication=channel_multiplication,
        classifier_dim=classifier_dim,
        hidden_units=hidden_units,
    )

    evaluate_settings = EvaluationSettings(pr_points=100, min_test_score=0.45)

    test_model = CNNMambaFast(**asdict(model_settings), n_classes=dataset_settings.annotation_settings.n_classes,
                              n_mels=n_mels)
    test_model = test_model.to("cuda")
    input_shape = (1, n_mels, 100)  # 100 frames per second is frame rate

    with torch.inference_mode():
        flops, macs, params = calculate_flops(model=test_model, input_shape=input_shape, output_as_string=False,
                                              print_results=False)
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("macs", macs)
    trial.set_user_attr("params", params)

    del test_model

    # train model
    model, score = train_model(
        Config(training=train_settings, dataset=dataset_settings, model=model_settings, evaluation=evaluate_settings),
        trial=trial, metric_to_track="Loss/Validation")
    del model

    torch.cuda.empty_cache()

    return score, flops


def final_configs_objective(trial: optuna.Trial):
    configs = final_experiment_params.keys()
    selected_config = trial.suggest_categorical("config", configs)

    seed = trial.number
    trial.set_user_attr("seed", seed)

    params: dict[str, Any] = final_experiment_params[selected_config]
    config = Config.from_flat_dict(params.copy())

    model, score = train_model(config=config, trial=trial, metric_to_track="F-Score/Validation", seed=seed)

    del model

    torch.cuda.empty_cache()

    return score


def attention():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cnn_attention"
    storage_name = "sqlite:///local.db"
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(mamba_objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def mamba():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "mamba"
    storage_name = ""
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.enqueue_trial({
        "activation": "selu",
        "batch_size": 25,
        "beta_1": 0.9128946577712855,
        "beta_2": 0.9662584822267997,
        "channel_multiplication": 4,
        "classifier_dim": 128,
        "d_state": 64,
        "decoupled_weight_decay": True,
        "down_sample_factor": 3,
        "dropout": 0.15,
        "expansion_factor": 2,
        "flux": True,
        "hidden_units": 768,
        "lr": 2e-4,
        "n_layers": 12,
        "n_mels": 64,
        "num_channels": 48,
        "num_conv_layer": 2,
        "segment_length": 5.0,
        "segment_overlap": 2.0,
        "weight_decay": 8.0e-5,
    }, skip_if_exists=True)

    study.enqueue_trial({
        "activation": "relu",
        "batch_size": 32,
        "beta_1": 0.92,
        "beta_2": 0.98,
        "channel_multiplication": 2,
        "classifier_dim": 1024,
        "d_state": 64,
        "decoupled_weight_decay": True,
        "down_sample_factor": 3,
        "dropout": 0.2,
        "expansion_factor": 2,
        "flux": True,
        "hidden_units": 128,
        "lr": 2e-4,
        "n_layers": 6,
        "n_mels": 84,
        "num_channels": 48,
        "num_conv_layer": 4,
        "segment_length": 5.0,
        "segment_overlap": 2.0,
        "weight_decay": 4.0e-5,
    }, skip_if_exists=True)

    study.enqueue_trial({
        "activation": "relu",
        "batch_size": 32,
        "beta_1": 0.92,
        "beta_2": 0.98,
        "channel_multiplication": 1,
        "classifier_dim": 2048,
        "d_state": 64,
        "decoupled_weight_decay": True,
        "down_sample_factor": 3,
        "dropout": 0.5,
        "expansion_factor": 4,
        "flux": True,
        "hidden_units": 128,
        "lr": 2e-4,
        "n_layers": 10,
        "n_mels": 84,
        "num_conv_layer": 0,
        "segment_length": 5.0,
        "segment_overlap": 2.0,
        "weight_decay": 4.0e-5,
    }, skip_if_exists=True)

    # study.enqueue_trial(params=study.trials[-2].params)

    study.optimize(mamba_objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def crnn():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///hyperparameters.db"

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name="CRNN fixes",
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.enqueue_trial({
        "activation": "selu",
        "batch_size": 7,
        "beta_1": 0.8465832330435744,
        "beta_2": 0.9080326213407318,
        "channel_multiplication": 4,
        "classifier_dim": 460,
        "decoupled_weight_decay": True,
        "down_sample_factor": 2,
        "dropout": 0.5,
        "epsilon": 1.889851774947018e-08,
        "flux": True,
        "hidden_units": 1090,
        "lr": 9.9e-5,
        "n_mels": 84,
        "num_channels": 25,
        "num_conv_layer": 2,
        "num_rnn_layers": 4,
        "segment_length": 2.2,
        "segment_overlap": 0.2,
        "weight_decay": 2.0891162774860116e-05,
    }, skip_if_exists=True)

    study.optimize(crnn_objective, n_trials=50, catch=(torch.cuda.OutOfMemoryError, RuntimeError), gc_after_trial=True)


def loss():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "loss_mamba"
    storage_name = ""
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=20),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(loss_objective, n_trials=200, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
                   gc_after_trial=True)


def final_configs():
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "final_configs"
    storage_name = ""
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600, "pool_timeout": 3600},
        heartbeat_interval=60,
        grace_period=3600,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(),
        pruner=None
    )

    repetitions_to_add = 0
    for i in range(repetitions_to_add):
        for config in final_experiment_params.keys():
            study.enqueue_trial({
                "config": config
            }, skip_if_exists=False, user_attrs={"number": i})


    # study.optimize(loss_objective, n_trials=200, catch=(torch.cuda.OutOfMemoryError, RuntimeError),
    #                gc_after_trial=True)

if __name__ == '__main__':
    final_configs()
