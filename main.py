import random
from dataclasses import asdict as dataclass_asdict

import matplotlib.pyplot as plt
import numpy as np
import resource

import optuna
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.datasets import get_dataset
from dataset.mapping import DrumMapping
from evallib import (
    peak_pick_max_mean,
    calculate_pr,
    calculate_f_score,
    evaluate_onsets,
    combine_onsets,
)
from dataset.generics import ADTDataset
from model import ModelEmaV2
from model.CRNN import CRNN, CRNN_Vogl
from model.cnn import CNN
from model.cnnA import CNNAttention
from model.cnnM2 import CNNMambaFast
from settings import (
    CNNSettings,
    EvaluationSettings,
    CNNAttentionSettings,
    CNNMambaSettings,
    asdict,
    CRNNSettings,
    Config,
)

COLORS = (
    np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
            [128, 128, 128],
            [0, 0, 0],
        ]
    )
    / 255
)


def step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    audio_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
    scaler: torch.amp.GradScaler,
    scheduler: optim.lr_scheduler.LRScheduler = None,
    clip_value: float = 1.0,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    model.train()
    optimizer.zero_grad()

    device = str(audio_batch.device)
    device = "cuda" if "cuda" in device else "cpu"

    with torch.autocast(device_type=device, dtype=torch.float16):
        prediction = model(audio_batch)
        unfiltered = criterion(prediction, lbl_batch)
        loss = unfiltered[lbl_batch != -1].mean()

    scaler.scale(loss).backward()

    # gradient clipping
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    return loss.item() if not torch.isnan(loss) else 0.0


def train_epoch(
    epoch,
    dataloader_train,
    device,
    ema_model,
    error,
    model,
    optimizer,
    scaler,
    scheduler,
    tensorboard_writer=None,
    clip_value: float = 1.0,
):
    total_loss = 0
    for _, data in tqdm(
        enumerate(dataloader_train),
        total=len(dataloader_train),
        unit="mini-batch",
        smoothing=0.1,
        mininterval=1 / 2 * 60 / len(dataloader_train),
        desc="Training",
        dynamic_ncols=True,
    ):
        audio, lbl, _ = data
        audio = audio.to(device)
        lbl = lbl.to(device)

        loss = step(
            model=model,
            criterion=error,
            optimizer=optimizer,
            audio_batch=audio,
            lbl_batch=lbl,
            scaler=scaler,
            scheduler=scheduler,
            clip_value=clip_value,
        )
        if ema_model is not None:
            ema_model.update(model)
        total_loss += loss
    loss = total_loss / len(dataloader_train)
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("Loss/Train", loss, global_step=epoch)
    return total_loss / len(dataloader_train)


def evaluate(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    evaluation_settings: EvaluationSettings,
    tensorboard_writer: SummaryWriter = None,
    tag: str = "Evaluation",
    thresholds: list[float] = None,
) -> tuple[float, float, float, list[float]]:
    model.eval()
    total_loss = 0
    device_str = str(device)
    device_str = "cuda" if "cuda" in device_str else "cpu"
    dataset: ADTDataset = dataloader.dataset
    mapping = dataset.mapping
    sample_rate, hop_size, fft_size, time_shift = (
        dataset.sample_rate,
        dataset.hop_size,
        dataset.fft_size,
        dataset.time_shift,
    )
    predictions = []
    groundtruth = []
    with torch.inference_mode():
        for data in tqdm(
            dataloader,
            total=len(dataloader),
            unit="mini-batch",
            smoothing=0.1,
            mininterval=1 / 2 * 60 / len(dataloader),
            desc="Evaluation",
            dynamic_ncols=True,
        ):
            audio, lbl, gts = data
            audio = audio.to(device)
            lbl = lbl.to(device)

            with torch.autocast(device_type=device_str, dtype=torch.float16):
                prediction = model(audio)
                unfiltered = criterion(prediction, lbl)
                filtered_pred = prediction.sigmoid() * (lbl != -1)
            peaks = peak_pick_max_mean(
                filtered_pred.cpu().detach().float(),
                sample_rate,
                hop_size,
                fft_size,
                evaluation_settings.peak_mean_range,
                evaluation_settings.peak_max_range,
            )
            # Shift back onsets
            for song_idx in range(len(peaks)):
                for cls_idx in range(len(peaks[song_idx])):
                    peaks[song_idx][cls_idx] -= time_shift
                    # filter out predictions before the start
                    peaks[song_idx][cls_idx] = peaks[song_idx][cls_idx][
                        :, peaks[song_idx][cls_idx][0, :] >= 0
                    ]

            if thresholds is not None:
                peaks = [
                    [
                        combine_onsets(
                            cls[0, :][cls[1, :] >= thresh],
                            cool_down=evaluation_settings.onset_cooldown,
                        )
                        for cls, thresh in zip(song, thresholds)
                    ]
                    for song in peaks
                ]
            # peaks = [[peak - time_shift for peak in cls] for cls in peaks]
            predictions.extend(peaks)
            groundtruth.extend(gts)
            loss = unfiltered[lbl != -1].mean()
            total_loss += loss.item()
            torch.cuda.empty_cache()

    best_thresholds = thresholds
    if thresholds is None:
        precisions, recalls, thresholds, f_sum, f_avg, best_thresholds = calculate_pr(
            predictions,
            groundtruth,
            onset_cooldown=evaluation_settings.onset_cooldown,
            detection_window=evaluation_settings.detect_tolerance,
            ignore_beats=evaluation_settings.ignore_beats,
            pr_points=evaluation_settings.pr_points,
        )
        f_scores = [
            calculate_f_score(precision, recall)
            for precision, recall in zip(precisions, recalls)
        ]
        for score in f_scores:
            assert not torch.isnan(score).any()
        if tensorboard_writer is not None:
            tensorboard_writer.add_tensor(
                f"{tag}/Best_Thresholds", best_thresholds, global_step=epoch
            )
            # FixMe This might break with different settings
            tensorboard_writer.add_tensor(
                f"{tag}/Precisions", torch.stack(precisions), global_step=epoch
            )
            tensorboard_writer.add_tensor(
                f"{tag}/Recalls", torch.stack(recalls), global_step=epoch
            )
            tensorboard_writer.add_tensor(
                f"{tag}/F_scores", torch.stack(f_scores), global_step=epoch
            )
            tensorboard_writer.add_tensor(
                f"{tag}/Thresholds", torch.stack(thresholds), global_step=epoch
            )

            fig = plt.figure()
            for i in range(len(precisions)):
                plt.plot(
                    recalls[i],
                    precisions[i],
                    color=COLORS[i],
                    label=DrumMapping.prettify(mapping[i]),
                )

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve for {tag}")
            plt.legend()
            plt.tight_layout()
            tensorboard_writer.add_figure(
                f"{tag}/PR-Curve/", fig, global_step=epoch, close=True
            )

            fig = plt.figure()
            for i in range(len(f_scores)):
                plt.plot(
                    thresholds[i],
                    f_scores[i],
                    color=COLORS[i],
                    label=DrumMapping.prettify(mapping[i]),
                )
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Threshold")
            plt.ylabel("F-Score")
            plt.title(f"F-Score for {tag}")
            plt.legend()
            plt.tight_layout()
            tensorboard_writer.add_figure(
                f"{tag}/Threshold-Curve/", fig, global_step=epoch, close=True
            )
    else:
        beats = (len(groundtruth[0]) - dataset.n_classes) == 0
        stats_counter = torch.zeros(
            dataset.n_classes - 2 * (beats - (1 - evaluation_settings.ignore_beats)), 3
        )
        assert len(predictions) == len(groundtruth), (
            f"{len(predictions)} != {len(groundtruth)}"
        )
        for song_index in range(len(predictions)):
            offset = 0
            if evaluation_settings.ignore_beats:
                offset = 2
                if len(predictions[song_index]) == len(groundtruth[song_index]):
                    predictions[song_index] = predictions[song_index][2:]

            assert len(predictions[song_index]) == len(groundtruth[song_index][offset:])
            for cls, (cls_onset, cls_gt) in enumerate(
                zip(predictions[song_index], groundtruth[song_index][offset:])
            ):
                stats_counter[cls] += torch.tensor(
                    evaluate_onsets(
                        cls_onset, cls_gt, window=evaluation_settings.detect_tolerance
                    )
                )
        f_scores = (
            2
            * stats_counter[:, 0]
            / (2 * stats_counter[:, 0] + stats_counter[:, 1] + stats_counter[:, 2])
        )
        f_avg = f_scores.mean().item()
        total_stats = stats_counter.sum(dim=0)
        f_sum = (
            2 * total_stats[0] / (2 * total_stats[0] + total_stats[1] + total_stats[2])
        ).item()
        if tensorboard_writer is not None:
            tensorboard_writer.add_tensor(
                f"Stats/{tag}", stats_counter, global_step=epoch
            )
            tensorboard_writer.add_tensor(
                f"F-Score/Class/{tag}", f_scores, global_step=epoch
            )

    loss = total_loss / len(dataloader)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(f"Loss/{tag}", loss, global_step=epoch)
        tensorboard_writer.add_scalar(f"F-Score/Sum/{tag}", f_sum, global_step=epoch)
        tensorboard_writer.add_scalar(f"F-Score/Avg/{tag}", f_avg, global_step=epoch)

    return loss, f_sum, f_avg, best_thresholds


def main(
    config: Config = Config(),
    trial: optuna.Trial = None,
    metric_to_track="F-Score/Sum/Validation",
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    stuck_threshold = 4
    min_delta = 1e-5

    training_settings = config.training
    evaluation_settings = config.evaluation
    dataset_settings = config.dataset
    model_settings = config.model

    early_stopping = training_settings.early_stopping  # if trial is None else None
    n_classes = dataset_settings.annotation_settings.n_classes
    n_mels = dataset_settings.audio_settings.n_mels

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    architecture = (
        training_settings.model_settings
        if model_settings is None
        else model_settings.get_identifier()
    )

    match architecture:
        case "cnn":
            model_settings = CNNSettings() if model_settings is None else model_settings
            model = CNN(
                **dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels
            )
        case "cnn_attention":
            model_settings = (
                CNNAttentionSettings() if model_settings is None else model_settings
            )
            model = CNNAttention(
                **dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels
            )
        case "mamba_fast" | "mamba":
            model_settings = (
                CNNMambaSettings() if model_settings is None else model_settings
            )
            model = CNNMambaFast(
                **dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels
            )
        case "crnn":
            model_settings = (
                CRNNSettings() if model_settings is None else model_settings
            )
            model = CRNN(
                **dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels
            )
        case "vogl":
            model = CRNN_Vogl(n_classes=n_classes, n_mels=n_mels, causal=True)
        case _:
            raise ValueError("Invalid model setting")
    model.to(device)
    training_settings.model_settings = architecture

    # Multiprocessing headaches
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    torch.multiprocessing.set_sharing_strategy("file_system")

    print(training_settings)
    print(evaluation_settings)
    print(dataset_settings)
    print(model_settings)

    loader_train, loader_val, test_sets = get_dataset(
        training_settings.batch_size,
        training_settings.test_batch_size,
        dataset_settings=dataset_settings,
        seed=seed,
    )

    ema_model = (
        ModelEmaV2(model, decay=training_settings.ema, device=device)
        if training_settings.ema is not None
        else None
    )
    best_model = None

    writer = SummaryWriter()

    metrics = {
        "Loss/Train": [],
        "Loss/Validation": [],
        "F-Score/Sum/Validation": [],
        "F-Score/Avg/Validation": [],
    }
    directions = {
        "Loss/Train": -1,
        "Loss/Validation": -1,
        "F-Score/Sum/Validation": 1,
        "F-Score/Avg/Validation": 1,
    }

    if training_settings.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_settings.learning_rate,
            weight_decay=training_settings.weight_decay,
            decoupled_weight_decay=training_settings.decoupled_weight_decay,
            betas=(training_settings.beta_1, training_settings.beta_2),
            eps=training_settings.epsilon,
        )
    elif training_settings.optimizer == "radam":
        optimizer = optim.RAdam(
            model.parameters(),
            lr=training_settings.learning_rate,
            weight_decay=training_settings.weight_decay,
            decoupled_weight_decay=training_settings.decoupled_weight_decay,
            betas=(training_settings.beta_1, training_settings.beta_2),
            eps=training_settings.epsilon,
        )
    else:
        raise ValueError("Invalid optimizer")

    scheduler = (
        # warmup-stable-decay schedule
        # warmup / anneal for at most 10% of the epochs each, but at least 5 epochs
        optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                # LinearLR keeps the LR constant after the warmup phase if end_factor = 1.0
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=max(training_settings.epochs // 10, 5),
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(training_settings.epochs // 10, 5)
                ),
            ],
            [training_settings.epochs - max(training_settings.epochs // 10, 5)],
        )
        # optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min" if directions[metric_to_track] < 0 else "max",
        #     factor=0.2,
        #     patience=10,
        #     eps=training_settings.epsilon,
        # )
        # optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=2e-3,
        #     div_factor= 2e-3 / training_settings.learning_rate,
        #     final_div_factor=training_settings.learning_rate / 8e-6, # min_lr = 8e-6
        #     steps_per_epoch=len(loader_train),
        #     epochs=training_settings.epochs,
        # )
        if training_settings.scheduler
        else None
    )
    current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
    # error = torch.nn.L1Loss(reduction="none")
    trainset: ADTDataset = loader_train.dataset
    num_pos, num_neg = trainset.get_sample_distribution()
    # weights according to https://markcartwright.com/files/cartwright2018increasing.pdf section 3.4.1 Task weights
    total = (num_pos + num_neg)[0]
    # if dataset_settings.annotation_settings.pad_annotations:
    #     num_pos = num_pos * (1 + 2 * dataset_settings.annotation_settings.pad_value)
    p_i = num_pos / total
    weight: torch.Tensor = 1 / (-p_i * p_i.log() - (1 - p_i) * (1 - p_i).log())
    weight = weight / 4  # shift weight closer to the ones used by Zheren and Vogl
    weight[0 + 2 * dataset_settings.annotation_settings.beats] = (
        1.0  # don't weigh kick as it is easy and common
    )
    # weight = None
    print(weight.numpy())
    weight = weight.unsqueeze(-1).to(device) if weight is not None else None
    error = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=weight)
    scaler = torch.amp.GradScaler(device=device)

    last_improvement = 0
    print("Starting Training")
    for epoch in range(training_settings.epochs):
        train_loss = train_epoch(
            epoch,
            loader_train,
            device,
            ema_model,
            error,
            model,
            optimizer,
            scaler,
            scheduler,
            tensorboard_writer=writer,
            clip_value=training_settings.gradient_clip_norm,
        )
        torch.cuda.empty_cache()
        val_loss, f_score_sum, f_score_avg, best_thresholds = evaluate(
            epoch,
            model if ema_model is None else ema_model.module,
            loader_val,
            error,
            device,
            evaluation_settings,
            tensorboard_writer=writer,
            tag="Validation",
        )
        metrics["Loss/Train"].append(train_loss)
        metrics["Loss/Validation"].append(val_loss)
        metrics["F-Score/Sum/Validation"].append(f_score_sum)
        metrics["F-Score/Avg/Validation"].append(f_score_avg)
        if trial is not None:
            trial.report(f_score_sum, epoch)
        torch.cuda.empty_cache()
        if scheduler is not None and isinstance(
            scheduler, optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(metrics[metric_to_track][-1])
        if scheduler is not None and not isinstance(
            scheduler,
            (optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler.OneCycleLR),
        ):
            scheduler.step()
        if scheduler is not None:
            if current_lr != scheduler.get_last_lr()[0]:
                current_lr = scheduler.get_last_lr()[0]
                logging.info("Setting learning rate to {:.6f}".format(current_lr))

        print(
            f"Epoch: {epoch + 1} "
            f"Loss: {train_loss * 100:.4f}\t "
            f"Val Loss: {val_loss * 100:.4f} F-Score: {f_score_avg * 100:.4f}/{f_score_sum * 100:.4f}"
        )
        last_improvement += 1
        if epoch == 0 or np.all(
            np.max(
                np.array(metrics[metric_to_track][:-1]) * directions[metric_to_track],
                axis=0,
            )
            + directions[metric_to_track] * min_delta
            < np.array(metrics[metric_to_track][-1]) * directions[metric_to_track]
        ):
            best_model = (
                ema_model.module if ema_model is not None else model
            ).state_dict()
            last_improvement = 0
        print(last_improvement)
        if f_score_sum > evaluation_settings.min_test_score and last_improvement == 0:
            for test_loader in test_sets:
                test_set: ADTDataset = test_loader.dataset
                identifier = test_set.get_identifier()
                test_loss, test_f_score, test_avg_f_score, _ = evaluate(
                    epoch,
                    model if ema_model is None else ema_model.module,
                    test_loader,
                    error,
                    device,
                    evaluation_settings,
                    tensorboard_writer=writer,
                    tag=f"Test/{identifier}",
                    thresholds=best_thresholds,
                )
                if trial is not None:
                    trial.set_user_attr(f"{identifier}_f_score_sum", test_f_score)
                    trial.set_user_attr(f"{identifier}_f_score_avg", test_avg_f_score)
                torch.cuda.empty_cache()
                print(
                    f"{identifier}: Test Loss: {test_loss * 100:.4f} F-Score: {test_avg_f_score * 100:.4f}/{test_f_score * 100:.4f}"
                )
        recent_scores = np.array(metrics["F-Score/Sum/Validation"][-stuck_threshold:])
        stuck = (
            (np.abs(recent_scores - recent_scores.mean(axis=0)) < 1e-5).all()
            and len(recent_scores) >= stuck_threshold
        ) or np.isnan(val_loss)
        if (
            early_stopping is not None
            and last_improvement >= training_settings.early_stopping
        ) or stuck:
            if stuck:
                print("Detected stuck training")
                if trial is not None:
                    raise optuna.TrialPruned()
            break
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()

    if best_model is not None:
        model.load_state_dict(best_model)

    if max(metrics["F-Score/Sum/Validation"]) > training_settings.min_save_score:
        print("Saving model")
        dic = {
            "model": model.state_dict(),
            "dataset_settings": asdict(dataset_settings),
            "training_settings": asdict(training_settings),
            "evaluation_settings": asdict(evaluation_settings),
        }
        if model_settings is not None:
            dic["model_settings"] = asdict(model_settings)
        dir_name = writer.get_logdir().split("/")[-1]
        torch.save(
            dic,
            f"./models/{dir_name}_{architecture}_{metrics['F-Score/Sum/Validation'][-1] * 100:.2f}.pt",
        )

    hyperparameters = {
        **asdict(training_settings),
        **asdict(evaluation_settings),
        **asdict(dataset_settings),
        "seed": seed,
    }
    if model_settings is not None:
        hyperparameters.update(asdict(model_settings))
    best_score = np.max(metrics["F-Score/Sum/Validation"], axis=0)
    writer.add_hparams(hparam_dict=hyperparameters, metric_dict={"F-Score": best_score})
    print(f"Best F-score: {best_score * 100:.4f}")

    # Make sure everything is written to disk
    writer.flush()
    writer.close()

    return model, best_score


if __name__ == "__main__":
    trained_model, _ = main()
    trained_model.eval()
    trained_model = trained_model.cpu()
    torch.cuda.empty_cache()
