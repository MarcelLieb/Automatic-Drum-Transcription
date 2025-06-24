from dataclasses import asdict as dataclass_asdict

import matplotlib
matplotlib.use('Agg')
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
from evallib import peak_pick_max_mean, calculate_pr, calculate_f_score, evaluate_onsets, combine_onsets
from dataset.generics import ADTDataset
from model import ModelEmaV2
from model.CRNN import CRNN
from model.cnn import CNN
from model.cnnA import CNNAttention
from model.cnnM import CNNMamba
from model.cnnM2 import CNNMambaFast
from model.unet import UNet
from settings import (
    CNNSettings,
    EvaluationSettings,
    CNNAttentionSettings,
    CNNMambaSettings,
    asdict, UNetSettings, CRNNSettings, Config,
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
    scaler: torch.cuda.amp.GradScaler,
    positive_weight: float,
    scheduler: optim.lr_scheduler.LRScheduler = None,
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
        no_silence = unfiltered * (lbl_batch != -1)
        # full_context = no_silence[..., 49:]  # Receptive field if causal model is 9 frames
        # full_context = full_context * torch.any(lbl_batch[..., 9:] > 0, dim=-1, keepdim=True)
        num_positives = torch.sum(lbl_batch > 0).detach()
        total = torch.prod(torch.tensor(no_silence.shape))
        scale_factor = (positive_weight - 1) * num_positives / total + 1

        mask = torch.ones_like(no_silence, requires_grad=False)
        mask[lbl_batch > 0] = positive_weight
        mask = mask / scale_factor

        no_silence = no_silence * mask

        loss = no_silence.mean()

    scaler.scale(loss).backward()

    # gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None and not isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return loss.item() if not torch.isnan(loss) else 0.0


def step_encoder(
    model: UNet,
    criterion,
    optimizer: optim.Optimizer,
    audio_batch: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: optim.lr_scheduler.LRScheduler = None,
) -> float:
    model.train()
    optimizer.zero_grad()

    device = str(audio_batch.device)
    device = "cuda" if "cuda" in device else "cpu"

    with torch.autocast(device_type=device, dtype=torch.float16):
        prediction = model(audio_batch)
        loss = criterion(prediction, audio_batch.unsqueeze(1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None and not isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    ):
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
    positive_weight,
    is_unet=False,
    tensorboard_writer=None,
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
        if is_unet:
            loss = step_encoder(
                model=model,
                criterion=error,
                optimizer=optimizer,
                audio_batch=audio,
                scaler=scaler,
                scheduler=scheduler,
            )

            if ema_model is not None:
                ema_model.update(model)

            total_loss += loss
            continue
        lbl = lbl.to(device)

        loss = step(
            model=model,
            criterion=error,
            optimizer=optimizer,
            audio_batch=audio,
            lbl_batch=lbl,
            scaler=scaler,
            scheduler=scheduler,
            positive_weight=positive_weight,
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
    is_unet: bool = False,
    tag: str = "Evaluation",
    thresholds: list[float]=None,
) -> (float, float, float, list[float]):
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

            if is_unet:
                with torch.autocast(device_type=device_str, dtype=torch.float16):
                    prediction = model(audio)
                    loss = criterion(prediction, audio.unsqueeze(1))
                    total_loss += loss.item()
                    continue

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
            for song in peaks:
                for cls in song:
                    cls[0] -= time_shift

            if thresholds is not None:
                peaks = [[combine_onsets(song[0, :][song[1, :] >= thresh], cool_down=evaluation_settings.onset_cooldown) for song in cls] for cls, thresh in zip(peaks, thresholds)]
            # peaks = [[peak - time_shift for peak in cls] for cls in peaks]
            predictions.extend(peaks)
            groundtruth.extend(gts)
            loss = unfiltered[lbl != -1].mean()
            total_loss += loss.item()
            torch.cuda.empty_cache()

    if is_unet:
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(f"Loss/{tag}", total_loss / len(dataloader), global_step=epoch)
        return total_loss / len(dataloader), 0, 0, []

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
        stats_counter = torch.zeros(dataset.n_classes, 3)
        assert len(predictions) == len(groundtruth), f"{len(predictions)} != {len(groundtruth)}"
        for song_index in range(len(predictions)):
            offset = 0
            if evaluation_settings.ignore_beats:
                offset = 2

            assert len(predictions[song_index]) == len(groundtruth[song_index][offset:])
            for cls, (cls_onset, cls_gt) in enumerate(zip(predictions[song_index], groundtruth[song_index][offset:])):
                stats_counter[cls] += torch.tensor(evaluate_onsets(cls_onset, cls_gt, window=evaluation_settings.detect_tolerance))
        f_scores = 2 * stats_counter[:, 0] / (2 * stats_counter[:, 0] + stats_counter[:, 1] + stats_counter[:, 2])
        f_avg = f_scores.mean().item()
        total_stats = stats_counter.sum(dim=1)
        f_sum = (2 * total_stats[0] / (2 * total_stats[0] + total_stats[1] + total_stats[2])).item()
        if tensorboard_writer is not None:
            tensorboard_writer.add_tensor(f"Stats/{tag}", stats_counter)
            tensorboard_writer.add_tensor(f"F-Score/Class/{tag}", f_scores)

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
    training_settings = config.training
    evaluation_settings = config.evaluation
    dataset_settings = config.dataset
    model_settings = config.model

    early_stopping = training_settings.early_stopping  # if trial is None else None
    n_classes = dataset_settings.annotation_settings.n_classes
    n_mels = dataset_settings.audio_settings.n_mels

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    architecture = training_settings.model_settings if model_settings is None else model_settings.get_identifier()

    match architecture:
        case "cnn":
            model_settings = CNNSettings() if model_settings is None else model_settings
            model = CNN(**dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels)
        case "cnn_attention":
            model_settings = CNNAttentionSettings() if model_settings is None else model_settings
            model = CNNAttention(**dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels)
        case "mamba":
            model_settings = CNNMambaSettings() if model_settings is None else model_settings
            model = CNNMamba(**dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels)
        case "mamba_fast":
            model_settings = CNNMambaSettings() if model_settings is None else model_settings
            model = CNNMambaFast(**dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels)
        case "unet":
            model_settings = UNetSettings() if model_settings is None else model_settings
            model = UNet(**dataclass_asdict(model_settings))
        case "crnn":
            model_settings = CRNNSettings() if model_settings is None else model_settings
            model = CRNN(**dataclass_asdict(model_settings), n_classes=n_classes, n_mels=n_mels)
        case _:
            raise ValueError("Invalid model setting")
    model.to(device)
    training_settings.model_settings = architecture

    is_unet = training_settings.model_settings == "unet"

    # Multiprocessing headaches
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    torch.multiprocessing.set_sharing_strategy("file_system")

    print(training_settings)
    print(evaluation_settings)
    print(dataset_settings)
    print(model_settings)

    loader_train, loader_val, test_sets = get_dataset(
        training_settings, dataset_settings=dataset_settings
    )

    ema_model = (
        ModelEmaV2(model, decay=0.998, device=device) if training_settings.ema else None
    )
    best_model = None

    writer = SummaryWriter()

    max_lr = training_settings.learning_rate * 2
    initial_lr = max_lr / 25
    _min_lr = initial_lr / 1e4
    initial_lr = (
        training_settings.learning_rate
        if not training_settings.scheduler
        else initial_lr
    )

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

    optimizer = optim.RAdam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=training_settings.weight_decay,
        decoupled_weight_decay=training_settings.decoupled_weight_decay,
        betas=(training_settings.beta_1, training_settings.beta_2),
        eps=training_settings.epsilon,
    )
    scheduler = (
        optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(loader_train),
            epochs=training_settings.epochs,
        )
        if training_settings.scheduler
        else None
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=10)
    current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
    # error = torch.nn.L1Loss(reduction="none")
    if is_unet:
        error = torch.nn.MSELoss()
    else:
        if dataset_settings.annotation_settings.n_classes == 3:
            # Weights from Vogl
            weight = torch.tensor([1, 4, 1.5])
            weight = torch.tensor([1.0780001453213364, 1.3531086684241876, 1.144276962353584])
        elif dataset_settings.annotation_settings.n_classes == 4:
            # truncated weights from ADTOF
            weight = torch.tensor([1.0780001453213364, 1.3531086684241876, 3.413723052423422, 1.144276962353584])
        elif dataset_settings.annotation_settings.n_classes == 5:
            # weights from ADTOF https://github.com/MZehren/ADTOF/blob/master/adtof/config.py
            weight = torch.tensor([1.0780001453213364, 1.3531086684241876, 3.413723052423422, 1.144276962353584, 1.76755104053326])
        else:
            weight = None
        num_pos, num_neg = loader_train.dataset.get_sample_distribution()
        weight = num_neg / num_pos / 20
        weight = None
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
            is_unet=is_unet,
            positive_weight=training_settings.positive_weight,
            tensorboard_writer=writer,
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
            is_unet=is_unet,
            tag="Validation",
        )
        metrics["Loss/Train"].append(train_loss)
        metrics["Loss/Validation"].append(val_loss)
        metrics["F-Score/Sum/Validation"].append(f_score_sum)
        metrics["F-Score/Avg/Validation"].append(f_score_avg)
        # if trial is not None:
        #     trial.report(f_score, epoch)
        torch.cuda.empty_cache()
        if scheduler is not None and isinstance(
            scheduler, optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(metrics[metric_to_track][-1])
            if current_lr > optimizer.state_dict()["param_groups"][0]["lr"]:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(f"Adjusting learning rate to {current_lr}")
                model.load_state_dict(best_model)
                optimizer = optim.RAdam(
                    model.parameters(), lr=current_lr, eps=1e-8, weight_decay=1e-5
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.2, patience=10
                )
        print(
            f"Epoch: {epoch + 1} "
            f"Loss: {train_loss * 100:.4f}\t "
            f"Val Loss: {val_loss * 100:.4f} F-Score: {f_score_sum * 100:.4f}/{f_score_avg * 100:.4f}"
        )
        last_improvement += 1
        if epoch == 0 or np.all(
            np.max(np.array(metrics[metric_to_track][:-1]) * directions[metric_to_track], axis=0) < np.array(
                metrics[metric_to_track][-1]) * directions[metric_to_track]):
            best_model = (
                ema_model.module if ema_model is not None else model
            ).state_dict()
            last_improvement = 0
        print(last_improvement)
        if f_score_sum > evaluation_settings.min_test_score and last_improvement == 0 and not is_unet:
            for test_loader in test_sets:
                identifier = test_loader.dataset.get_identifier()
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
        # elif (
        #         last_improvement >= 5
        #         and dataset_settings.annotation_settings.time_shift > 0.0
        # ):
        #     last_improvement = 0
        #     """
        #     optimizer = optim.RAdam(model.parameters(), lr=initial_lr, eps=1e-8, weight_decay=1e-4)
        #     best_score = f_score
        #     dataset.adjust_time_shift(max(time_shift - 0.01, 0))
        #     time_shift = dataset.time_shift
        #     print(f"Adjusting time shift to {time_shift}")
        #     """
        recent_scores = np.array(metrics["F-Score/Sum/Validation"][-3:])
        stuck = ((np.abs(recent_scores - recent_scores.mean(axis=0)) < 1e-4).all() and len(
            recent_scores) >= 3) or np.isnan(val_loss)
        if (
            early_stopping is not None
            and last_improvement >= training_settings.early_stopping or stuck
        ):
            if stuck:
                print("Detected stuck training")
            break
        # if trial is not None and trial.should_prune():
        #     raise optuna.TrialPruned()

    if best_model is not None:
        model.load_state_dict(best_model)

    if max(metrics["F-Score/Sum/Validation"]) > training_settings.min_save_score or is_unet:
        print("Saving model")
        dic = {
            "model": model.state_dict(),
            "model_settings": asdict(model_settings),
            "dataset_settings": asdict(dataset_settings),
            "training_settings": asdict(training_settings),
            "evaluation_settings": asdict(evaluation_settings),
        }
        dir_name = writer.get_logdir().split("/")[-1]
        torch.save(dic, f"./models/{dir_name}_{architecture}_{metrics['F-Score/Sum/Validation'][-1] * 100:.2f}.pt")

    hyperparameters = {
        **asdict(training_settings),
        **asdict(evaluation_settings),
        **asdict(dataset_settings),
        **asdict(model_settings),
        "seed": seed,
    }
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
