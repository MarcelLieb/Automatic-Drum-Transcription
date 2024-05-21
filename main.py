from dataclasses import asdict

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataset, ADTDataset
from dataset.mapping import DrumMapping
from evallib import peak_pick_max_mean, calculate_pr, calculate_f_score
from model.cnn import CNN
from model import ModelEmaV2
from settings import (
    AnnotationSettings,
    AudioProcessingSettings,
    TrainingSettings,
    CNNSettings,
    EvaluationSettings,
)


def step(
    model: nn.Module,
    criterion,
    optimizer: optim.Optimizer,
    audio_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
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
        full_context = no_silence[..., 9:]  # Receptive field if causal model is 9 frames
        filtered = full_context.mean()
        loss = filtered

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None and not isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return filtered.item()


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
):
    total_loss = 0
    for _, data in tqdm(
        enumerate(dataloader_train),
        total=len(dataloader_train),
        unit="mini-batch",
        smoothing=0.1,
        mininterval=1 / 2 * 60 / len(dataloader_train),
        desc="Training",
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
) -> (float, float):
    """Evaluates the model on the specified dataset

    @return: The loss for the specified dataset
    """
    model.eval()
    total_loss = 0
    device_str = str(device)
    device_str = "cuda" if "cuda" in device_str else "cpu"
    dataset: ADTDataset = dataloader.dataset
    mapping = dataset.mapping
    sample_rate, hop_size = dataset.sample_rate, dataset.hop_size
    predictions = []
    groundtruth = []
    with torch.no_grad():
        for data in tqdm(
            dataloader,
            total=len(dataloader),
            unit="mini-batch",
            smoothing=0.1,
            mininterval=1 / 2 * 60 / len(dataloader),
            desc="Evaluation",
        ):
            audio, lbl, gts = data
            audio = audio.to(device)
            lbl = lbl.to(device)
            with torch.autocast(device_type=device_str, dtype=torch.float16):
                prediction = model(audio)
                loss = criterion(prediction, lbl)
            peaks = peak_pick_max_mean(
                prediction.sigmoid().cpu().detach().float(),
                sample_rate,
                hop_size,
                evaluation_settings.peak_mean_range,
                evaluation_settings.peak_max_range,
            )
            predictions.extend(peaks)
            groundtruth.extend(gts)
            loss = loss.mean()
            total_loss += loss.item()
    precisions, recalls, thresholds, f, f_avg, best_thresholds = calculate_pr(
        predictions,
        groundtruth,
        onset_cooldown=evaluation_settings.onset_cooldown,
        detection_window=evaluation_settings.detect_tolerance,
        ignore_beats=evaluation_settings.ignore_beats,
    )
    f_scores = [
        calculate_f_score(precision, recall)
        for precision, recall in zip(precisions, recalls)
    ]

    loss = total_loss / len(dataloader)

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(f"F-Score/Sum/{tag}", f, global_step=epoch)
        tensorboard_writer.add_scalar(f"F-Score/Avg/{tag}", f_avg, global_step=epoch)
        tensorboard_writer.add_tensor(
            f"{tag}/Thresholds", best_thresholds, global_step=epoch
        )

        colors = (
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

        fig = plt.figure()
        for i in range(len(precisions)):
            plt.plot(
                recalls[i],
                precisions[i],
                color=colors[i],
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
                color=colors[i],
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
        tensorboard_writer.add_scalar(f"Loss/{tag}", loss, global_step=epoch)

    return total_loss / len(dataloader), f, f_avg


def main(
    training_settings: TrainingSettings = TrainingSettings(),
    audio_settings: AudioProcessingSettings = AudioProcessingSettings(),
    annotation_settings: AnnotationSettings = AnnotationSettings(),
    evaluation_settings: EvaluationSettings = EvaluationSettings(),
):
    cnn_settings = CNNSettings(
        n_classes=annotation_settings.n_classes, n_mels=audio_settings.n_mels
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(training_settings)
    print(evaluation_settings)
    print(audio_settings)
    print(annotation_settings)
    print(cnn_settings)

    loader_train, loader_val, loader_test_rbma, loader_test_mdb = get_dataset(
        training_settings, audio_settings, annotation_settings
    )

    cnnA_settings = CNNAttentionSettings(n_classes=annotation_settings.n_classes, n_mels=audio_settings.n_mels)
    model = CNNAttention(**asdict(cnnA_settings))
    model = CNN(**asdict(cnn_settings))
    model.to(device)

    ema_model = (
        ModelEmaV2(model, decay=0.999, device=device) if training_settings.ema else None
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

    optimizer = optim.RAdam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-5,  # decoupled_weight_decay=True,
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
    error = torch.nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    best_score = 0
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
        )
        torch.cuda.empty_cache()
        val_loss, f_score, avg_f_score = evaluate(
            epoch,
            model if ema_model is None else ema_model.module,
            loader_val,
            error,
            device,
            evaluation_settings,
            tensorboard_writer=writer,
            tag="Validation",
        )
        torch.cuda.empty_cache()
        if scheduler is not None and isinstance(
            scheduler, optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(f_score)
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
            f"Val Loss: {val_loss * 100:.4f} F-Score: {avg_f_score * 100:.4f}/{f_score * 100:.4f}"
        )
        if f_score > 0.60 and f_score >= best_score:
            test_loss, test_f_score, test_avg_f_score = evaluate(
                epoch,
                model if ema_model is None else ema_model.module,
                loader_test_rbma,
                error,
                device,
                evaluation_settings,
                tensorboard_writer=writer,
                tag="Test/RBMA",
            )
            torch.cuda.empty_cache()
            print(
                f"RBMA: Test Loss: {test_loss * 100:.4f} F-Score: {test_avg_f_score * 100:.4f}/{test_f_score * 100:.4f}"
            )
            test_loss, test_f_score, test_avg_f_score = evaluate(
                epoch,
                model if ema_model is None else ema_model.module,
                loader_test_mdb,
                error,
                device,
                evaluation_settings,
                tensorboard_writer=writer,
                tag="Test/MDB",
            )
            torch.cuda.empty_cache()
            print(
                f"MDB: Test Loss: {test_loss * 100:.4f} F-Score: {test_avg_f_score * 100:.4f}/{test_f_score * 100:.4f}"
            )
            if test_f_score > 0.75:
                break
        last_improvement += 1
        if best_score <= f_score:
            best_score = f_score
            best_model = (ema_model.module if ema_model is not None else model).state_dict()
            last_improvement = 0
        elif last_improvement >= 5 and annotation_settings.time_shift > 0.0:
            last_improvement = 0
            """
            optimizer = optim.RAdam(model.parameters(), lr=initial_lr, eps=1e-8, weight_decay=1e-4)
            best_score = f_score
            dataset.adjust_time_shift(max(time_shift - 0.01, 0))
            time_shift = dataset.time_shift
            print(f"Adjusting time shift to {time_shift}")
            """
        if val_loss <= best_loss:
            best_loss = val_loss
        if training_settings.early_stopping is not None and last_improvement >= training_settings.early_stopping:
            break

    hyperparameters = {
        **asdict(training_settings),
        **asdict(evaluation_settings),
        **asdict(audio_settings),
        **asdict(annotation_settings),
        **asdict(cnn_settings),
        "splits": str(training_settings.splits),
        "mapping": str(annotation_settings.mapping.name),
        "activation": cnn_settings.activation.__class__.__name__,
    }
    writer.add_hparams(hparam_dict=hyperparameters, metric_dict={"F-Score": best_score})
    print(f"Best F-score: {best_score * 100:.4f}")

    # Make sure everything is written to disk
    writer.flush()
    writer.close()

    return ema_model.module if ema_model is not None else model


if __name__ == "__main__":
    trained_model = main()
    trained_model.eval()
    trained_model = trained_model.cpu()
    torch.cuda.empty_cache()
