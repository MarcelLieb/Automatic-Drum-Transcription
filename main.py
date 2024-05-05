from dataclasses import asdict

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataset, ADTDataset
from evallib import peak_pick_max_mean, calculate_pr
from model.SpecFlux import SpecFlux
from model.cnn import CNN
from model import ModelEmaV2
from settings import (
    AnnotationSettings,
    AudioProcessingSettings,
    TrainingSettings,
    CNNSettings,
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
        filtered = no_silence.mean()
        loss = filtered

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None and not isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return filtered.item()


def evaluate(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    device,
    ignore_beats: bool,
) -> (float, float):
    """Evaluates the model on the specified dataset

    @return: The loss for the specified dataset. Return a float and not a PyTorch tensor
    """
    model.eval()
    total_loss = 0
    device_str = str(device)
    device_str = "cuda" if "cuda" in device_str else "cpu"
    dataset: ADTDataset = dataloader.dataset
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
                prediction.cpu().detach().float(), sample_rate, hop_size
            )
            predictions.extend(peaks)
            groundtruth.extend(gts)
            loss = loss.mean()
            total_loss += loss.item()
    p, r, f, f_avg,thresholds = calculate_pr(
        predictions, groundtruth, ignore_beats=ignore_beats
    )
    print(f"Thresholds: {thresholds}")
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.xlim([0, 1])
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.savefig(f"./plots/pr_curve_{epoch}.png")
    plt.clf()
    return total_loss / len(dataloader), f, f_avg


def main(
    training_settings: TrainingSettings = TrainingSettings(),
    audio_settings: AudioProcessingSettings = AudioProcessingSettings(),
    annotation_settings: AnnotationSettings = AnnotationSettings(),
):
    cnn_settings = CNNSettings(
        n_classes=annotation_settings.n_classes, n_mels=audio_settings.n_mels
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(training_settings)
    print(audio_settings)
    print(annotation_settings)
    print(cnn_settings)

    dataloader_train, dataloader_val, dataloader_test = get_dataset(
        training_settings, audio_settings, annotation_settings
    )

    model = CNN(**asdict(cnn_settings))
    model.to(device)

    ema_model = (
        ModelEmaV2(model, decay=0.999, device=device) if training_settings.ema else None
    )
    best_model = None

    max_lr = training_settings.learning_rate * 2
    initial_lr = max_lr / 25
    _min_lr = initial_lr / 1e4
    initial_lr = (
        training_settings.learning_rate
        if not training_settings.scheduler
        else initial_lr
    )

    optimizer = optim.RAdam(
        model.parameters(), lr=initial_lr, eps=1e-8, weight_decay=1e-5
    )
    scheduler = (
        optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(dataloader_train),
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
        val_loss, f_score, avg_f_score = evaluate(
            epoch,
            model if ema_model is None else ema_model.module,
            dataloader_val,
            error,
            device,
            training_settings.ignore_beats,
        )
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
            f"Loss: {total_loss / len(dataloader_train) * 100:.4f}\t "
            f"Val Loss: {val_loss * 100:.4f} F-Score: {avg_f_score * 100:.4f}/{f_score * 100:.4f}"
        )
        if f_score > 0.60 and f_score >= best_score:
            test_loss, test_f_score, test_avg_f_score = evaluate(
                epoch,
                model if ema_model is None else ema_model.module,
                dataloader_test,
                error,
                device,
                training_settings.ignore_beats,
            )
            print(f"Test Loss: {test_loss * 100:.4f} F-Score: {avg_f_score * 100:.4f}/{test_f_score * 100:.4f}")
            if test_f_score > 0.73:
                break
        last_improvement += 1
        if best_score <= f_score:
            best_score = f_score
            best_model = model.state_dict()
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
        if last_improvement > 10 and training_settings.early_stopping:
            break

    print(f"Best F-score: {best_score * 100:.4f}")

    return ema_model.module if ema_model is not None else model


if __name__ == "__main__":
    trained_model = main()
    trained_model.eval()
    trained_model = trained_model.cpu()
    torch.no_grad()
    if trained_model is not None and isinstance(trained_model, SpecFlux):
        weight = trained_model.feature_extractor.weight.cpu().detach().squeeze()
        for i in range(4):
            print(list(weight[i].numpy()))
        # print thresholds
        print(
            trained_model.drum_threshold.threshold.weight,
            trained_model.drum_threshold.threshold.bias,
        )
        print(
            trained_model.snare_threshold.threshold.weight,
            trained_model.snare_threshold.threshold.bias,
        )
        print(
            trained_model.hihat_threshold.threshold.weight,
            trained_model.hihat_threshold.threshold.bias,
        )
        print(
            trained_model.onset_threshold.threshold.weight,
            trained_model.onset_threshold.threshold.bias,
        )
