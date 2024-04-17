import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import get_dataset
from dataset.A2MD import five_class_mapping
from model.SpecFlux import SpecFlux
from model.cnn import CNN
from model import ModelEmaV2


def step(
        model: nn.Module,
        criterion,
        optimizer: optim.Optimizer,
        audio_batch: torch.Tensor,
        lbl_batch: torch.Tensor,
        scheduler: optim.lr_scheduler.LRScheduler = None,
) -> (float, float):
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    model.train()
    optimizer.zero_grad()

    prediction = model(audio_batch)
    unfiltered = criterion(prediction, lbl_batch)
    no_silence = unfiltered * (lbl_batch != -1)
    filtered = no_silence.mean()
    loss = filtered

    over_detected = torch.sum(prediction[lbl_batch != -1].cpu().detach() > 0) - torch.sum(
        lbl_batch[lbl_batch != -1].cpu().detach())

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return filtered.item(), over_detected.item()


def evaluate(model: torch.nn.Module, dataloader: DataLoader, criterion, device) -> (float, float):
    """Evaluates the model on the specified dataset

    @return: The loss for the specified dataset. Return a float and not a PyTorch tensor
    """
    model.eval()
    total_loss = 0
    total_over_detected = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            audio, lbl = data
            audio = audio.to(device)
            lbl = lbl.to(device)
            prediction = model(audio)
            loss = criterion(prediction, lbl)
            loss = loss.mean()
            total_loss += loss.item()
            over_detected = torch.sum(prediction[lbl != -1].cpu().detach() > 0) - torch.sum(
                lbl[lbl != -1].cpu().detach())
            total_over_detected += over_detected.item()
    return total_loss / len(dataloader), total_over_detected / len(dataloader)


def main(
        learning_rate: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 4,
        ema: bool = False,
        scheduler: bool = True,
        n_mels: int = 84,
        early_stopping: bool = False,
        version: str = "L"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(
        f"Settings: Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}, EMA: {ema}, Scheduler: {scheduler}")

    num_workers = min(8, batch_size)

    mapping = five_class_mapping
    dataloader_train, dataloader_val, dataloader_test = get_dataset(
        batch_size, num_workers,
        splits=[0.8, 0.1, 0.1],
        version=version,
        time_shift=0.02, mapping=mapping,
        n_mels=n_mels,
    )

    model = CNN(n_mels=n_mels, n_classes=len(mapping) + 2)
    model.to(device)

    ema_model = ModelEmaV2(model, decay=0.8, device="cpu") if ema else None

    max_lr = learning_rate * 2
    initial_lr = max_lr / 25
    min_lr = initial_lr / 1e4

    optimizer = optim.RAdam(model.parameters(), lr=initial_lr, eps=1e-8, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader_train),
                                              epochs=epochs) if scheduler else None
    error = torch.nn.MSELoss(reduction="none")

    best_loss = float("inf")
    best_detection = float("inf")
    last_improvement = 0
    print("Starting Training")
    for epoch in range(epochs):
        total_loss = 0
        total_over = 0
        for _, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            audio, lbl = data
            audio = audio.to(device)
            lbl = lbl.to(device)

            loss, over = step(
                model=model,
                criterion=error,
                optimizer=optimizer,
                audio_batch=audio,
                lbl_batch=lbl,
                scheduler=scheduler
            )
            if ema_model is not None:
                ema_model.update(model)
            total_loss += loss
            total_over += over
        val_loss, val_over = evaluate(model if ema_model is None else ema_model.module, dataloader_val, error, device)
        print(
            f"Epoch: {epoch + 1} Loss: {total_loss / len(dataloader_train) * 100:.4f} Over: {total_over / len(dataloader_train): .0f}\t Val Loss: {val_loss * 100:.4f} Val Over: {val_over: .0f}")
        last_improvement += 1
        if val_loss <= best_loss:
            best_loss = val_loss
            last_improvement = 0
            if abs(val_over) <= abs(best_detection):
                best_detection = val_over
        if last_improvement > 10 and early_stopping:
            break

    return ema_model.module if ema_model is not None else model


if __name__ == '__main__':
    trained_model = main()
    trained_model.eval()
    trained_model = trained_model.cpu()
    torch.no_grad()
    if trained_model is not None and isinstance(trained_model, SpecFlux):
        weight = trained_model.feature_extractor.weight.cpu().detach().squeeze()
        for i in range(4):
            print(list(weight[i].numpy()))
        # print thresholds
        print(trained_model.drum_threshold.threshold.weight, trained_model.drum_threshold.threshold.bias)
        print(trained_model.snare_threshold.threshold.weight, trained_model.snare_threshold.threshold.bias)
        print(trained_model.hihat_threshold.threshold.weight, trained_model.hihat_threshold.threshold.bias)
        print(trained_model.onset_threshold.threshold.weight, trained_model.onset_threshold.threshold.bias)
