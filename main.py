import torch
from torch import optim, nn
from tqdm import tqdm

from model.SpecFlux import SpecFlux
from model.cnn import CNN
from model.model import ModelEmaV2
from dataset.RBMA13 import rbma_13_path, get_dataloader


def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float, mask: torch.Tensor = None) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader
    @return: A tensor with the same shape as labels
    """

    negatives = ~labels
    flattened = torch.flatten(labels, start_dim=1)
    num_positives = torch.sum(flattened, dim=1)
    num_negatives = num_positives * neg_ratio
    max_samples = flattened.shape[1]
    # make sure negative examples are bounded by available examples
    num_negatives = torch.min(max_samples - num_positives, num_negatives)
    random = torch.rand_like(labels.float()) * negatives
    if mask is None:
        mask = torch.ones_like(labels)
    random = random * mask
    out = torch.zeros_like(labels)
    for i in range(labels.shape[0]):
        random_flat = torch.flatten(random[i])
        _, indices = torch.topk(random_flat, int(num_negatives[i]), dim=0)
        mask = torch.zeros_like(random_flat)
        mask[indices] = 1
        mask = torch.reshape(mask, labels[i].shape)
        mask = mask + labels[i]
        assert torch.sum(mask) == num_positives[i] + num_negatives[i]
        out[i] = mask
    assert out.shape == labels.shape
    out = out * (labels.bool() | mask.bool())
    return out


def hard_negative_mining(labels: torch.Tensor, predictions: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader
    @param predictions: The prediction tensor that is returned by your model
    @return: A tensor with the same shape as labels
    """
    negatives = ~labels
    flattened = torch.flatten(labels, start_dim=1)
    num_positives = torch.sum(flattened, dim=1)
    num_negatives = num_positives * neg_ratio
    max_samples = flattened.shape[1]
    # make sure negative examples are bounded by available examples
    num_negatives = torch.min(max_samples - num_positives, num_negatives)
    out = torch.zeros_like(labels)
    for i in range(labels.shape[0]):
        negative = ((predictions[i]) * negatives[0]).flatten()
        _, indices = torch.topk(negative, int(num_negatives[i]), largest=True, dim=0)
        mask = torch.zeros_like(negative)
        mask[indices] = 1
        mask = torch.reshape(mask, labels[i].shape)
        mask = mask + labels[i]
        assert torch.sum(mask) == num_positives[i] + num_negatives[i]
        out[i] = mask

    return out


def step(
        model: nn.Module,
        criterion,
        optimizer: optim.Optimizer,
        audio_batch: torch.Tensor,
        lbl_batch: torch.Tensor,
        negative_ratio: float,
        negative_mining: bool = False,
        scheduler: optim.lr_scheduler.LRScheduler = None,
) -> (float, float):
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    model.train()
    optimizer.zero_grad()

    prediction = model(audio_batch)
    unfiltered = criterion(prediction, lbl_batch)
    labels = (lbl_batch * (lbl_batch != -1)).bool()
    no_silence = unfiltered * (lbl_batch != -1)
    # mask = hard_negative_mining(labels, no_silence, negative_ratio)
    if negative_mining:
        mask = get_random_sampling_mask(labels, negative_ratio, mask=(lbl_batch != -1))
        no_silence = no_silence * mask
    filtered = no_silence.mean()
    loss = filtered

    over_detected = torch.sum(prediction[lbl_batch != -1].cpu().detach() > 0) - torch.sum(lbl_batch[lbl_batch != -1].cpu().detach())

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return filtered.item(), over_detected.item()


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion, device, neg_ratio: float) -> (float, float):
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
            labels = (lbl * (lbl != -1)).bool()
            # mask = get_random_sampling_mask(labels, neg_ratio // 16, mask=(lbl != -1))
            # loss = (loss * mask).mean()
            loss = loss.mean()
            total_loss += loss.item()
            over_detected = torch.sum(prediction[lbl != -1].cpu().detach() > 0) - torch.sum(
                lbl[lbl != -1].cpu().detach())
            total_over_detected += over_detected.item()
    return total_loss / len(dataloader), total_over_detected / len(dataloader)


def main(
        learning_rate: float = 5e-4,
        epochs: int =  90,
        batch_size: int = 4,
        negative_ratio = 50,
        ema: bool = False,
        scheduler: bool = True,
        n_mels: int = 82
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Settings: Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}, Negative Ratio: {negative_ratio}, EMA: {ema}, Scheduler: {scheduler}")

    print("Initializing Spectral Flux")
    model = SpecFlux(
        eps=1e-10,
        lamb=0.1,
        n_mels=82,
    )
    model = CNN(n_mels=n_mels, n_classes=4)
    model.to(device)
    num_workers = min(8, batch_size)

    ema_model = ModelEmaV2(model, decay=0.95, device=device) if ema else None

    print("Initializing Dataloader")
    dataloader_train = get_dataloader(rbma_13_path, "train_big", batch_size, num_workers, 48000, 480, 2048, label_shift=-0.01, pad_annotations=1, n_mels=n_mels)
    dataloader_val = get_dataloader(rbma_13_path, "test", min(batch_size, 9), min(num_workers, 9), 48000, 480, 2048, label_shift=-0.01, pad_annotations=1, n_mels=n_mels)

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
                negative_ratio=negative_ratio,
                scheduler=scheduler
            )
            if ema_model is not None:
                ema_model.update(model)
            total_loss += loss
            total_over += over
        val_loss, val_over = evaluate(model if ema_model is None else ema_model.module, dataloader_val, error, device, negative_ratio)
        print(f"Epoch: {epoch + 1} Loss: {total_loss / len(dataloader_train) * 100:.4f} Over: {total_over / len(dataloader_train): .0f}\t Val Loss: {val_loss * 100:.4f} Val Over: {val_over: .0f}")
        last_improvement += 1
        if val_loss <= best_loss:
            best_loss = val_loss
            last_improvement = 0
            if abs(val_over) <= abs(best_detection):
                best_detection = val_over
        if last_improvement > 10:
            break

    return ema_model.module if ema_model is not None else model


if __name__ == '__main__':
    trained_model = main()
    trained_model.eval()
    trained_model = trained_model.cpu()
    torch.no_grad()
    """
    # print weights of feature extractor
    weight = trained_model.feature_extractor.weight.cpu().detach().squeeze()
    for i in range(4):
        print(list(weight[i].numpy()))
    # print thresholds
    print(trained_model.drum_threshold.threshold.weight, trained_model.drum_threshold.threshold.bias)
    print(trained_model.snare_threshold.threshold.weight, trained_model.snare_threshold.threshold.bias)
    print(trained_model.hihat_threshold.threshold.weight, trained_model.hihat_threshold.threshold.bias)
    print(trained_model.onset_threshold.threshold.weight, trained_model.onset_threshold.threshold.bias)
    """
