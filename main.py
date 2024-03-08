import torch
from torch import optim
from tqdm import tqdm

from SpecFlux import SpecFlux, ModelEmaV2
from dataset import rbma_13_path, get_dataloader

def step(
        model: torch.nn.Module,
        criterion,
        optimizer: optim.Optimizer,
        audio_batch: torch.Tensor,
        lbl_batch: torch.Tensor,
        scheduler: optim.lr_scheduler.LRScheduler = None,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    model.train()
    optimizer.zero_grad()

    prediction = model(audio_batch)
    unfiltered = criterion(prediction, lbl_batch)
    unfiltered[lbl_batch == -1] = 0
    filtered = unfiltered.mean()

    filtered.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return filtered.item()



def main(
        learning_rate: float = 4e-2,
        epochs: int = 25,
        batch_size: int = 4,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print("Initializing Spectral Flux")
    model = SpecFlux(
        sample_rate=48000,
        device=device,
        n_fft=2048,
        win_length=1024,
        window='hann',
        center=False,
        pad_mode='constant',
        eps=1e-10,
        lamb=0.1,
        n_mels=82,
    )
    model.to(device)

    ema_model = ModelEmaV2(model, decay=0.99, device=device)

    print("Initializing Dataloader")
    dataloader_train = get_dataloader(rbma_13_path, "all", batch_size, batch_size, 48000, 480, 2048, label_shift=-0.02)

    max_lr = learning_rate * 2
    initial_lr = max_lr / 25
    min_lr = initial_lr / 1e4

    optimizer = optim.RAdam(model.parameters(), lr=initial_lr, decoupled_weight_decay=True,
                            weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader_train),
                                              epochs=epochs)

    print("Starting Training")
    for epoch in range(epochs):
        total_loss = 0
        for i, data in tqdm(enumerate(dataloader_train)):
            audio, lbl = data
            audio = audio.to(device)
            lbl = lbl.to(device)

            loss = step(model, torch.nn.KLDivLoss(reduction="none"), optimizer, audio, lbl, scheduler)
            ema_model.update(model)
            total_loss += loss
        print(f"Epoch: {epoch + 1}\t Loss: {total_loss / len(dataloader_train) * 100:.4f}")


    return ema_model.module


if __name__ == '__main__':
    trained_model = main()
    print(list(trained_model.drum_mask.cpu().detach().numpy()))
    print(list(trained_model.snare_mask.cpu().detach().numpy()))
    print(list(trained_model.hihat_mask.cpu().detach().numpy()))
