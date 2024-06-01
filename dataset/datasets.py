import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from dataset.RBMA13 import RBMA13
from dataset.MDB_Drums import MDBDrums
from dataset.A2MD import A2MD, get_splits as get_a2md_splits
from generics import ConcatADTDataset
from settings import TrainingSettings, DatasetSettings



def get_dataset(
    training_settings: TrainingSettings,
    audio_settings: AudioProcessingSettings,
    annotation_settings: AnnotationSettings,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
]:
    path = "./data/a2md_public/"
    train, val, test = get_splits(
        training_settings.dataset_version, training_settings.splits, path
    )
    torch.multiprocessing.set_start_method("spawn")
    train = A2MD(
        split=train,
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        path=path,
        use_dataloader=True,
        is_train=True,
    )
    val = A2MD(
        split=val,
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        path=path,
        use_dataloader=True,
        is_train=False,
    )
    test_rbma = RBMA13(
        root="./data/rbma_13",
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        use_dataloader=True,
        is_train=False,
    )
    test_mdb = MDBDrums(
        path="./data/MDB Drums",
        audio_settings=audio_settings,
        annotation_settings=annotation_settings,
        use_dataloader=True,
        is_train=False,
    )
    torch.multiprocessing.set_start_method("fork", force=True)
    dataloader_train = get_dataloader(
        train,
        training_settings.batch_size,
        training_settings.num_workers,
        is_train=True,
    )
    dataloader_val = get_dataloader(
        val, 1, training_settings.num_workers, is_train=False
    )
    dataloader_test_rbma = get_dataloader(
        test_rbma, 1, training_settings.num_workers, is_train=False
    )
    dataloader_test_mdb = get_dataloader(
        test_mdb, 1, training_settings.num_workers, is_train=False
    )
    return dataloader_train, dataloader_val, dataloader_test_rbma, dataloader_test_mdb