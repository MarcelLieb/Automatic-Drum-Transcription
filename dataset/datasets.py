import torch
from torch.utils.data import DataLoader

from dataset import get_dataloader
from dataset.RBMA13 import RBMA13
from dataset.MDB_Drums import MDBDrums
from dataset.A2MD import A2MD, get_splits as get_a2md_splits
from dataset.TMIDT import TMIDT
from dataset.generics import ConcatADTDataset
from settings import TrainingSettings, DatasetSettings


def get_dataset(
    training_settings: TrainingSettings,
    dataset_settings: DatasetSettings,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    list[DataLoader[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]]
]:
    path = "./data/a2md_public/"
    train_split, val_split, test_split = get_a2md_splits(
        training_settings.dataset_version, training_settings.splits, path
    )
    torch.multiprocessing.set_start_method("spawn", force=True)

    match training_settings.train_set:
        case "a2md":
            train = A2MD(
                split=train_split,
                settings=dataset_settings,
                path=path,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )
        case "tmidt" | "midi":
            train = TMIDT(
                path="./data/midi",
                settings=dataset_settings,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )
        case "all":
            a2md = A2MD(
                split=None,
                settings=dataset_settings,
                path=path,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )
            rbma = RBMA13(
                path="./data/rbma_13",
                settings=dataset_settings,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )
            mdb = MDBDrums(
                path="./data/MDB Drums",
                settings=dataset_settings,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )
            train = ConcatADTDataset(dataset_settings, [a2md, rbma, mdb])
        case _:
            train = A2MD(
                split=train_split,
                settings=dataset_settings,
                path=path,
                use_dataloader=True,
                is_train=True,
                segment=True,
            )

    val = A2MD(
        split=val_split,
        settings=dataset_settings,
        path=path,
        use_dataloader=True,
        is_train=False,
        segment=not training_settings.full_length_test,
    )
    test_sets = []
    for test_set in training_settings.test_sets:
        match test_set:
            case "RBMA":
                test_rbma = RBMA13(
                    path="./data/rbma_13",
                    settings=dataset_settings,
                    use_dataloader=True,
                    is_train=False,
                    segment=not training_settings.full_length_test,
                )
                loader = get_dataloader(
                    test_rbma,
                    training_settings.test_batch_size if training_settings.full_length_test else training_settings.batch_size,
                    training_settings.num_workers,
                    is_train=False,
                )
                test_sets.append(loader)
            case "MDB":
                test_mdb = MDBDrums(
                    path="./data/MDB Drums",
                    settings=dataset_settings,
                    use_dataloader=True,
                    is_train=False,
                    segment=not training_settings.full_length_test,
                )
                loader = get_dataloader(
                    test_mdb,
                    training_settings.test_batch_size if training_settings.full_length_test else training_settings.batch_size,
                    training_settings.num_workers,
                    is_train=False,
                )
                test_sets.append(loader)

    torch.multiprocessing.set_start_method("fork", force=True)
    dataloader_train = get_dataloader(
        train,
        training_settings.batch_size,
        training_settings.num_workers,
        is_train=True,
    )
    dataloader_val = get_dataloader(
        val,
        training_settings.test_batch_size if training_settings.full_length_test else training_settings.batch_size,
        training_settings.num_workers,
        is_train=False,
    )

    return dataloader_train, dataloader_val, test_sets
